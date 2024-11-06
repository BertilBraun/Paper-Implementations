import math
import torch
from torch import nn

from papers.attention_is_all_you_need.src.transformer import ModelConfig, LayerNorm, PositionwiseFeedforward


class RoPETransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(RoPETransformer, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, src, trg, src_mask, tgt_mask):
        # The src is the source sequence (input to the encoder), and the trg is the target sequence which the decoder should have generated and will be the input to the decoder.
        # The src_mask is used to mask the padding tokens in the source sequence, ignoring them while allowing the model to train in batches of different length sequences.
        # The tgt_mask is used to mask the future tokens in the target sequence, preventing the model from cheating by looking at the future tokens.
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, tgt_mask)
        output = self.out(d_output)
        return output


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Encoder, self).__init__()
        self.embed = TransformerEmbedding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.N)])

    def forward(self, src, mask):
        x = self.embed(src)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Decoder, self).__init__()
        self.embed = TransformerEmbedding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.N)])

    def forward(self, trg, e_outputs, src_mask, tgt_mask):
        x = self.embed(trg)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, tgt_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.self_attn = MultiHeadAttention(config)
        self.ff = PositionwiseFeedforward(config)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.norm3 = LayerNorm(config.d_model)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.ff = PositionwiseFeedforward(config)

    def forward(self, x, e_outputs, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, e_outputs, e_outputs, src_mask)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MultiHeadAttention, self).__init__()
        self.d_model = config.d_model
        self.d_k = config.d_model // config.heads
        self.h = config.heads

        self.q_linear = nn.Linear(config.d_model, config.d_model)
        self.v_linear = nn.Linear(config.d_model, config.d_model)
        self.k_linear = nn.Linear(config.d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.attention = RoPEScaledDotProductAttention(config.max_len, self.d_k)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # Split the d_model into h heads
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        scores, _ = self.attention(q, k, v, mask)

        # Concatenate the heads
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(concat)


class RoPEScaledDotProductAttention(nn.Module):
    def __init__(self, max_len: int, d_k: int):
        super(RoPEScaledDotProductAttention, self).__init__()
        self.dim = d_k
        self.max_len = max_len

        # Initialize sine and cosine tables
        self.register_buffer('sin', torch.zeros(self.max_len, self.dim))
        self.register_buffer('cos', torch.zeros(self.max_len, self.dim))

        # Compute sine and cosine embeddings using nested loops
        for pos in range(self.max_len):
            for i in range(0, self.dim, 2):
                angle = pos / (10000 ** ((2 * (i // 2)) / self.dim))
                self.sin[pos, i] = self.sin[pos, i + 1] = math.sin(angle)
                self.cos[pos, i] = self.cos[pos, i + 1] = math.cos(angle)

    def _rotate_half(self, x):
        """
        Helper function to rotate half of the dimensions.
        >>> x1, x2, x3, x4, x5, x6 -> -x2, x1, -x4, x3, -x6, x5
        """
        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        x_rotated = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x_rotated

    def _apply_rotary_pos_emb(self, x):
        seq_len = x.size(-2)

        # Ensure that sequence length does not exceed maximum
        assert seq_len <= self.max_len, f'Sequence length {seq_len} exceeds maximum position embeddings {self.max_len}'

        # Get positional embeddings for the sequence length
        cos_emb = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, dim)
        sin_emb = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, dim)

        # Apply rotary position embedding
        return (x * cos_emb) + (self._rotate_half(x) * sin_emb)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for scaled dot-product attention with RoPE.
        Args:
            q, k, v: Tensors of shape (batch_size, num_heads, seq_len, dim)
            mask: Optional attention mask
        Returns:
            output: Tensor after attention, shape (batch_size, num_heads, seq_len, dim)
            attn_weights: Attention weights
        """
        # Apply rotary position embeddings to q and k
        q_rot = self._apply_rotary_pos_emb(q)
        k_rot = self._apply_rotary_pos_emb(k)

        # Scaled dot-product attention with rotated q and k
        attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(self.dim)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        return torch.matmul(attn_weights, v), attn_weights


class TransformerEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TransformerEmbedding, self).__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.embed(x))
