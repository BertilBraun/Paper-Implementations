import math
import torch
from torch import nn


class RoPETransformer(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, max_len=512, dropout=0.1):
        super(RoPETransformer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, N, heads, d_ff, max_len, dropout)
        self.decoder = Decoder(vocab_size, d_model, N, heads, d_ff, max_len, dropout)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg, src_mask, tgt_mask):
        # The src is the source sequence (input to the encoder), and the trg is the target sequence which the decoder should have generated and will be the input to the decoder.
        # The src_mask is used to mask the padding tokens in the source sequence, ignoring them while allowing the model to train in batches of different length sequences.
        # The tgt_mask is used to mask the future tokens in the target sequence, preventing the model from cheating by looking at the future tokens.
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, tgt_mask)
        output = self.out(d_output)
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, max_len, dropout):
        super(Encoder, self).__init__()
        self.embed = TransformerEmbedding(d_model, vocab_size, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, max_len, dropout) for _ in range(N)])

    def forward(self, src, mask):
        x = self.embed(src)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, max_len, dropout):
        super(Decoder, self).__init__()
        self.embed = TransformerEmbedding(d_model, vocab_size, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, max_len, dropout) for _ in range(N)])

    def forward(self, trg, e_outputs, src_mask, tgt_mask):
        x = self.embed(trg)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, tgt_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, max_len, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(d_model, heads, max_len)
        self.ff = PositionwiseFeedforward(d_model, d_ff, dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, max_len, dropout):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(d_model, heads, max_len)
        self.cross_attn = MultiHeadAttention(d_model, heads, max_len)
        self.ff = PositionwiseFeedforward(d_model, d_ff, dropout)

    def forward(self, x, e_outputs, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, e_outputs, e_outputs, src_mask)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, max_len):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.attention = RoPEScaledDotProductAttention(max_len, self.d_k)

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
    def __init__(self, max_len, d_k):
        super(RoPEScaledDotProductAttention, self).__init__()
        self.dim = d_k  # Should be d_k (dimension per head)
        self.max_len = max_len

        # Initialize sine and cosine tables
        self.register_buffer('sin', torch.zeros(max_len, d_k))
        self.register_buffer('cos', torch.zeros(max_len, d_k))

        # Compute sine and cosine embeddings using nested loops
        for pos in range(max_len):
            for i in range(0, d_k, 2):
                angle = pos / (10000 ** ((2 * (i // 2)) / d_k))
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


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(std + self.eps) + self.bias


class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, dropout):
        super(TransformerEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.embed(x))
