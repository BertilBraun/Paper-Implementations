import math
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int  # Number of tokens in the vocabulary
    d_model: int  # Internal dimension of the model
    N: int  # Number of layers
    heads: int  # Number of heads in the multi-head attention
    d_ff: int  # Dimension of the feedforward network
    max_len: int = 512  # Maximum length of the input sequence
    dropout: float = 0.1  # Dropout probability


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Transformer, self).__init__()
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

        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q)  # (batch_size, seq_len, d_model)
        k = self.k_linear(k)  # (batch_size, seq_len, d_model)
        v = self.v_linear(v)  # (batch_size, seq_len, d_model)

        # Split the d_model into h heads
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)

        scores, _ = self.attention(q, k, v, mask)  # (batch_size, h, seq_len, d_k)

        # Concatenate the heads
        concat = (
            scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )  # (batch_size, seq_len, d_model)

        return self.out(concat)


class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

        d_k = q.size(-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v), attn


class PositionwiseFeedforward(nn.Module):
    def __init__(self, config: ModelConfig):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)

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
    def __init__(self, config: ModelConfig):
        super(TransformerEmbedding, self).__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = PositionalEncoding(config.d_model, config.max_len)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        pos = self.pos_embed(x)
        emb = self.embed(x)
        return self.dropout(emb + pos)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                location = pos / 10000 ** (i / d_model)
                self.encoding[pos, i] = math.sin(location)
                self.encoding[pos, i + 1] = math.cos(location)

        # Add batch dimension. New shape: (1, max_len, d_model)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        sequence_len = x.size(1)
        return self.encoding[:, :sequence_len]
