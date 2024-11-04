from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    N: int
    heads: int
    d_ff: int
    max_batch_size: int = 64
    max_len: int = 512
    dropout: float = 0.1


class Transformer(nn.Module):
    # Decoder only model with kv-cache (only 1 token at a time)

    def __init__(self, config: ModelConfig):
        super(Transformer, self).__init__()

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = PositionalEncoding(config.d_model, config.max_len)

        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.N)])
        self.out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, position, mask):
        assert x.size(1) == 1, 'Only 1 token at a time'

        x = self.embed_dropout(self.embed(x) + self.pos_embed(position))
        for layer in self.layers:
            x = layer(x, position, mask)
        return self.out(x)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.self_attn = MultiHeadAttention(config)
        self.ff = PositionwiseFeedforward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x, position, mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, position, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
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

        self.k_cache = torch.zeros((config.max_batch_size, config.max_len, self.d_model))
        self.v_cache = torch.zeros((config.max_batch_size, config.max_len, self.d_model))

    def forward(self, x, position, mask=None):
        assert x.size(1) == 1, 'Only 1 token at a time'

        batch_size = x.size(0)

        xq: torch.Tensor = self.q_linear(x)  # (batch_size, 1, d_model)
        xk: torch.Tensor = self.k_linear(x)  # (batch_size, 1, d_model)
        xv: torch.Tensor = self.v_linear(x)  # (batch_size, 1, d_model)

        # Update cache
        self.k_cache[:batch_size, position] = xk
        self.v_cache[:batch_size, position] = xv

        keys = self.k_cache[:batch_size, : position + 1]
        values = self.v_cache[:batch_size, : position + 1]

        # Split the d_model into h heads
        q = xq.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        k = keys.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)
        v = values.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch_size, h, seq_len, d_k)

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

    def forward(self, position: int):
        return self.encoding[:, position, :].unsqueeze(1)  # (1, 1, d_model)
