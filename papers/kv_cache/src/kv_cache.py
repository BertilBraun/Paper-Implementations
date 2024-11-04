import torch
from torch import nn

from papers.attention_is_all_you_need.src.transformer import (
    ModelConfig,
    PositionalEncoding,
    PositionwiseFeedforward,
    LayerNorm,
    ScaledDotProductAttention,
)


MAX_BATCH_SIZE = 128


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

        x = self.embed_dropout(self.embed(x) + self.pos_embed(x)[:, position, :])
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
        self.ff = PositionwiseFeedforward(config)

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

        self.k_cache = torch.zeros((MAX_BATCH_SIZE, config.max_len, self.d_model))
        self.v_cache = torch.zeros((MAX_BATCH_SIZE, config.max_len, self.d_model))

    def forward(self, x, position, mask=None):
        assert x.size(0) <= MAX_BATCH_SIZE, 'Batch size should be less than MAX_BATCH_SIZE'
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
