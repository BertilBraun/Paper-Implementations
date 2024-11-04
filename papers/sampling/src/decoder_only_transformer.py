from torch import nn
from papers.attention_is_all_you_need.src.transformer import (
    MultiHeadAttention,
    PositionwiseFeedforward,
    LayerNorm,
    TransformerEmbedding,
    ModelConfig,
)


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(DecoderOnlyTransformer, self).__init__()
        self.embed = TransformerEmbedding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.N)])
        self.out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, mask):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
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

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x
