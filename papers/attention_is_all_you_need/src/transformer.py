import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, N, heads, d_ff, dropout)
        self.decoder = Decoder(vocab_size, d_model, N, heads, d_ff, dropout)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg, src_mask, tgt_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, tgt_mask)
        output = self.out(d_output)
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.embed = TransformerEmbedding(d_model, vocab_size)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])

    def forward(self, src, mask):
        x = self.embed(src)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.embed = TransformerEmbedding(d_model, vocab_size)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])

    def forward(self, trg, e_outputs, src_mask, tgt_mask):
        x = self.embed(trg)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, tgt_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads)
        self.ff = PositionwiseFeedforward(d_model, d_ff)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn1 = MultiHeadAttention(d_model, heads)
        self.attn2 = MultiHeadAttention(d_model, heads)
        self.ff = PositionwiseFeedforward(d_model, d_ff)

    def forward(self, x, e_outputs, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout1(self.attn1(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.attn2(x, e_outputs, e_outputs, src_mask)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        scores, _ = self.attention(q, k, v, mask)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(concat)


class ScaleDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(attn, dim=-1)
        return attn @ v, attn


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
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
    def __init__(self, d_model, vocab_size, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pos_embed = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pos = self.pos_embed(x)
        emb = self.embed(x)
        return self.dropout(emb + pos)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return self.encoding[:, : x.size(1)]