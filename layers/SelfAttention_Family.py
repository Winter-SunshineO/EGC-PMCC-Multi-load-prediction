import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        _, _, _, d_k = queries.shape
        scale = self.scale or 1.0 / math.sqrt(d_k)
        scores = torch.einsum("blhd,bshd->bhls", queries, keys) * scale

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -torch.inf)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        if self.output_attention:
            return out, attn
        return out, None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x, attn_mask=None):
        bsz, seq_len, _ = x.shape
        _, key_len, _ = x.shape

        queries = self.query_projection(x).view(bsz, seq_len, self.n_heads, -1)
        keys = self.key_projection(x).view(bsz, key_len, self.n_heads, -1)
        values = self.value_projection(x).view(bsz, key_len, self.n_heads, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask=attn_mask)
        out = out.reshape(bsz, seq_len, -1)
        return self.out_projection(out), attn
