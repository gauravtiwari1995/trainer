import torch
import torch.nn as nn
import math
from scaled_dot_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0
        self.dim_per_head = embedding_dim // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.query(query).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Apply attention
        out, _ = self.attention(query, key, value, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        return self.fc(out)