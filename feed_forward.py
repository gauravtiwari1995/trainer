import torch
import torch.nn as nn
import math

from multi_head_attention import MultiHeadAttention
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        attention_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attention_out)
        feed_forward_out = self.feed_forward(x)
        return self.norm2(x + feed_forward_out)