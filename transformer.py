import torch
import torch.nn as nn
import math

from feed_forward import EncoderLayer
from positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)