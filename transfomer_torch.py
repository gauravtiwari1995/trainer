class CustomTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def fit(self, text):
        chars = sorted(set(text))
        self.char_to_id = {char: idx for idx, char in enumerate(chars)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}

    def encode(self, text):
        return [self.char_to_id[char] for char in text if char in self.char_to_id]

    def decode(self, token_ids):
        return ''.join(self.id_to_char[id_] for id_ in token_ids)

import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = tokenizer.encode(text)
        
        # Create input and target sequences
        self.inputs = []
        self.targets = []
        for i in range(len(self.data) - seq_len):
            self.inputs.append(self.data[i:i+seq_len])
            self.targets.append(self.data[i+1:i+1+seq_len])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_ff, dropout), num_layers
        )
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        return self.decoder(x)

# Prepare data
text = "This is a simple example to demonstrate a Transformer from scratch."
tokenizer = CustomTokenizer()
tokenizer.fit(text)
seq_len = 10

dataset = TextDataset(text, tokenizer, seq_len)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
vocab_size = len(tokenizer.char_to_id)
d_model = 64
num_heads = 4
num_layers = 2
dim_ff = 128
max_seq_len = seq_len

model = Transformer(vocab_size, d_model, num_heads, num_layers, dim_ff, max_seq_len)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        # Inputs and targets
        inputs = inputs.transpose(0, 1)  # Transformer expects [seq_len, batch_size]
        targets = targets.transpose(0, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

def generate_text(model, tokenizer, start_text, max_length):
    model.eval()
    generated = tokenizer.encode(start_text)
    for _ in range(max_length):
        input_tensor = torch.tensor(generated).unsqueeze(1)  # [seq_len, batch_size]
        with torch.no_grad():
            output = model(input_tensor)
            next_token = output.argmax(dim=-1)[-1, 0].item()  # Get most likely next token
            generated.append(next_token)
            if tokenizer.id_to_char[next_token] == '.':
                break
    return tokenizer.decode(generated)

# Test generation
start_text = "This is"
print(generate_text(model, tokenizer, start_text, max_length=50))
