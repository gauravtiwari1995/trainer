import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Constants
EMBED_DIM = 256  # Embedding dimension
NUM_HEADS = 8  # Number of attention heads
NUM_LAYERS = 4  # Number of transformer layers
SEQ_LEN = 50  # Sequence length
BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 0.001  # Learning rate
EPOCHS = 10  # Number of epochs

# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.char2idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char2idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx2char[idx] for idx in indices])

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=SEQ_LEN):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# GPT Model
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 0, 2)
        logits = self.fc_out(x)
        return logits

# Dataset
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.data = tokenizer.encode(text)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)

# Train Model
def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Generate Text
def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    generated = input_ids.tolist()[0]
    for _ in range(max_length):
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        generated.append(next_token)
        input_ids = torch.tensor([generated[-SEQ_LEN:]]).to(device)
    return tokenizer.decode(generated)

# Main
if __name__ == "__main__":
    # Load data
    #with open('your_text_data.txt', 'r') as file:
     #   raw_text = file.read()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_text = "Gaurav birthday is on 01 Nov. His hometown is in Azamgarh. He loves to drive and owns a BMW X1. He frequently visits different places and is very fond of branded clothing."
    tokenizer = CharTokenizer(raw_text)
    dataset = TextDataset(raw_text, tokenizer, seq_len=SEQ_LEN)
    print("----------------------")
    print(dataset.seq_len)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, optimizer, criterion
    model = GPT(tokenizer.vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Train
    train_model(model, dataloader, optimizer, criterion, EPOCHS)

    # Save the model
    torch.save(model.state_dict(), 'gpt_model.pth')

    # Generate text
    prompt = "hometown"
    print("Generated Text:")
    print(generate_text(model, tokenizer, prompt, max_length=200))
