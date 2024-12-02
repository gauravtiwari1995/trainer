import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Initialize AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Add a padding token if it doesn't already exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden_size, max_seq_len):
        super(TransformerModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_size))
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=ff_hidden_size
            ),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.token_embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.encoder_layers(x)
        x = self.output_layer(x)
        return x

# Dataset and Training Code (similar to earlier example)
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_seq_len):
        self.text = text
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = self.tokenize_text()

    def tokenize_text(self):
        tokens = self.tokenizer(self.text, truncation=True, padding=True, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze().tolist()

        # Handle short text
        if len(input_ids) < self.max_seq_len:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))
            return [input_ids]

        return [
            input_ids[i:i + self.max_seq_len]
            for i in range(0, len(input_ids) - self.max_seq_len + 1, self.max_seq_len)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        return torch.tensor(input_ids[:-1]), torch.tensor(input_ids[1:])

# Training Loop
def train(model, dataloader, optimizer, criterion, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Debugging: Ensure inputs and targets are valid
            print(f"Batch {batch_idx + 1}: Inputs {inputs.shape}, Targets {targets.shape}")
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.permute(0, 2, 1)  # Rearrange for loss calculation
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item()}")

            # Debugging: Stop after a few batches for testing
            if batch_idx > 5:  # Remove or increase for actual training
                break

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss / len(dataloader)}")

# Query Function
def query(model, tokenizer, prompt, max_len=20):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids

    for _ in range(max_len):  # Limit generation to `max_len` tokens
        outputs = model(generated)
        next_token = torch.argmax(outputs[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        
        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0])

# Main Script
if __name__ == "__main__":
    # Parameters
    vocab_size = 30000  # Modify based on your tokenizer
    embed_size = 128
    num_heads = 4
    num_layers = 2
    ff_hidden_size = 512
    max_seq_len = 50
    batch_size = 16
    epochs = 10
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example text data
    text_data = (
        "This is a sample dataset. Add more text for training. "
        "It should be long enough to ensure there are multiple samples "
        "based on the sequence length."
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Dataset and DataLoader
    dataset = TextDataset(text_data, tokenizer, max_seq_len)
    print(f"Number of Samples in Dataset: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Number of Batches in DataLoader: {len(dataloader)}")

    # Model, Loss, and Optimizer
    model = TransformerModel(vocab_size, embed_size, num_heads, num_layers, ff_hidden_size, max_seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, dataloader, optimizer, criterion, epochs, device)

    # Query the model
    prompt = "This is a"
    response = query(model, tokenizer, prompt)
    print("Generated Text:", response)
