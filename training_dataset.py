import torch.optim as optim
import torch
import torch.nn as nn
import math

from transformer import TransformerEncoder
# Hyperparameters

dataset = [
    {"input": "What is AI?", "output": "AI is the simulation of human intelligence by machines."},
    {"input": "Explain Python.", "output": "Python is a versatile programming language used for various purposes."},
    {"input": "What is deep learning?", "output": "Deep learning is a subset of machine learning focused on neural networks."}
]

# Tokenizer (basic, replace with your own tokenization logic or use a library like NLTK)
vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
for pair in dataset:
    for sentence in [pair["input"], pair["output"]]:
        for word in sentence.lower().split():
            print(word)
            if word not in vocab:
                vocab[word] = len(vocab)

# Encode sentences
def encode_sentence(sentence):
    return [vocab["<SOS>"]] + [vocab[word] for word in sentence.lower().split()] + [vocab["<EOS>"]]

encoded_data = [(encode_sentence(pair["input"]), encode_sentence(pair["output"])) for pair in dataset]

embedding_dim = 64
num_heads = 4
hidden_dim = 256
num_layers = 2
num_epochs = 10
learning_rate = 0.001

# Initialize model, loss, and optimizer
model = TransformerEncoder(len(vocab), embedding_dim, num_heads, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for input_seq, target_seq in encoded_data:
        input_seq = torch.tensor(input_seq).unsqueeze(0)
        target_seq = torch.tensor(target_seq).unsqueeze(0)

        optimizer.zero_grad()
        output = model(input_seq)
        
        loss = criterion(output.view(-1, len(vocab)), target_seq.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")