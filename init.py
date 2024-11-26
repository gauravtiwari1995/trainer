print("Hello")
# Example dataset
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
            if word not in vocab:
                vocab[word] = len(vocab)

# Encode sentences
def encode_sentence(sentence):
    return [vocab["<SOS>"]] + [vocab[word] for word in sentence.lower().split()] + [vocab["<EOS>"]]

encoded_data = [(encode_sentence(pair["input"]), encode_sentence(pair["output"])) for pair in dataset]