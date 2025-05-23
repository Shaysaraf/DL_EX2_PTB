import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# Settings
batch_size = 20
seq_len = 15
bptt = seq_len  # For Transformer positional encoding
embed_size = 200
hidden_size = 200
num_layers = 2
num_epochs = 15
lstm_nodrop_lr = 3
lstm_drop_lr = 10
transformer_lr = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(file_path):
    tokens = []
    with open(file_path, 'r') as file:
        for line in file:
            line_tokens = line.strip().split()
            line_tokens.append('')
            tokens.extend(line_tokens)
    return tokens

train_tokens = preprocess_data('ptb.train.txt')
valid_tokens = preprocess_data('ptb.valid.txt')
test_tokens = preprocess_data('ptb.test.txt')

def build_vocab(token_list):
    counter = Counter(token_list)
    vocab = sorted(counter, key=counter.get, reverse=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word, len(word2idx)

word2idx, idx2word, vocab_size = build_vocab(train_tokens)

def encode_tokens(tokens, word2idx):
    return [word2idx.get(token, word2idx[""]) for token in tokens]

train_data = encode_tokens(train_tokens, word2idx)
valid_data = encode_tokens(valid_tokens, word2idx)
test_data = encode_tokens(test_tokens, word2idx)

def batchify(data, batch_size):
    nbatch = len(data) // batch_size
    data = data[:nbatch * batch_size]
    data = torch.tensor(data, dtype=torch.long).to(device)
    return data.view(batch_size, -1)

train_data = batchify(train_data, batch_size)
valid_data = batchify(valid_data, batch_size)
test_data = batchify(test_data, batch_size)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, seq_len, batch_size):
    seq_start = i * seq_len
    seq_end = seq_start + seq_len
    data = source[:, seq_start:seq_end]
    target = source[:, seq_start + 1:seq_end + 1]
    return data, target

def calculate_perplexity(loss):
    return math.exp(loss)

class ZarembaLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size))

def train_epoch(model, data, optimizer, loss_fn, batch_size, seq_len):
    model.train()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    num_batches = data.size(1) // seq_len
    for i in tqdm(range(num_batches)):
        inputs, targets = get_batch(data, i, seq_len, batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_batches, calculate_perplexity(total_loss / num_batches)

def evaluate(model, data, loss_fn, batch_size, seq_len):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    num_batches = data.size(1) // seq_len
    with torch.no_grad():
        for i in range(num_batches):
            inputs, targets = get_batch(data, i, seq_len, batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)
            loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()
    return total_loss / num_batches, calculate_perplexity(total_loss / num_batches)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, bptt, embed_size))
        encoder_layers = nn.TransformerEncoderLayer(embed_size, nhead=2, dim_feedforward=512, dropout=0.0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.decoder = nn.Linear(embed_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        output = self.decoder(output.reshape(-1, output.size(2)))
        return output, None

def train_epoch_transformer(model, data, optimizer, loss_fn, batch_size, seq_len):
    model.train()
    total_loss = 0
    num_batches = data.size(1) // seq_len
    for i in tqdm(range(num_batches)):
        inputs, targets = get_batch(data, i, seq_len, batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = loss_fn(outputs, targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_batches, calculate_perplexity(total_loss / num_batches)

def evaluate_transformer(model, data, loss_fn, batch_size, seq_len):
    model.eval()
    total_loss = 0
    num_batches = data.size(1) // seq_len
    with torch.no_grad():
        for i in range(num_batches):
            inputs, targets = get_batch(data, i, seq_len, batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = loss_fn(outputs, targets.reshape(-1))
            total_loss += loss.item()
    return total_loss / num_batches, calculate_perplexity(total_loss / num_batches)

def run_training(model, label, test_data, is_transformer=False, lr=None):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr) if not is_transformer else optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_ppls = []
    test_ppls = []
    for epoch in range(1, num_epochs + 1):
        if is_transformer:
            train_loss, train_ppl = train_epoch_transformer(model, train_data, optimizer, loss_fn, batch_size, seq_len)
            test_loss, test_ppl = evaluate_transformer(model, test_data, loss_fn, batch_size, seq_len)
        else:
            train_loss, train_ppl = train_epoch(model, train_data, optimizer, loss_fn, batch_size, seq_len)
            test_loss, test_ppl = evaluate(model, test_data, loss_fn, batch_size, seq_len)
        train_ppls.append(train_ppl)
        test_ppls.append(test_ppl)
        print(f"[{label}] Epoch {epoch}: Train PPL = {train_ppl:.2f}, Test PPL = {test_ppl:.2f}")
        if not is_transformer and epoch >= 7:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        if not is_transformer and epoch >= 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
    return train_ppls, test_ppls

def plot_perplexity(results, title, save_path):
    plt.figure(figsize=(10, 6))
    for label, train_ppls, test_ppls in results:
        plt.plot(range(1, len(train_ppls) + 1), train_ppls, label=f'{label} Train')
        plt.plot(range(1, len(test_ppls) + 1), test_ppls, linestyle='--', label=f'{label} Test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

# ========== Run Experiments ==========

results = []

# Train LSTM without dropout
#model1 = ZarembaLSTM(vocab_size, embed_size, hidden_size, num_layers, dropout=0.0)
#train1, test1 = run_training(model1, "LSTM (no dropout)", test_data, lr=lstm_nodrop_lr)
#results.append(("LSTM (no dropout)", train1, test1))

# Train LSTM with dropout
#model2 = ZarembaLSTM(vocab_size, embed_size, hidden_size, num_layers, dropout=0.3)
#train2, test2 = run_training(model2, "LSTM (dropout 0.3)", test_data, lr=lstm_drop_lr)
#results.append(("LSTM (dropout 0.5)", train2, test2))

# Transformer baseline
model3 = TransformerModel(vocab_size)
train3, test3 = run_training(model3, "Transformer", test_data, is_transformer=True, lr=transformer_lr)
results.append(("Transformer", train3, test3))

# Plot results
plot_perplexity(results, "Train vs Test Perplexity Comparison", "all_models_perplexity.png")
