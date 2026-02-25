"""
CoreGPT

Small transformer-based language model implementation for learning and experimentation.
"""

import os
import random
import math
from typing import List, Tuple
from config import Config


# =========================================================
# RANDOM SEED (so results are repeatable between runs)
# =========================================================

random.seed(42)


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset(path: str) -> str:
    """
    Open the dataset file and return its contents as a string.
    Raises an error if the file path is invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"[Data] Loaded dataset | characters={len(text)}")
    return text


# =========================================================
# TOKENIZER (character level)
# =========================================================

class CharTokenizer:
    """
    Converts characters to numeric IDs and back.
    Any unknown character maps to a special <UNK> token.
    """

    def __init__(self, text: str):
        unique_chars = sorted(set(text))
        unique_chars.append("<UNK>")

        self.vocab_size = len(unique_chars)
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for i, ch in enumerate(unique_chars)}
        self.unk_token = self.stoi["<UNK>"]

    def encode(self, text: str) -> List[int]:
        """Turn text into a list of token IDs."""
        return [self.stoi.get(c, self.unk_token) for c in text]

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back into readable text."""
        chars = []
        for t in tokens:
            ch = self.itos.get(t, "")
            if ch != "<UNK>":
                chars.append(ch)
        return "".join(chars)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

def train_val_split(data: List[int], split_ratio: float = 0.9):
    """
    Split tokenized data into training and validation parts.
    """
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"[Split] train_tokens={len(train_data)} | val_tokens={len(val_data)}")
    return train_data, val_data


# =========================================================
# BATCH SAMPLING
# =========================================================

def get_batch(
    data: List[int],
    block_size: int,
    batch_size: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Randomly sample chunks of tokens for training.
    Y is just X shifted by one position.
    """
    if len(data) <= block_size:
        raise ValueError("Dataset too small for chosen block size")

    x_batch, y_batch = [], []

    for _ in range(batch_size):
        idx = random.randint(0, len(data) - block_size - 1)
        x_batch.append(data[idx:idx + block_size])
        y_batch.append(data[idx + 1:idx + block_size + 1])
    return x_batch, y_batch


# =========================================================
# BASIC MATH HELPERS
# =========================================================

def random_matrix(rows, cols):
    """Initialize a small random matrix."""
    return [[random.uniform(-0.02, 0.02) for _ in range(cols)] for _ in range(rows)]


def matmul(a, b):
    """Simple matrix multiplication (pure Python)."""
    result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result


# =========================================================
# SOFTMAX
# =========================================================

def softmax(vec):
    """Numerically stable softmax."""
    max_val = max(vec)
    exps = [math.exp(v - max_val) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


# =========================================================
# ATTENTION HELPERS
# =========================================================

def transpose(m):
    return list(map(list, zip(*m)))


def apply_causal_mask(scores):
    """
    Prevent tokens from attending to future positions.
    """
    size = len(scores)
    for i in range(size):
        for j in range(i + 1, size):
            scores[i][j] = -1e9
    return scores


# =========================================================
# LAYER NORMALIZATION
# =========================================================

class LayerNorm:
    """
    Normalizes each token’s feature vector.
    """

    def __init__(self, dim, eps=1e-5):
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim
        self.eps = eps

    def forward(self, x):
        normalized = []
        for row in x:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            norm_row = [(v - mean) / math.sqrt(var + self.eps) for v in row]
            norm_row = [self.gamma[i] * norm_row[i] + self.beta[i] for i in range(len(row))]
            normalized.append(norm_row)
        return normalized


# =========================================================
# POSITION EMBEDDING
# =========================================================

class PositionalEmbedding:
    """
    Learns a vector for each position in the sequence.
    """

    def __init__(self, block_size, embed_dim):
        self.weight = random_matrix(block_size, embed_dim)

    def forward(self, length):
        return self.weight[:length]


# =========================================================
# MULTI-HEAD ATTENTION
# =========================================================

class MultiHeadAttention:
    """
    Runs several attention heads in parallel, then merges them.
    """

    def __init__(self, embed_dim, num_heads=2):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.heads = [SelfAttention(self.head_dim) for _ in range(num_heads)]
        self.proj = Linear(embed_dim, embed_dim)

    def split_heads(self, x):
        split = []
        for head in range(self.num_heads):
            start = head * self.head_dim
            end = start + self.head_dim
            split.append([row[start:end] for row in x])
        return split

    def concat_heads(self, heads):
        concat = []
        for i in range(len(heads[0])):
            row = []
            for h in heads:
                row.extend(h[i])
            concat.append(row)
        return concat

    def forward(self, x):
        heads_in = self.split_heads(x)
        heads_out = [h.forward(heads_in[i]) for i, h in enumerate(self.heads)]
        concat = self.concat_heads(heads_out)
        return self.proj.forward(concat)


# =========================================================
# SINGLE ATTENTION HEAD
# =========================================================

class SelfAttention:
    """
    Standard scaled dot-product attention.
    """

    def __init__(self, embed_dim):
        self.query = Linear(embed_dim, embed_dim)
        self.key = Linear(embed_dim, embed_dim)
        self.value = Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        Q = self.query.forward(x)
        K = self.key.forward(x)
        V = self.value.forward(x)

        scores = matmul(Q, transpose(K))

        for i in range(len(scores)):
            for j in range(len(scores[i])):
                scores[i][j] /= self.scale

        scores = apply_causal_mask(scores)
        weights = [softmax(row) for row in scores]
        out = matmul(weights, V)

        return out


# =========================================================
# CORE LAYERS
# =========================================================

class Embedding:
    """Lookup table that maps token IDs to vectors."""

    def __init__(self, vocab_size, embed_dim):
        self.weight = random_matrix(vocab_size, embed_dim)

    def forward(self, tokens):
        return [self.weight[t] for t in tokens]


class Linear:
    """Fully connected layer."""

    def __init__(self, in_dim, out_dim):
        self.weight = random_matrix(in_dim, out_dim)
        self.bias = [0.0] * out_dim

    def forward(self, x):
        out = matmul(x, self.weight)
        for row in out:
            for i in range(len(row)):
                row[i] += self.bias[i]
        return out


def relu(x):
    """ReLU activation."""
    return [[max(0.0, v) for v in row] for row in x]


# =========================================================
# FEED-FORWARD NETWORK
# =========================================================

class FeedForward:
    """
    Expands dimension → applies non-linearity → projects back.
    """

    def __init__(self, embed_dim):
        hidden_dim = embed_dim * 4
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = relu(x)
        x = self.fc2.forward(x)
        return x


# =========================================================
# TRANSFORMER BLOCK
# =========================================================

class TransformerBlock:
    """
    One transformer layer:
    LayerNorm → Attention → Residual
    LayerNorm → FFN → Residual
    """

    def __init__(self):
        self.ln1 = LayerNorm(Config.embed_dim)
        self.attn = MultiHeadAttention(Config.embed_dim, Config.num_heads)

        self.ln2 = LayerNorm(Config.embed_dim)
        self.ffn = FeedForward(Config.embed_dim)

    def forward(self, x):
        # attention with residual
        x_norm = self.ln1.forward(x)
        attn_out = self.attn.forward(x_norm)
        x = [[x[i][j] + attn_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]

        # ffn with residual
        x_norm = self.ln2.forward(x)
        ffn_out = self.ffn.forward(x_norm)
        x = [[x[i][j] + ffn_out[i][j] for j in range(len(x[i]))] for i in range(len(x))]

        return x


# =========================================================
# MODEL DEFINITION
# =========================================================

class TinyLanguageModel:
    """
    Small GPT-style language model built from scratch.
    """

    def __init__(self, vocab_size):
        self.token_embedding = Embedding(vocab_size, Config.embed_dim)

        # stack multiple transformer blocks
        self.blocks = [TransformerBlock() for _ in range(Config.num_layers)]

        self.linear = Linear(Config.embed_dim, vocab_size)

    def forward(self, tokens):
        x = self.token_embedding.forward(tokens)

        for block in self.blocks:
            x = block.forward(x)

        logits = self.linear.forward(x)
        return logits

    def _forward_with_hidden(self, tokens):
        x = self.token_embedding.forward(tokens)

        for block in self.blocks:
            x = block.forward(x)

        logits = self.linear.forward(x)
        return logits, x


# =========================================================
# LOSS FUNCTIONS
# =========================================================

def cross_entropy_loss(logits, targets):
    """Average cross-entropy across sequence."""
    loss = 0.0
    for i in range(len(logits)):
        probs = softmax(logits[i])
        loss -= math.log(probs[targets[i]] + 1e-9)
    return loss / len(logits)


def _loss_with_probs(logits, targets):
    """
    Internal helper used for training.
    Returns loss + probability cache.
    """
    loss = 0.0
    probs_cache = []

    for i in range(len(logits)):
        probs = softmax(logits[i])
        probs_cache.append(probs)
        loss -= math.log(probs[targets[i]] + 1e-9)

    return loss / len(logits), probs_cache


# =========================================================
# TRAINING STEP
# =========================================================

def train_step(model, x, y, lr=1e-2):
    """
    Performs one SGD update (only output layer for simplicity).
    """
    logits, hidden = model._forward_with_hidden(x)
    loss, probs = _loss_with_probs(logits, y)

    for t in range(len(x)):
        for j in range(len(model.linear.bias)):
            grad = probs[t][j]
            if j == y[t]:
                grad -= 1.0

            model.linear.bias[j] -= lr * grad

            for k in range(len(hidden[t])):
                model.linear.weight[k][j] -= lr * grad * hidden[t][k]

    return loss


# =========================================================
# TRAIN LOOP
# =========================================================

def evaluate(model, data, tokenizer=None):
    """Compute validation loss on random batch."""
    xb, yb = get_batch(data, Config.block_size, 1)
    logits = model.forward(xb[0])
    return cross_entropy_loss(logits, yb[0])


def train_model(model, train_data, val_data, epochs=5, steps_per_epoch=200):
    """
    Basic epoch loop with validation monitoring.
    """
    for epoch in range(epochs):

        total_loss = 0.0

        for step in range(steps_per_epoch):
            xb, yb = get_batch(train_data, Config.block_size, 1)
            loss = train_step(model, xb[0], yb[0])
            total_loss += loss

        avg_train_loss = total_loss / steps_per_epoch
        val_loss = evaluate(model, val_data)

        print(f"[Epoch {epoch+1}] train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f}")


# =========================================================
# TEXT GENERATION
# =========================================================

def sample_next_token(logits, temperature=1.0):
    """Sample from probability distribution with temperature."""
    scaled = [v / temperature for v in logits]
    probs = softmax(scaled)

    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return len(probs) - 1


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0):
    """
    Autoregressive generation loop.
    """
    tokens = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        context = tokens[-Config.block_size:]
        logits = model.forward(context)
        next_token = sample_next_token(logits[-1], temperature)
        tokens.append(next_token)

    return tokenizer.decode(tokens)


# =========================================================
# ENTRY POINT
# =========================================================

def main(verbose: bool = True):
    print("\n=== CoreGPT Phase 13 ===\n")

    text = load_dataset(Config.dataset_path)
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    train_data, val_data = train_val_split(data)

    model = TinyLanguageModel(tokenizer.vocab_size)

    # run training
    train_model(
        model,
        train_data,
        val_data,
        epochs=Config.epochs,
        steps_per_epoch=200
    )

    print("\n--- Generated Text ---\n")
    output = generate_text(model, tokenizer, "CoreGPT ", 200, temperature=0.6)
    print(output)


if __name__ == "__main__":
    main()