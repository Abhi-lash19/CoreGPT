"""
CoreGPT

Phase 2 — Data Pipeline

Includes:
- Character tokenizer
- Vocabulary creation
- Encoding / decoding
- Train/validation split
- Random batch sampling
- Deterministic behavior (seed)
"""

import os
import random
import math
from typing import List, Tuple
from config import Config


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(42)


# =========================================================
# DATASET
# =========================================================

def load_dataset(path: str) -> str:
    """
    Reads dataset file and returns text.

    Args:
        path (str): Path to dataset file

    Returns:
        str: Raw text content
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"[Data] Loaded dataset | characters={len(text)}")
    return text


# =========================================================
# TOKENIZER
# =========================================================

class CharTokenizer:
    """
    Character-level tokenizer.

    Creates mapping:
        char -> index
        index -> char
    """

    def __init__(self, text: str):
        unique_chars = sorted(set(text))
        self.vocab_size = len(unique_chars)

        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for i, ch in enumerate(unique_chars)}

    def encode(self, text: str) -> List[int]:
        """Convert text string to token indices."""
        return [self.stoi[c] for c in text]

    def decode(self, tokens: List[int]) -> str:
        """Convert token indices back to string."""
        return "".join(self.itos[t] for t in tokens)


# =========================================================
# SPLIT + BATCH
# =========================================================

def train_val_split(data: List[int], split_ratio: float = 0.9):
    """
    Split token list into train and validation sets.
    """
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"[Split] train_tokens={len(train_data)} | val_tokens={len(val_data)}")
    return train_data, val_data


# =========================================================
# BATCH SAMPLER
# =========================================================

def get_batch(
    data: List[int],
    block_size: int,
    batch_size: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Sample random training batch.

    Returns:
        X: input tokens
        Y: shifted target tokens
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
# MATH UTILITIES
# =========================================================

def random_matrix(rows, cols):
    return [[random.uniform(-0.02, 0.02) for _ in range(cols)] for _ in range(rows)]


def matmul(a, b):
    result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result


# =========================================================
# PHASE 4 — ATTENTION UTILITIES
# =========================================================

def softmax(vec):
    max_val = max(vec)
    exps = [math.exp(v - max_val) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]


# =========================================================
# ATTENTION
# =========================================================

def transpose(m):
    return list(map(list, zip(*m)))


def apply_causal_mask(scores):
    size = len(scores)
    for i in range(size):
        for j in range(i + 1, size):
            scores[i][j] = -1e9
    return scores


# =========================================================
# PHASE 4 — ATTENTION LAYER
# =========================================================

class SelfAttention:
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
# LAYERS
# =========================================================

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.weight = random_matrix(vocab_size, embed_dim)

    def forward(self, tokens):
        return [self.weight[t] for t in tokens]


class Linear:
    def __init__(self, in_dim, out_dim):
        self.weight = random_matrix(in_dim, out_dim)
        self.bias = [0.0] * out_dim

    def forward(self, x):
        out = matmul(x, self.weight)
        for row in out:
            for i in range(len(row)):
                row[i] += self.bias[i]
        return out


# =========================================================
# MODEL
# =========================================================

class TinyLanguageModel:
    def __init__(self, vocab_size):
        self.embedding = Embedding(vocab_size, Config.embed_dim)
        self.attention = SelfAttention(Config.embed_dim)
        self.linear = Linear(Config.embed_dim, vocab_size)

    def forward(self, tokens):
        """
        Standard forward pass used by inference/tests.
        Returns logits only.
        """
        emb = self.embedding.forward(tokens)
        attn = self.attention.forward(emb)
        logits = self.linear.forward(attn)
        return logits

    def _forward_with_hidden(self, tokens):
        """
        Internal forward pass used during training.
        Returns logits + hidden states for gradients.
        """
        emb = self.embedding.forward(tokens)
        attn = self.attention.forward(emb)
        logits = self.linear.forward(attn)
        return logits, attn


# =========================================================
# LOSS
# =========================================================

def cross_entropy_loss(logits, targets):
    """
    Computes scalar cross-entropy loss.
    Public API used by tests.
    """
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
# TRAIN STEP
# =========================================================

def train_step(model, x, y, lr=1e-2):
    """
    Single SGD update on output projection layer.
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
# MAIN
# =========================================================

def main(verbose: bool = True):
    if verbose:
        print("\n=== CoreGPT Phase 5 ===\n")

    # 1️Load dataset
    text = load_dataset(Config.dataset_path)

    # 2️Build tokenizer
    tokenizer = CharTokenizer(text)
    # 3️Encode entire dataset
    data = tokenizer.encode(text)

    # 4️Train/val split
    train_data, val_data = train_val_split(data)

    # 5️ Sample batch from train set (single sample for training step)
    xb, yb = get_batch(train_data, Config.block_size, 1)

    # preview after batch exists
    

    model = TinyLanguageModel(tokenizer.vocab_size)
    logits = model.forward(xb[0])
    print(tokenizer.decode(xb[0][:50]))

    # 6️ Initialize model
    model = TinyLanguageModel(tokenizer.vocab_size)

    for step in range(5):
        xb, yb = get_batch(train_data, Config.block_size, 1)
        loss = train_step(model, xb[0], yb[0])
        print(f"[Step {step}] loss = {loss:.4f}")


if __name__ == "__main__":
    main()