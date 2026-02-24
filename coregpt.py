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
# TRAIN / VALIDATION SPLIT
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
        start_idx = random.randint(0, len(data) - block_size - 1)

        x = data[start_idx : start_idx + block_size]
        y = data[start_idx + 1 : start_idx + block_size + 1]

        x_batch.append(x)
        y_batch.append(y)

    return x_batch, y_batch


# =========================================================
# PHASE 3 — MATRIX UTILITIES
# =========================================================

def random_matrix(rows: int, cols: int):
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
    def __init__(self, embed_dim: int):
        self.query = Linear(embed_dim, embed_dim)
        self.key = Linear(embed_dim, embed_dim)
        self.value = Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        Q = self.query.forward(x)
        K = self.key.forward(x)
        V = self.value.forward(x)

        KT = transpose(K)
        scores = matmul(Q, KT)

        for i in range(len(scores)):
            for j in range(len(scores[i])):
                scores[i][j] /= self.scale

        scores = apply_causal_mask(scores)
        weights = [softmax(row) for row in scores]
        out = matmul(weights, V)

        return out


# =========================================================
# PHASE 3 — MODEL LAYERS
# =========================================================

class Embedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        self.weight = random_matrix(vocab_size, embed_dim)

    def forward(self, tokens: List[int]):
        return [self.weight[t] for t in tokens]


class Linear:
    def __init__(self, in_dim: int, out_dim: int):
        self.weight = random_matrix(in_dim, out_dim)
        self.bias = [0.0] * out_dim

    def forward(self, x):
        out = matmul(x, self.weight)
        for row in out:
            for i in range(len(row)):
                row[i] += self.bias[i]
        return out


# =========================================================
# PHASE 4 — MODEL
# =========================================================

class TinyLanguageModel:
    def __init__(self, vocab_size: int):
        self.embedding = Embedding(vocab_size, Config.embed_dim)
        self.attention = SelfAttention(Config.embed_dim)
        self.linear = Linear(Config.embed_dim, vocab_size)

    def forward(self, tokens: List[int]):
        x = self.embedding.forward(tokens)
        x = self.attention.forward(x)
        logits = self.linear.forward(x)
        return logits


# =========================================================
# MAIN
# =========================================================

def main(verbose: bool = True):
    if verbose:
        print("\n=== CoreGPT Phase 4 ===\n")

    # 1️Load dataset
    text = load_dataset(Config.dataset_path)

    # 2️Build tokenizer
    tokenizer = CharTokenizer(text)
    # 3️Encode entire dataset
    data = tokenizer.encode(text)

    # 4️Train/val split
    train_data, val_data = train_val_split(data)

    # 5️Sample batch from train set
    xb, yb = get_batch(train_data, Config.block_size, Config.batch_size)

    # preview after batch exists
    

    model = TinyLanguageModel(tokenizer.vocab_size)
    logits = model.forward(xb[0])
    print(tokenizer.decode(xb[0][:50]))

    if verbose:
        print(f"[Forward] logits shape = {len(logits)} x {len(logits[0])}")


if __name__ == "__main__":
    main()