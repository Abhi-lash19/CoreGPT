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
# MAIN
# =========================================================

def main(verbose: bool = True):
    if verbose:
        print("\n=== CoreGPT Phase 2 ===\n")

    # 1️Load dataset
    text = load_dataset(Config.dataset_path)

    # 2️Build tokenizer
    tokenizer = CharTokenizer(text)
    if verbose:
        print(f"[Tokenizer] vocab_size={tokenizer.vocab_size}")

    # 3️Encode entire dataset
    data = tokenizer.encode(text)

    # 4️Train/val split
    train_data, val_data = train_val_split(data)

    # 5️Sample batch from train set
    xb, yb = get_batch(
        train_data,
        Config.block_size,
        Config.batch_size
    )

    if verbose:
        print(f"[Batch] X={len(xb)}x{len(xb[0])} | Y={len(yb)}x{len(yb[0])}")

        # Preview one example
        print("\n[Preview]")
        print("Input text:")
        print(tokenizer.decode(xb[0][:80]))
        print("\nTarget text:")
        print(tokenizer.decode(yb[0][:80]))


if __name__ == "__main__":
    main()