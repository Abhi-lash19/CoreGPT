"""
CoreGPT

Phase 1:
- Dataset loading
- Basic project entrypoint

This script will evolve into the full model.
"""

import os
from config import Config


# =========================
# DATASET LOADER
# =========================

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

    print(f"✅ Dataset loaded | Characters: {len(text)}")
    return text


# =========================
# MAIN
# =========================

def main():
    print("🧠 CoreGPT Phase 1 Starting...\n")

    text = load_dataset(Config.dataset_path)

    print("Preview:")
    print("-" * 40)
    print(text[:500])
    print("-" * 40)


if __name__ == "__main__":
    main()