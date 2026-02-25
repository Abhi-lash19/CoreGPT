"""
CoreGPT Configuration

This file centralizes all hyperparameters and settings
so the model can be easily tuned.
"""

class Config:
    # ===== Model =====
    block_size = 32        # context length
    embed_dim = 64
    hidden_dim = 128

    # ===== Training =====
    learning_rate = 1e-3
    epochs = 3
    batch_size = 16

    # ===== Generation =====
    temperature = 0.8

    # ===== Paths =====
    dataset_path = "data/dataset.txt"
    checkpoint_path = "checkpoints/model.json"

    # ===== Transformer Architecture =====
    num_layers = 2
    num_heads = 2
    ffn_multiplier = 4