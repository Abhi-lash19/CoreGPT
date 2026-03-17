"""
CoreGPT Configuration

This file centralizes all hyperparameters and settings
so the model can be easily tuned.
"""

class Config:
    # ===== Model =====
    block_size = 32        # gives better context
    embed_dim = 48
    hidden_dim = 96

    # ===== Training =====
    learning_rate = 1e-3
    epochs = 5             # train longer
    batch_size = 1

    # ===== Generation =====
    temperature = 0.5      # slightly higher for diversity

    # ===== Paths =====
    dataset_path = "data/dataset.txt"
    checkpoint_path = "checkpoints/model.json"

    # ===== Transformer Architecture =====
    num_layers = 2
    num_heads = 2          # VERY IMPORTANT
    ffn_multiplier = 2

    # ===== Regularization =====
    dropout = 0.05