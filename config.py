"""
CoreGPT Configuration

This file centralizes all hyperparameters and settings
so the model can be easily tuned.
"""

class Config:
    # ===== Model =====
    block_size = 32        # ↑ more context/longer memory, ↓ faster but worse understanding (increase if text needs longer dependencies)
    embed_dim = 64         # ↑ model capacity & quality, ↓ faster but simpler patterns (increase gradually: 32 → 48 → 64)
    hidden_dim = 128        # ↑ FFN power, ↓ lighter compute (usually 2x–4x embed_dim, scale with embed_dim)

    # ===== Training =====
    learning_rate = 5e-4   # ↑ faster learning but unstable, ↓ slower but stable (reduce if loss fluctuates)
    epochs = 10             # ↑ better learning, ↓ faster runs (increase until loss stops improving)
    batch_size = 1         # ↑ smoother gradients (if >1), ↓ more noisy but fine for pure Python (keep 1 for speed)

    # ===== Generation =====
    temperature = 0.5      # ↑ more random/creative, ↓ more deterministic/repetitive (0.4–0.7 is sweet spot)

    # ===== Paths =====
    dataset_path = "data/dataset.txt"         # path to training text file
    checkpoint_path = "checkpoints/model.json"  # where model weights are saved/loaded

    # ===== Transformer Architecture =====
    num_layers = 3         # ↑ deeper understanding, ↓ faster but shallow (increase slowly: 1 → 2 → 3)
    num_heads = 2          # ↑ better attention diversity, ↓ simpler attention (must divide embed_dim)
    ffn_multiplier = 3     # ↑ FFN capacity, ↓ faster compute (common: 2–4, increase with bigger models)

    # ===== Regularization =====
    dropout = 0.05         # ↑ prevents overfitting but slows learning, ↓ faster learning but risk overfit (0.05–0.1 ideal for small models)