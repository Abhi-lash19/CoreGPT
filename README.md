# CoreGPT 

*A tiny Transformer language model built from scratch in pure Python*

---

## Overview

**CoreGPT** is an educational project where I implemented a miniature GPT-style language model **from first principles**, without using any machine learning libraries.

The goal of this project was not performance, but **deep understanding** — to explore how modern transformer-based LLMs actually work under the hood.

This repository demonstrates the full lifecycle of a language model:

* Tokenization
* Embeddings
* Multi-head self-attention
* Transformer blocks
* Training loop
* Text generation

All implemented in a single, readable Python codebase.

---

##  What I Learned

Building CoreGPT helped me understand the core ideas behind modern LLMs:

### 🔹 How text becomes numbers

Characters are converted into tokens and mapped to vectors using embeddings.

### 🔹 How context is understood

Self-attention allows each token to look at previous tokens and decide what matters.

### 🔹 How deeper reasoning emerges

Stacking transformer layers enables hierarchical pattern learning.

### 🔹 How models learn

Cross-entropy loss and gradient updates improve predictions over time.

### 🔹 How text is generated

The model predicts the next token step-by-step using probability sampling.

This project gave me intuition into **why transformers work**, not just how to use them.

---

## Architecture

CoreGPT implements a simplified transformer pipeline:

```
Text → Tokenizer → Embeddings + Positional Encoding
      → Transformer Blocks (Attention + FFN + Residuals)
      → Linear Projection → Next Token Probabilities
```

### Components included

- Character tokenizer with UNK support
- Token embeddings
- Positional embeddings
- Multi-head causal self-attention
- Layer normalization
- Feed-forward network
- Residual connections
- Multiple transformer layers
- Training loop with validation loss
- Autoregressive text generation

---

## Example Output

After training on a small dataset, CoreGPT generates text with learned structure:

```
CoreGPT explores how language models learn patterns in text,
gradually improving predictions through attention mechanisms...
```

(The model is intentionally small, so output is noisy but structured.)

---

## How to Run Locally

### Clone the repository

```bash
git clone https://github.com/Abhi-lash19/coregpt.git
cd coregpt
```

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

### Install dependencies

CoreGPT uses only standard Python libraries, so no heavy installs are required.

(Optional)

```bash
pip install -r requirements.txt
```

### Run training

```bash
python coregpt.py
```

You’ll see:

- Training loss for each epoch  
- Validation loss for monitoring generalization  
- A sample of generated text after training completes 

---

## Project Scope

This is an **educational implementation**, not a production model.

It intentionally avoids:

* GPU acceleration
* Large datasets
* Advanced optimizers
* Distributed training

The focus is clarity over scale.

---

## How This Differs from Real LLMs

Large language models such as GPT-4 or Claude are built with:

* Billions of parameters
* Extremely large training datasets
* Highly optimized training systems
* Additional alignment and safety components

CoreGPT is much smaller and simpler. Its goal is to demonstrate the main ideas behind transformer language models in a clear and easy-to-understand way.

---

## Why I Built This

I wanted to move beyond using AI tools and truly understand:

How transformers process language
How attention enables context awareness
How training shapes predictions

Building CoreGPT from scratch gave me a strong mental model of how modern AI systems work internally.

---

## How CoreGPT Works (Deep Dive)

This section walks through the full lifecycle of how CoreGPT processes text from raw characters to generated output.

---

### Tokenization — Turning Text into Numbers

The model cannot understand text directly, so the first step is converting characters into numeric tokens.

**Process**

1. Scan the dataset and collect all unique characters
2. Assign each character an integer ID
3. Replace each character in the text with its ID

**Example**

```text
Input text:  "Core"
Tokens:      [12, 7, 19, 4]
```

This numeric representation allows the model to perform mathematical operations.

---

### Embeddings — Turning Tokens into Vectors

Each token is mapped to a vector of floating-point numbers using an embedding matrix.

Think of this as giving each character a “meaningful coordinate” in a high-dimensional space.

```text
Token 12 → [0.14, -0.02, 0.31, ...]
```

These vectors allow the model to learn relationships between tokens.

---

### Positional Encoding — Adding Order Information

Transformers process tokens in parallel, so they need a way to know **where each token appears**.

CoreGPT adds a positional vector to each token embedding:

```text
Final vector = token_embedding + position_embedding
```

This allows the model to distinguish:

* “cat bites dog”
* “dog bites cat”

---

### Multi-Head Self-Attention — Understanding Context

This is the core of the transformer.

For each token, the model computes three vectors:

* Query → what this token is looking for
* Key → what this token offers
* Value → information this token carries

The model compares queries with keys to compute attention scores.

#### Attention formula

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

This lets each token “look back” at previous tokens and decide what is important.

#### Multi-head attention

Instead of one attention calculation, the model runs multiple heads in parallel.

Each head learns different patterns, such as:

* grammar
* punctuation
* word boundaries

The outputs are then combined.

---

### Feed-Forward Network — Processing Information

After attention, each token vector passes through a small neural network:

```text
Linear → ReLU → Linear
```

This step allows the model to transform and refine the information learned from attention.

---

### Residual Connections + Layer Normalization

Residual connections help preserve information:

```text
output = input + transformed_input
```

Layer normalization stabilizes training by keeping values in a consistent range.

These two techniques are critical for deep transformer models.

---

### Stacking Transformer Blocks — Building Reasoning Depth

CoreGPT stacks multiple transformer blocks.

Each layer:

* reinterprets context
* refines understanding
* builds more abstract representations

This hierarchical processing is what enables modern LLMs to perform reasoning.

---

### Output Projection — Predicting the Next Token

After the final transformer layer, the model projects each token vector into a probability distribution over the vocabulary.

```text
logits → softmax → probabilities
```

The model then predicts the most likely next token.

---

### Training — Learning from Mistakes

During training:

1. The model predicts the next token
2. Cross-entropy loss measures the error
3. We update weights to reduce this error

Over many iterations, predictions improve.

---

### Text Generation — Autoregressive Loop

To generate text:

1. Start with a prompt
2. Predict the next token
3. Append it to the input
4. Repeat

```text
"CoreGPT" → "CoreGPT is" → "CoreGPT is a" → ...
```

Temperature sampling controls randomness:

* low temperature → safer text
* high temperature → more creative text

---

## Big Picture

CoreGPT demonstrates the fundamental pipeline used by modern language models:

```text
Text → Tokens → Vectors → Attention → Reasoning Layers → Prediction → Generated Text
```

Even though the model is small, the same principles scale up to large-scale LLMs used in real-world AI systems.

---

https://deepwiki.com/Abhi-lash19/CoreGPT

---
