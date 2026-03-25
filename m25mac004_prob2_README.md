# Problem 2: Character-Level Name Generation using RNN Variants
  
**File:** m25mac004_prob2.py

## Overview

This script implements character-level Indian name generation using three recurrent neural network architectures, all built in PyTorch:

1. **Vanilla RNN** — 2-layer Elman RNN (62,488 parameters)
2. **Bidirectional LSTM (BLSTM)** — 2-layer BiLSTM with concatenated forward/backward states (601,624 parameters)
3. **RNN with Attention** — RNN encoder-decoder with Bahdanau-style causal attention (169,624 parameters)

The script trains all three models on 1000 Indian names, generates 300 names from each, evaluates novelty and diversity, analyses failure modes, and visualises learned character embeddings using PCA and t-SNE.

## Dataset

- **File:** TrainingNames.txt — 1000 synthetic Indian names
- **Alphabet:** 21 unique characters + 3 special tokens (PAD, SOS, EOS) = 24 total
- **Name lengths:** min 4, max 10, mean 6.3

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- NumPy
- Matplotlib
- scikit-learn

## How to Run Locally

### Step 1: Install Dependencies

```bash
pip install torch numpy matplotlib scikit-learn
```

If you have an NVIDIA GPU and want faster training:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Prepare the Dataset

Place `TrainingNames.txt` in the same directory as the script:

```
project/
├── m25mac004_prob2.py
└── TrainingNames.txt
```

### Step 3: Modify the Script

Before running, make the following changes in m25mac004_prob2.py:

**1. Remove the Google Drive mount (near the top):**
```python
# DELETE or comment out these lines:
from google.colab import drive
drive.mount('/content/drive')
```

**2. Update the dataset path** — replace the Google Drive path with a local path:
```python
# CHANGE FROM:
names = load_names('/content/drive/MyDrive/NLU assignment1/TrainingNames.txt')

# CHANGE TO:
names = load_names('TrainingNames.txt')
```

### Step 4: Run the Script

```bash
python m25mac004_prob2.py
```

The script runs all four tasks sequentially:
- Task 1: Build models and train for 100 epochs each (~5 min on CPU)
- Task 2: Generate 300 names per model and compute metrics
- Task 3: Show sample names and failure analysis
- Task 4: Plot PCA and t-SNE character embedding visualisations

Total runtime is approximately **5–6 minutes on CPU**, faster with GPU.

## Script Structure

| Section | Description |
|---------|-------------|
| Task 1: Model Implementation | Dataset loading, vocabulary creation, model definitions (Vanilla RNN, BLSTM, AttentionRNN), training loop with cross-entropy loss, Adam optimiser, teacher forcing, gradient clipping |
| Task 2: Quantitative Evaluation | Name generation with temperature scaling, top-k sampling, repetition penalty; novelty and diversity metrics |
| Task 3: Qualitative Analysis | 15 sample names per model, automated failure detection (repeated chars, short names, unnatural consonant clusters) |
| Task 4: Visualisation | PCA and t-SNE projections of learned character embeddings, vowel vs consonant colouring |

## Hyperparameters

All three models share the same hyperparameters:

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 64 |
| Hidden size | 128 |
| Number of layers | 2 |
| Dropout | 0.2 |
| Learning rate | 0.003 |
| Epochs | 100 |
| Teacher forcing ratio | 0.5 |
| Batch size | 64 |
| Gradient clipping | 5 |

## Key Results

| Model | Novelty | Diversity | Parameters | Training Time |
|-------|---------|-----------|------------|---------------|
| Vanilla RNN | 21.3% | 73.3% | 62,488 | ~30s |
| BLSTM | 68.0% | 38.3% | 601,624 | ~152s |
| Attention RNN | 21.0% | 47.0% | 169,624 | ~104s |

- **Vanilla RNN** produces the most diverse names but mostly copies from training data
- **BLSTM** achieves highest novelty but low diversity with repetitive, unusual names
- **Attention RNN** generates the most realistic-sounding names (Karesh, Dhruyash, Shivraj, Navdev)

## Output

The script prints all results to the console and displays matplotlib plots for:
- Training loss curves (all 3 models)
- PCA projection of character embeddings (3 subplots)
- t-SNE projection of character embeddings (3 subplots)
