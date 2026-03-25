# Guide to Running m25mac004_prob2.py

Hello again! If you want to check out m25mac004_prob2.py, this guide will walk you through exactly what it is and how to get it going.

This script is all about teaching deep learning models how to generate names character by character. It feeds on a dataset of 1000 Indian names, learns their structural patterns (like common syllables, prefixes, and suffixes), and tries to hallucinate entirely new, unique names. 

To compare different approaches, it implements three different Neural Network architectures from the ground up:
1. A Vanilla Recurrent Neural Network (RNN)
2. A Bidirectional Long Short-Term Memory Network (BLSTM)
3. An RNN enhanced with a Bahdanau-style Attention Mechanism

---

## What Does This Script Actually Do?

1. Data Prep and Vocab Building: It reads the TrainingNames.txt file, converts everything to lowercase, and assigns a unique index number to every character (including special padding, start, and end tokens).
2. Model Training: It builds and trains the three architectures side by side using PyTorch. The models learn to predict the very next character based on all the characters they've seen so far.
3. Neural Name Generation: After training, it samples characters progressively (using some neat tricks like top-k sampling, temperature scaling, and repetition penalties to avoid gibberish) to create 300 novel names for each model.
4. Qualitative and Quantitative Evaluation: The script mathematically judges the models by calculating the "Novelty Rate" (how many generated names were NOT in the training data) and "Diversity" (how many unique names it generated). It also flags failure modes like names being too short or having impossible consonant clusters.
5. Character Embeddings visualization: Finally, it extracts the learned mathematical representations of all the alphabet characters and shrinks them down into 2D scatter plots using PCA and t-SNE algorithms. This visually separates vowels from consonants!

---

## How to Run It on Your Machine

### Step 1: Install the Requirements
This script relies heavily on PyTorch and Scikit-Learn. Open your terminal and run this to install the dependencies:

```bash
pip install torch numpy matplotlib scikit-learn
```

### Step 2: Fix the File Paths! (Very Important)
Just like the first script, this one was originally written for Google Colab with Google Drive mounted. 

Before you run it on your own computer, open m25mac004_prob2.py in your code editor and change the file path for the dataset:
- Look for the load_names() function call. Update the path from:
  /content/drive/MyDrive/NLU assignment1/TrainingNames.txt
  To wherever TrainingNames.txt is saved on your local machine.

### Step 3: Run It!
With the packages installed and paths verified, simply fire up the script from your terminal:

```bash
python m25mac004_prob2.py
```
Note: If your machine has a compatible GPU (CUDA), PyTorch will automatically use it for much faster training! Otherwise, it will fall back to the CPU setup.

### Step 4: Check Your Outputs
Because this script runs iteratively, you can watch it train in the console step-by-step. 

Once it's finished, it will output:
- Live plots to your screen containing the Training Loss curves.
- A table in the terminal comparing Novelty Rates and Diversity Metrics.
- Some sample generated names and an automated failure analysis summary printed in the terminal.
- Cool 2D dot-plots (PCA and t-SNE) showing how the models learned to separate vowels from consonants.

Let the models train and see what kind of weird and wonderful names they can come up with!
