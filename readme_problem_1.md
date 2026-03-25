# Guide to Running m25mac004_prob1.py

Hey there! If you're looking to run m25mac004_prob1.py, you're in the right place. This script is essentially a full Natural Language Understanding (NLU) pipeline built from the ground up. 

It takes a bunch of raw PDF documents, cleans up the text, and trains its own Word2Vec models (both CBOW and Skip-gram) completely from scratch using just NumPy. After training, it runs some cool semantic tests (like finding similar words and solving word analogies) and plots everything out visually.

Here's exactly what the script is doing behind the scenes and how you can run it on your own machine.

---

## What Does This Script Actually Do?

1. Reads and Cleans PDFs: It grabs text from a few specific PDF files, strips out all the junk (like URLs, page numbers, and weird characters), and converts everything to lowercase.
2. Trains Word2Vec from Scratch: Instead of using a pre-built library, this script trains Word2Vec models purely using math and NumPy arrays. It builds a vocabulary and learns the "meanings" of words based on their context.
   - Bonus: It handles Out-Of-Vocabulary (OOV) words. If you ask it for a word it hasn't seen, it's smart enough to find the closest matching word it does know!
3. Semantic Analysis: It tests the models by asking questions like: "What are the 5 words most similar to 'research'?" or "If A is to B, then C is to what?"
4. Draws Pretty Graphs: The word embeddings are high-dimensional, so the script uses PCA and t-SNE algorithms to smash them down into 2D maps and saves them as PNG images so you can actually see the relationships between academic categories (like "People", "Degrees", and "Metrics").

---

## How to Run It on Your Machine

### Step 1: Install the Requirements
You're going to need a few standard Python libraries. Open up your terminal and run this command to install everything you need:

```bash
pip install nltk numpy pandas matplotlib wordcloud langchain-community pypdf scikit-learn adjustText
```

### Step 2: Fix the File Paths! (Very Important)
This script was originally written to run in Google Colab with Google Drive mounted. Because of this, the file paths in the code are hardcoded to look for things inside /content/drive/.

Before you run the script locally, you need to open m25mac004_prob1.py in your code editor and change the paths so it knows where to find your PDFs and where to save the outputs.

Look for these specific lines in the code:
- pdf_docs_list: Update these three paths to point to where your doc1.pdf, doc2.pdf, and doc3.pdf are located on your PC.
- CORPUS_DIR: Update this to point to a local folder where you want the script to save the generated text corpora.

### Step 3: Run It!
Once your packages are installed and the file paths are pointing to the right places, just run the script from your terminal:

```bash
python m25mac004_prob1.py
```

### Step 4: Check Your Outputs
Sit back and wait for it to finish grinding through the math! When it's done, check your working directory and the CORPUS_DIR you set. You'll find:
- Cleaned Text Files: corpus_1.txt and a big final_corpus.txt.
- Model Files: word2vec_cbow_final.npz and word2vec_sg_final.npz (these are the brains of the models you just trained!).
- Cool Visuals: Several PNG files containing word clouds, PCA scatter plots, and colorful t-SNE clusters showing how the words map out conceptually.

Have fun experimenting with it, and let the model do its magic!
