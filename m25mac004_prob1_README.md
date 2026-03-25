# Problem 1: Corpus Creation, Word2Vec & Semantic Analysis

**File:** m25mac004_prob1.py

## Overview

This script implements a complete Word2Vec pipeline from scratch:

1. **Data Collection & Preprocessing** — Extract text from IIT Jodhpur PDF documents using PyPDFLoader, clean and tokenise using a custom tokeniser (no word_tokenize), save individual and merged corpora.
2. **Word2Vec Training** — CBOW and Skip-gram with Negative Sampling, implemented from scratch without any deep learning library. Includes OOV handling via Levenshtein edit distance.
3. **Hyperparameter Experiments** — Systematic variation of embedding dimension (50, 100, 200), context window (2, 5, 8), and negative samples (3, 5, 10).
4. **Semantic Analysis** — Top-5 nearest neighbours using cosine similarity, word analogy experiments (A : B :: C : ?).
5. **Visualisation** — PCA projections, t-SNE projections, and semantic category clustering for both CBOW and Skip-gram models.

## Dataset

Three PDF documents from IIT Jodhpur:

- doc1.pdf — Academic Rules & Regulations
- doc2.pdf — Course Catalog
- doc3.pdf — Curriculum / Syllabus

After preprocessing: **34,193 tokens**, **6,464 vocabulary**, **2,421 sentences**.

## How to Run Locally

### Step 1: Install Python

Make sure you have Python 3.8 or higher installed.

### Step 2: Install Dependencies

```bash
pip install langchain-community pypdf wordcloud matplotlib nltk scikit-learn adjustText pandas numpy
```

### Step 3: Prepare PDF Files

Create a folder called `data/` in the same directory as the script and place your three PDF files inside it:

```
project/
├── m25mac004_prob1.py
└── data/
    ├── doc1.pdf
    ├── doc2.pdf
    └── doc3.pdf
```

### Step 4: Modify the Script

Before running, make the following changes in m25mac004_prob1.py:

**1. Remove the Colab pip install line (near the top):**
```python
# DELETE or comment out this line:
!pip install langchain-community pypdf wordcloud matplotlib nltk scikit-learn adjustText -q
```

**2. Remove the Google Drive mount (near the top):**
```python
# DELETE or comment out these lines:
from google.colab import drive
drive.mount('/content/drive')
```

**3. Update PDF file paths** — replace the Google Drive paths with local paths:
```python
# CHANGE FROM:
pdf_docs_list = [
    "/content/drive/MyDrive/NLU assignment1/doc1.pdf",
    "/content/drive/MyDrive/NLU assignment1/doc2.pdf",
    "/content/drive/MyDrive/NLU assignment1/doc3.pdf",
]

# CHANGE TO:
pdf_docs_list = [
    "data/doc1.pdf",
    "data/doc2.pdf",
    "data/doc3.pdf",
]
```

**4. Update corpus save directory** — replace the Drive path with a local path:
```python
# CHANGE FROM:
CORPUS_DIR = '/content/drive/MyDrive/NLU assignment1/'

# CHANGE TO:
CORPUS_DIR = 'output/'
```

Make sure the `output/` directory exists:
```bash
mkdir -p output
```

### Step 5: Run the Script

```bash
python m25mac004_prob1.py
```

The script will run all tasks sequentially: PDF extraction, preprocessing, hyperparameter experiments, final model training, semantic analysis, and visualisation. Total runtime is approximately 10–15 minutes on a modern CPU.

## Script Structure

| Section | Description |
|---------|-------------|
| Task 1: Data Collection & Preprocessing | Load PDFs via PyPDFLoader, clean text with regex, custom tokenise, save corpora, compute statistics, generate word cloud |
| Task 2: Model Training | Word2Vec class implementation (CBOW + Skip-gram + negative sampling), hyperparameter experiments varying dim/window/neg, train final models |
| Task 3: Semantic Analysis | Top-5 nearest neighbours for research/student/phd/exam, word analogy experiments |
| Task 4: Visualisation | PCA plots, t-SNE plots, semantic category clustering (People/Academics/Degrees/Metrics) |

## Output Files

| File | Description |
|------|-------------|
| output/corpus_1.txt, corpus_2.txt, corpus_3.txt | Individual cleaned corpora |
| output/final_corpus.txt | Merged corpus |
| word2vec_cbow_final.npz | Saved CBOW model weights |
| word2vec_sg_final.npz | Saved Skip-gram model weights |
| pca_cbow.png, pca_skip-gram.png | PCA projection plots |
| tsne_cbow.png, tsne_skip-gram.png | t-SNE projection plots |
| cat_tsne_cbow.png, cat_pca_cbow.png, etc. | Category clustering plots |

## Key Results

- **Vocabulary:** 1,964 words (min_count=3)
- **CBOW training time:** ~20s | **Skip-gram training time:** ~48s
- **Best analogy result:** UG : BTech :: PG : ? → Skip-gram correctly predicts **mtech**
- **Skip-gram** produces more semantically meaningful embeddings than CBOW on this small corpus
