# Semantic Frame Induction

A research-grade pipeline for unsupervised and weakly-supervised semantic frame induction using the FrameNet dataset. This project aims to discover event schemas (predicates and their semantic roles) automatically from raw text using dependency parsing, contextual embeddings (MiniLM, BERT), and graph-based clustering techniques.

## Features

- **Interactive AI Studio**: A stunning, ultra-premium FastAPI web dashboard featuring a dark-theme, responsive split-screen glassmorphism UI, animated mesh gradient backgrounds, and real-time visualization widgets.
- **Targeted Semantic Induction**: Uses precise `Predicate Windowing` over `Sentence Transformers` (MiniLM) to drastically increase vector alignment accuracy, clustering over 5,000 cached datasets in milliseconds.
- **Data Extraction**: Automated extraction and preprocessing of sentences and semantic roles from NLTK's FrameNet corpus.
- **Linguistic Analysis**: Case-insensitive, spaCy-driven predicate-argument graph construction based on syntactic dependency relations.
- **Contextual Representations**: Support for TF-IDF, spaCy vectors, BERT, and Sentence Transformers (`all-MiniLM-L6-v2`).
- **Clustering & Evaluation**: Graph-based community detection (Louvain) and traditional clustering with automated scoring metrics (ARI, NMI, Purity).

## Installation

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### 1. Interactive AI Studio UI (Web Dashboard)
To run the web interface and test semantic frame inductions in real-time, start the FastAPI server:

```bash
python -m uvicorn api:app
```
*Note: The startup sequence pre-loads AI embeddings into memory and may take about 30 seconds. Once started, open your browser to **http://localhost:8000/**.*

### 2. Batch Experimentation Pipeline
To run the main headless evaluation pipeline in the terminal:

**Run on a development subset** (top 100 verbs, max 2000 sentences):
```bash
python main.py
```

**Run on the entire FrameNet dataset** (warning: computationally expensive):
```bash
python main.py --full
```

## Structure

- `src/load_framenet.py`: NLTK FrameNet extraction.
- `src/feature_extraction.py`: Dependency parsing and predicate-argument graph creation.
- `src/embedding_models.py`: TF-IDF, Word2Vec (SpaCy), and Transformer embeddings.
- `src/clustering_methods.py`: Traditional and graph-based clustering implementations.
- `src/evaluation_metrics.py`: Metrics for unsupervised cluster evaluation.
- `src/visualization.py`: PCA, t-SNE, and heatmap generation scripts.
- `main.py`: Experiment runner tying the pipeline together.

## Results
The pipeline outputs `comparison_results.csv` logging the Runtime, ARI, NMI, Purity, and V-measure across all representations and clustering algorithms. Visualizations such as t-SNE plots of the induced frame clusters are saved in the `results/` directory.
