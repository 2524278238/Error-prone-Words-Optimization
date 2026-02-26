# Part 2: Inference Optimization & RAG Experiments

This directory contains code for optimizing machine translation inference using Retrieval-Augmented Generation (RAG) and other strategies.

## 1. Main Inference Scripts
- **`inference_rag_main.py`** (formerly `inference_badword.py`):
  - The primary script for running inference with various RAG strategies.
  - Supports loading models (Llama-2, Llama-3, etc.) and using different prompt construction methods.
- **`baseline_inference.py`** (formerly `run_tgt.py`):
  - Baseline inference script for Zero-shot or Random Few-shot.
- **`trigger_word_inference.py`** (formerly `tgt_badword.py`):
  - Inference script specifically handling trigger words in prompts.

## 2. Retrieval Strategies (检索策略)
- **`bm25_retrieval.py`** / **`bm25_all.py`**:
  - Implements BM25 lexical retrieval for finding relevant context examples.
- **`bge_retrieval_inference.py`** (formerly `run_tgt_bge.py`):
  - Implements BGE (Semantic Embedding) based retrieval.
- **`bge_embedding.py`**:
  - Helper script for generating/loading BGE embeddings.
- **`hybrid_retrieval.py`** (formerly `comet+bge.py`):
  - Implements the **Hybrid Strategy** (Quality + Semantic) described in the paper.
  - Combines COMET quality scores and BGE semantic similarity.
- **`quality_based_selection.py`** (formerly `comet_select.py`):
  - Implements Quality-based (High COMET) example selection.

## 3. Index Generation & Scoring
- **`generate_index.py`** (formerly `generate_tgt_comet_index.py`):
  - Generates indices (e.g., COMET scores) for the training data to speed up retrieval.
- **`badword_comet_scoring.py`**:
  - Analyzes COMET scores for translations containing bad words.

## 4. Experiment Execution
- **`run_inference_experiments.sh`**:
  - Shell script to run batch inference experiments across different languages and models.
