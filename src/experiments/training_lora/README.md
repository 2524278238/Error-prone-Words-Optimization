# Part 3: Training Optimization & LoRA Experiments

This directory contains code for constructing targeted training data and fine-tuning Large Language Models (LLMs) using LoRA (Low-Rank Adaptation).

## 1. Data Construction Strategies (训练数据构建策略)

The core contribution is the **Error-prone Word Weighted Sampling** strategy. We also implement several baselines for comparison.

- **`sample_error_prone.py`** (formerly `sample_tgt_train_data.py`):
  - **Proposed Method**: Implements the Error-prone Word Weighted Sampling strategy.
  - Prioritizes sentences containing identified trigger words that have high translation quality.
- **`sample_high_quality.py`** (formerly `sample_train_data_cometlen.py`):
  - **Baseline 1**: High COMET Sampling.
  - Selects sentences with high COMET scores, balancing for sentence length.
- **`sample_random_length_matched.py`** (formerly `sample_train_data_len.py`):
  - **Baseline 2 / Ablation**: Random Sampling / Length-matched Sampling.
  - Used for constructing "Pseudo Error-prone Word" datasets (Ablation Study) or simple random baselines.
- **`run_data_sampling.sh`**:
  - Shell script to run the various data sampling strategies.

## 2. LoRA Fine-tuning (LoRA微调)

- **`lora_finetune.py`** (formerly `train_tgt.py`):
  - The main training script using `peft` (LoRA) and `transformers`.
  - Supports multi-language (Zh-En, Ar-En, En-De) and multi-model (Llama-2, Llama-3) configurations.
  - Key parameters: `r=16`, `lora_alpha=16`, `target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
- **`run_lora_training.sh`**:
  - Shell script to launch LoRA fine-tuning jobs.

## 3. Evaluation & Baselines

- **`baseline_evaluation.py`**:
  - Script for evaluating baseline model performance (Zero-shot) before fine-tuning.
- **`inference_test.py`**:
  - Simple script for testing model inference or data processing.
