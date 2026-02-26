# 基于源端易错词的大语言模型机器翻译优化方法研究

本仓库包含了论文《基于源端易错词的大语言模型机器翻译优化方法研究》的实验代码。本项目旨在通过挖掘源端易错词（Trigger Words）、分析其机理，并利用检索增强生成（RAG）和参数高效微调（LoRA）技术来优化大语言模型的机器翻译性能。

## 项目结构

项目代码重构为以下模块化结构：

- `src/`
  - `experiments/`: 核心实验代码，分为三个部分。
    - `mining_and_mechanism/`: **第一部分：易错词挖掘与机理探究**
    - `inference_rag/`: **第二部分：推理阶段优化 (RAG)**
    - `training_lora/`: **第三部分：训练阶段优化 (LoRA)**
  - `utils/`: 通用工具函数（数据读取、Prompt生成等）。
  - `legacy/`: 存档的旧版代码。
- `data/`: 数据存放目录（需根据实际路径配置）。
- `requirements.txt`: 项目依赖。

## 环境准备

1. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

2. **配置路径**

   代码中涉及大量绝对路径（如 `/public/home/...`），请务必在使用前根据您的本地环境修改各脚本中的 `CONFIG` 变量或路径常量。

## 实验列表与运行指南

### 第一部分：易错词挖掘与机理探究 (`src/experiments/mining_and_mechanism`)

本部分主要负责识别导致翻译错误的源端词汇，并分析其语言学特征及致错机理。

1.  **易错词挖掘 (Error-prone Word Mining)**
    *   **脚本**: `mine_error_prone_words.py`
    *   **描述**: 基于统计对齐信息，计算每个源端词的平均翻译质量与全局平均质量的差值，挖掘显著降低翻译质量的“易错词”。
    *   **运行**:
        ```bash
        python -m src.experiments.mining_and_mechanism.mine_error_prone_words
        ```

2.  **句法复杂度分析 (Syntactic Complexity Analysis)**
    *   **脚本**: `syntactic_complexity_experiment.py`
    *   **描述**: 分析易错词所在句子的依存句法深度、平均依存距离等指标，探究句法复杂度对翻译错误的影响。

3.  **错误定位与因果验证 (Causal Analysis)**
    *   **脚本**: `analyze_error_spans.py`
    *   **描述**: 利用 XCOMET 等工具检测译文中的错误片段，并验证这些错误片段是否与源端易错词存在对齐关系，从而确认因果性。

4.  **困惑度对比分析 (Perplexity Analysis)**
    *   **脚本**: `analyze_perplexity.py`
    *   **描述**: 计算模型对包含易错词句子的困惑度（PPL），并与普通句子对比，验证模型在处理这些词汇时的不确定性。

5.  **词性与错误类型关联分析**
    *   **脚本**: `word_error_association_analysis.py`
    *   **描述**: 统计不同词性（如动词、名词）的易错词主要导致何种类型的翻译错误（如漏译、误译）。

### 第二部分：推理阶段优化 (RAG) (`src/experiments/inference_rag`)

本部分通过检索增强生成（RAG）技术，在推理阶段引入高质量的范例来辅助模型翻译易错词。

1.  **基线推理**
    *   **脚本**: `baseline_inference.py`
    *   **描述**: 无检索增强的直接翻译推理，作为性能对比的基线。

2.  **基于语义的检索 (BGE Retrieval)**
    *   **脚本**: `bge_retrieval_inference.py`
    *   **描述**: 使用 BGE (BAAI General Embedding) 模型对源句进行向量化检索，召回语义相似的平行句对作为上下文范例。

3.  **检索与生成主流程**
    *   **脚本**: `inference_rag_main.py`
    *   **描述**: 集成多种检索策略（BM25, BGE）和上下文筛选机制（基于 COMET 质量筛选），执行最终的 RAG 翻译推理。

### 第三部分：训练阶段优化 (LoRA) (`src/experiments/training_lora`)

本部分通过构造针对性的训练数据，利用 LoRA 技术对大模型进行参数高效微调。

1.  **训练数据采样策略**
    *   **易错词加权采样 (Proposed)**: `sample_error_prone.py`
        *   优先采样包含易错词且参考译文质量较高的句对，通过加权提高其在训练数据中的占比。
    *   **高质量采样 (Baseline)**: `sample_high_quality.py`
        *   仅根据 COMET 分数采样高质量句对。
    *   **随机/长度匹配采样 (Baseline)**: `sample_random_length_matched.py`
        *   随机采样或控制长度分布的采样，用于消融实验。

2.  **LoRA 微调 (LoRA Fine-tuning)**
    *   **脚本**: `lora_finetune.py`
    *   **描述**: 使用 `peft` 库对 Llama-2/Llama-3 模型进行 LoRA 微调。支持多语言（Zh-En, Ar-En, En-De）配置。
    *   **运行示例**:
        ```bash
        # 针对中英任务，使用 Llama-3 模型进行微调
        python -m src.experiments.training_lora.lora_finetune --lang zhen --model_ver llama3 --gpu 0
        ```

3.  **基线评估**
    *   **脚本**: `baseline_evaluation.py`
    *   **描述**: 对微调前后的模型进行 BLEU 和 COMET 指标的自动化评估。

## 注意事项

*   **数据安全**: 请确保不上传包含敏感信息的私有数据文件。
*   **依赖版本**: 建议使用 Python 3.8+ 及 PyTorch 2.0+ 环境。

## 引用

如果您使用了本项目的代码或思路，请引用我们的论文：

> [论文引用格式待补充]
