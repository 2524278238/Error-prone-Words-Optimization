# Part 1: Error-prone Word Mining and Mechanism Exploration Experiments

This directory contains code for identifying source-side error-prone words (trigger words) and analyzing their underlying mechanisms (syntax, semantics, error causality).

## 1. Error-prone Word Identification (易错词挖掘)
- **`mine_error_prone_words.py`** (formerly `src_badword.py`): 
  - Implements the mining algorithm: `score(w) = (μ_global - μ_w) * count_w`.
  - Filters words based on frequency and quality difference thresholds.
- **`preprocess_triggers.py`**: 
  - Preprocesses and filters identified trigger words for downstream tasks.

## 2. Causal Analysis & Error Localization (机理探究：因果性与错误定位)
- **`analyze_error_spans.py`** (formerly `find_badst.py`): 
  - Uses XCOMET to detect error spans in translations.
  - Verifies if trigger words fall within error spans (calculates error/omission rates).
- **`alignment_based_analysis.py`**: 
  - Uses word alignment (e.g., awesome-align) to check if trigger words are correctly aligned to target words.
- **`word_alignment_finder.py`**: 
  - Helper script for finding word alignments.
- **`word_error_association_analysis.py`**: 
  - Analyzes the correlation (LogOR) between specific POS tags and error types (Mistranslation, Omission, Untranslated).

## 3. Syntactic Complexity Analysis (句法复杂度分析)
- **`syntactic_complexity_experiment.py`**: 
  - Compares syntactic metrics (Dependency Depth, Sentence Length, Arc Span, Clause Ratio) between sentences containing trigger words and random sentences.

## 4. Semantic Ambiguity Analysis (语义歧义度分析)
- **`semantic_ambiguity_experiment.py`**: 
  - Compares semantic ambiguity (number of senses using WordNet/OpenHowNet) between trigger words and non-trigger words.

## 5. Part-of-Speech (POS) Analysis (词性分布分析)
- **`pos_tag_glm45.py`**: 
  - Uses GLM-4.5 to tag the Part-of-Speech for trigger words.
- **`refine_pos_tags_glm45.py`**: 
  - Refines and standardizes POS tags.
- **`plot_pos_distribution.py`**: 
  - Generates visualizations for POS distribution of trigger words.

## 6. MQM Error Analysis (多维质量度量分析)
- **`mqm_error_classifier.py`**: 
  - Classifies translation errors into MQM categories (Accuracy, Fluency, etc.) using LLMs.
- **`backfill_mqm_failures.py`**: 
  - Retries failed classification requests.
- **`generate_word_profile.py`**: 
  - Aggregates all analysis results (POS, Error Rates, MQM types) into a comprehensive profile for each trigger word.
