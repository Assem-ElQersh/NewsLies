<div align="center">

# NewsLies

**Arabic Fake News Detection & Source Leakage Analysis**

![Status](https://img.shields.io/badge/STATUS-COMPLETED-green?style=for-the-badge&logo=github)  

</div>

A deep learning project that classifies Arabic news articles as **credible**, **not credible**, or **undecided**, trained on the Arabic Fake News Dataset (AFND). 

This project explores a critical flaw in weakly-supervised news datasets: **Source Leakage**. The project proves experimentally that models achieve high accuracy on standard random splits not by learning to detect "fake news" (Factual Verification), but by identifying the publisher's writing style (Source-Induced Credibility).

## Task Definition

To properly evaluate fake news detection, we must distinguish between two tasks:
- **Task A: Source-Induced Credibility (The Shortcut)**: The model predicts the credibility label by recognizing *who* wrote the article (e.g., using specific vocabulary, dialect, or boilerplate text like "Reuters" or "Ennahar Online"). Since the dataset's ground truth labels were assigned at the *publisher* level by Misbar, this task reduces to source-attribution rather than fact-checking.
- **Task B: Factual Verification (The True Goal)**: The model predicts credibility by evaluating the actual claims in the text, independent of the publisher's specific style or identity. This requires generalization to unseen publishers (evaluated via a Source-Disjoint Split).

## The Epistemological Ledger

All empirical findings and deductions from the experimental pipeline are logged below:

| The Empirical Finding / Metric | Exact Script/Notebook Name | Analytical Deduction (What this rules out/forces next) |
| :--- | :--- | :--- |
| 100.0% Missing Publication Dates | `notebooks/A_data_audit.py` | Eliminates the possibility of temporal (rolling) evaluation. Models cannot use time to predict credibility in AFND. |
| Model can predict the publisher identity with >98% accuracy based solely on article text. | `notebooks/A_data_audit.py` (Text-to-Source Leakage) | Forces the creation of a source-disjoint split. The text is highly contaminated with publisher-specific stylistic markers. |
| AraBERT Random Split Macro-F1: 0.878 | `notebooks/B2_development_transformers.py` | Baseline transformer performance when source leakage is fully available. Shows the upper bound of Task A (Source-Induced Credibility). |
| AraBERT Source-Disjoint Split Macro-F1: 0.370 | `notebooks/C_hard_evaluation.py` | Proves that the model fails to generalize to unseen publishers (Task B). The original high accuracy was largely an artifact of source memorization. |
| 47.6% of errors on unseen sources are caused by "Government & Official Statements" misclassification. | `notebooks/D_taxonomy_generator.py` | Forces conclusion that the model learned superficial topics (e.g., assuming government statements are always credible) rather than factual veracity. |

## Model Comparison

| Model (Architecture) | Random Split (Task A) Macro-F1 | Disjoint Split (Task B) Macro-F1 | Latency (Local) | Hardware Notes |
| :--- | :--- | :--- | :--- | :--- |
| E0: Majority Class | ~0.33 | ~0.33 | < 1ms | Baseline |
| E1: TF-IDF + Linear SVM | 0.72 | (not evaluated) | ~10ms | VRAM independent |
| E2: LSTM (2x64) | 0.73 | (not evaluated) | ~20ms | Legacy System |
| E3: BiGRU + Attention | 0.75 | (not evaluated) | ~25ms | - |
| E4: AraBERTv0.2-base | **0.878** | **0.370** | ~150ms | Requires AMP (RTX 3050 4GB) |
| E5: CAMeLBERT-MSA | 0.876 | (not evaluated) | ~150ms | Requires AMP (RTX 3050 4GB) |

*Note: Models evaluated on the disjoint split exhibit complete collapse, proving that the high Macro-F1 on the random split is due to source memorization.*

## Error Taxonomy (Source-Disjoint Split)

A taxonomy of 500 randomly sampled errors from the AraBERT disjoint evaluation:

- **Government & Official Statements** (238 errors): The model over-associates terms like "الحكومة" (Government) or "وزير" (Minister) with credibility, failing when unseen unreliable sources quote officials or when reliable sources report on denied rumors.
- **Source-Specific Artifacts / Boilerplate** (108 errors): Reliance on unseen publisher-specific sign-offs or location tags that the model mapped incorrectly (e.g., "النهار أونلاين").
- **General/Ambiguous Context** (52 errors): Articles where the textual claim lacks enough context for zero-shot credibility assignment.
- **Health & COVID-19** (44 errors): Topics like vaccines or the pandemic where the semantic boundaries of "fake" vs "real" require external knowledge grounding, which the model lacks.
- **Crime, Economics & Sports** (58 errors): Niche topics where style varies wildly across different news networks, confounding the model.

## Requirements and Setup

- Python 3.9+
- `torch`, `transformers`, `pandas`, `scikit-learn`
- NVIDIA GPU with at least 4GB VRAM (e.g., RTX 3050) is recommended for AraBERT training.

To reproduce the findings:
1. Run `notebooks/A_data_audit.py` to audit leakage and build the disjoint splits.
2. Run `notebooks/B2_development_transformers.py` to train baseline and transformer models.
3. Run `notebooks/C_hard_evaluation.py` to evaluate on the source-disjoint split.
4. Run `notebooks/D_taxonomy_generator.py` to extract the error taxonomy.
