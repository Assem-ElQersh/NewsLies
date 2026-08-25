# Research Versions Archive

This directory serves as a chronological ledger of the experimental phases and models
tested in the NewsLies project. Active, deployment-ready code lives at the repository
root and `docs/`.

## v1_lstm_baseline
Initial implementation phase. LSTM model over a small custom-stemmed Arabic vocabulary.
- Discovered catastrophic source-leakage symptoms (the model largely mapped publisher
  style to credibility instead of semantics).
- Contains legacy `docs/` artifacts including the v1 ONNX bundle (`vocab.json`,
  `stopwords.json`, `model.onnx`).

## v2_arabert_hf
Second iteration upgrading to `aubmindlab/bert-base-arabertv02` (**AraBERTv0.2-base**).
- Validated the leakage hypothesis: ~0.878 Macro-F1 on random splits vs ~0.370 on
  source-disjoint splits (different training budgets — preliminary numbers).
- Modernized web deployment via Hugging Face `@xenova/transformers` tokenization and an
  optimized ONNX model served from the HF Hub.
- Historical caveat fixed during the v3 audit: the loader read `published_date` but the
  raw AFND field is `published date`; the earlier claim of "100% missing dates" was a
  loader artifact. Verified reality (full re-audit): ~45% of raw date values are
  missing/invalid and some are future-dated; the rest parse as ISO-8601 timestamps,
  making temporal evaluation possible on the dated subset.

## v3_controlled_research_system (current)
Repository refactored from a score-chasing pipeline into a hypothesis-driven protocol:
- Config-driven experiments (`configs/*.yaml`) with an experiment registry recording
  split fingerprints and configuration hashes.
- Controlled random-vs-source-disjoint AraBERT comparison (single variable: the split).
- Audited duplicates (SHA-256 exact + MinHash/LSH→exact-Jaccard→connected-components
  near-duplicates) used as atomic units in splitting rather than silent deletion.
- Rebuilt source-disjoint allocator with class-proportion tolerance; temporal and
  source+temporal benchmarks added after the date-field audit.
- Statistical layer: grouped bootstrap by source, Δ-Macro-F1 CIs, five-seed protocol.
- Calibration (temperature scaling on validation only), Integrated-Gradients diagnostics.
- Distillation path to a deployment student with an explicit robustness acceptance gate;
  browser demo renamed to "Arabic News Credibility Classifier" with calibrated confidence
  and uncertainty display.

## Notebooks lifecycle (v3 restructure)
The original phase notebooks (`A_data_audit`, `B_development`, `B2_*`,
`C_hard_evaluation`, `D_taxonomy_generator`, `AFND_End_to_End_Kaggle`) were removed
from the working tree when v3 landed. They remain fully recoverable from Git history
(last seen in the commit that introduced `src/pipelines.py`) and their executed
artifacts are preserved under `results/` and registered as HIST-* entries in
`experiments/registry.json`. Their replacements: numbered `notebooks/00–15*.py`
(local/dev interfaces) and `notebooks/kaggle/K1–K6` (canonical training stages).

Reasons for removal, per `experiments/repo_audit.md`:
- duplicated model/split/dedup logic with diverging hyperparameters vs `src/`;
- broken when run as real notebooks (`__file__` usage);
- built on the pre-audit loader (publication dates silently lost) and the
  order-dependent LSH dedup / count-only disjoint split;
- not budget-controlled, so unsuitable as citable comparisons.
