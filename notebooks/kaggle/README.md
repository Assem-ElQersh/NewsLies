# Kaggle Pipeline — NewsLies

All model **training** happens on Kaggle. Local machine is used only for code edits,
lightweight inspection and reviewing returned artifacts.

Every stage below is a self-contained notebook (`K*.py`, jupytext format) that:
1. clones this repository,
2. resolves inputs from attached Kaggle datasets,
3. writes everything to `/kaggle/working` (which becomes the stage output).

## Stage contracts

| Stage | Notebook | Accelerator | Input dataset(s) | Produces |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `K1_audit_and_splits.py` | CPU (2–4 h) | AFND raw dataset | `_annotated_full.parquet`, all split parquets + `split_manifest.json`, `experiments/data_audit/*`, near-dup cluster assignments |
| 2 | `K2_baselines.py` | GPU T4 (1–2 h) | stage-1 output as `afnd-splits-v2` | E0–E3 metrics JSONs, baseline checkpoints |
| 3 | `K3_arabert_controlled.py` | GPU T4 x2 (per run 2–6 h) | `afnd-splits-v2` | AraBERTv0.2 metrics/predictions/checkpoints for one `(SPLIT, MAX_LEN, SEED)` cell of the controlled grid |
| 4 | `K4_camelbert_msa.py` | GPU T4 x2 | `afnd-splits-v2` | CAMeLBERT metrics/predictions/checkpoints |
| 5 | `K5_eval_stats_calib_explain.py` | GPU T4 | `afnd-splits-v2` + checkpoints + prediction `.npz`s from stages 3–4 | hard-generalization table, grouped-bootstrap CIs, Δ-F1 comparisons, calibration bundle, IG summaries |
| 6 | `K6_distill_export.py` | GPU T4 | `afnd-splits-v2` + chosen teacher checkpoint | distilled student metrics + `docs/` deployment bundle (ONNX + tokenizer + calibration.json) |

## Workflow

1. Run K1 once. Download its output, upload to Kaggle as a **new dataset** named
   `afnd-splits-v2` (or keep the chain via "New Dataset from Output").
2. Duplicate-and-edit K3 for each controlled-grid cell:
   - `SPLIT = rand` and `SPLIT = src` at `MAX_LEN = 128` (the mandatory controlled pair),
   - then `MAX_LEN ∈ {256, 512}` on both splits for the context-length ablation,
   - final-model seeds `[13, 42, 71, 101, 202]` only after the finalist is chosen.
   Every other knob stays fixed by construction (`src.pipelines.train_transformer`).
3. Bring produced artifacts back (download `/kaggle/working`) or chain datasets.
4. Never compare runs that differ in more than the declared variable; the registry
   entries written by each stage make violations auditable.

## Notes

- The AFND input path is auto-detected among common Kaggle layouts; no hardcoded
  mount assumptions beyond `sources.json` discovery.
- Registry (`experiments/registry.json`) travels with the repo clone inside the
  working dir — include it when archiving stage outputs so fingerprints persist.
- Historical numbers (≈0.878 random / ≈0.370 disjoint) are preliminary; treat
  K3's controlled pair as the citable comparison going forward.
