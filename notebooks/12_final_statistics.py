# %% [markdown]
# # 12 — Final Statistics (Phase G)
#
# 5 training seeds on the FINAL model, grouped bootstrap CIs by source for
# source-disjoint evaluation, and Δ Macro-F1 model comparisons with 95% CI.

# %%
import json
import os
import sys
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
for p in [HERE] + list(HERE.parents):
    if (p / "src" / "data" / "loader.py").exists():
        ROOT = p
        break
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.statistical import paired_model_comparison, multi_seed_summary
from sklearn.metrics import f1_score

SEEDS = [13, 42, 71, 101, 202]
results_dir = Path("experiments/final_statistics")
results_dir.mkdir(exist_ok=True)

per_seed = []
for seed in SEEDS:
    path = results_dir / f"seed_{seed}_predictions.npz"
    if not path.exists():
        print(f"Missing {path}. Run the final-model training script per seed "
              "(07_arabert with NEWLIES_SEED) saving predictions first.")
        continue
    blob = np.load(path, allow_pickle=True)
    per_seed.append({"seed": seed,
                     "macro_f1": f1_score(blob["y_true"], blob["y_pred"], average="macro")})

if per_seed:
    summary = multi_seed_summary(per_seed)
    print(json.dumps(summary, indent=2))
    with open(results_dir / "multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

# %% [markdown]
# ## Grouped bootstrap + model comparison (works on saved prediction arrays)

# %%
pred_files = sorted(results_dir.glob("*_predictions.npz"))
if len(pred_files) >= 1:
    a = np.load(pred_files[0], allow_pickle=True)
    y_true, y_pred_a, groups = a["y_true"], a["y_pred"], a.get("groups")

    ci = bootstrap_ci(lambda t, p: f1_score(t, p, average="macro"),
                      y_true, y_pred_a, n_boot=1000,
                      groups=groups if groups is not None else None)
    print("Macro-F1 bootstrap CI:", json.dumps(ci, indent=2))
    with open(results_dir / "bootstrap_ci.json", "w") as f:
        json.dump(ci, f, indent=2)

if len(pred_files) >= 2:
    b = np.load(pred_files[1], allow_pickle=True)
    comp = paired_model_comparison(
        None, a["y_true"], a["y_pred"], b["y_pred"],
        n_boot=1000,
        groups=a["groups"] if "groups" in a else None)
    print("\nModel comparison:", json.dumps(comp, indent=2))
    with open(results_dir / "model_comparison.json", "w") as f:
        json.dump(comp, f, indent=2)

# %%
print("""
Reporting contract:
- Source-disjoint metrics: bootstrap resampled BY SOURCE (report unit explicitly).
- Random/temporal article-level analyses: article-level bootstrap acceptable.
- Model comparisons report Δ Macro-F1 with 95% CI; overlapping CI → 'no
  convincing evidence of superiority'.
""")
