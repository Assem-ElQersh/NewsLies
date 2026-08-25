# %% [markdown]
# # 00 — Repository Audit
#
# Verifies that every src module imports cleanly and records the component
# inventory. The full audit table lives in experiments/repo_audit.md.

# %%
import importlib
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
else:
    raise RuntimeError("repo root not found")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

MODULES = [
    "src.data.loader",
    "src.data.cleaner",
    "src.data.duplicates",
    "src.data.splits",
    "src.data.preprocessing",
    "src.data.dataset",
    "src.models.lstm",
    "src.models.bigru_attention",
    "src.models.transformer_clf",
    "src.models.hierarchical",
    "src.models.distillation",
    "src.training.trainer",
    "src.training.losses",
    "src.training.scheduler",
    "src.training.seed",
    "src.training.registry",
    "src.evaluation.metrics",
    "src.evaluation.bootstrap",
    "src.evaluation.statistical",
    "src.evaluation.calibration",
    "src.evaluation.explainability",
    "src.evaluation.analysis",
    "src.inference.pipeline",
]

results = {}
for mod in MODULES:
    try:
        importlib.import_module(mod)
        results[mod] = "OK"
    except Exception as e:
        results[mod] = f"FAIL: {type(e).__name__}: {e}"

for mod, status in results.items():
    print(f"{mod:42s} {status}")

n_fail = sum(1 for s in results.values() if not s.startswith("OK"))
print(f"\n{n_fail} failed imports" if n_fail else "\nAll modules import cleanly.")

STALE_FILES = [
    "docs/vocab.json (v1 LSTM deployment bundle)",
    "docs/stopwords.json (v1; still used as optional classical stopword list)",
    "docs/model.onnx (legacy 4.3MB LSTM-era export; current demo loads from HF hub)",
]
print("\nStale artifacts recorded in repo_audit.md:")
for s in STALE_FILES:
    print(" -", s)
