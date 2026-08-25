# %% [markdown]
# # 15 — Distillation & Deployment (Phase H)
#
# Distills the frozen research teacher into compact students, verifies that
# robustness survives (random AND source-disjoint), then exports the chosen
# student to docs/ as ONNX + tokenizer + calibration.json.
# > **Training runs on Kaggle** — use `notebooks/kaggle/` stages for GPU execution; this notebook remains the local/dev interface.


# %%
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

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import TransformerDataset, map_labels
from src.models.bigru_attention import BiGRUAttention
from src.models.distillation import DistillationTrainer, SmallTransformerStudent
from src.models.transformer_clf import TransformerClassifier
from src.data.dataset import build_vocab, ClassicalDataset
from src.evaluation.metrics import print_metrics

CFG = yaml.safe_load(open("configs/distillation.yaml"))
SEED = 42
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_CKPT = os.environ.get(
    "NEWLIES_TEACHER", "experiments/arabert_v02_src_len128_seed42.pt")
TEACHER_NAME = "aubmindlab/bert-base-arabertv02"

teacher = TransformerClassifier(TEACHER_NAME, num_classes=3)
blob = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
teacher.load_state_dict(blob.get("model_state_dict", blob))
tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)

splits = {}
for s in ("rand", "src"):
    tr = pd.read_parquet(f"data/splits/train_{s}.parquet")
    va = pd.read_parquet(f"data/splits/val_{s}.parquet")
    te = pd.read_parquet(f"data/splits/test_{s}.parquet")
    splits[s] = tuple(map_labels(d)[0] for d in (tr, va, te))

d_cfg = CFG["distillation"]
alpha, T = d_cfg["alpha"], d_cfg["T"]

# %% [markdown]
# ## Student 1: BiGRU + attention (teacher tokenization drives inputs; the
# student consumes word-id sequences from its own vocabulary)

# %%
results = {}

tr_rand, va_rand, te_rand = splits["rand"]
vocab = build_vocab(tr_rand["title"] + " " + tr_rand["text"], vocab_size=30000, seed=SEED)

def classical_loaders(train_df, val_df, test_df):
    mk = lambda df, sh: DataLoader(
        ClassicalDataset(df["title"] + " " + df["text"], df["label_id"], vocab,
                         max_len=MAX_LEN),
        batch_size=d_cfg["batch_size"], shuffle=sh)
    return mk(train_df, True), mk(val_df, False), mk(test_df, False)

train_loader_r, val_loader_r, test_loader_r = classical_loaders(tr_rand, va_rand, te_rand)

class BigruStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = BiGRUAttention(vocab_size=len(vocab), num_classes=3)

    def forward(self, input_ids, attention_mask=None, **_):
        return self.net(input_ids)

student_bigru = BigruStudent()

distiller = DistillationTrainer(student_bigru, teacher, train_loader_r, val_loader_r,
                                device, alpha=alpha, T=T, lr=d_cfg["lr"])
best = distiller.fit(epochs=d_cfg["epochs"], save_path="experiments/student_bigru.pt")
print_metrics(distiller.evaluate(test_loader_r), "Student BiGRU (random TEST)")
results["bigru_random"] = distiller.evaluate(test_loader_r)["macro_f1"]

# %%
_, _, te_src = splits["src"]
_, _, test_loader_s = classical_loaders(*splits["src"])
m_src = distiller.evaluate(test_loader_s)
print_metrics(m_src, "Student BiGRU (source-disjoint TEST)")
results["bigru_disjoint"] = m_src["macro_f1"]

print("""
ACCEPTANCE GATE: compare the student's random vs source-disjoint drop with the
teacher's. A student that keeps random performance but collapses further on
the disjoint split is rejected regardless of size savings.
""")

# %%
import json

with open("experiments/distillation_results.json", "w") as f:
    json.dump({
        "alpha": alpha, "T": T,
        "student_bigru": {
            "random_macro_f1": results["bigru_random"],
            "source_disjoint_macro_f1": results["bigru_disjoint"],
        },
    }, f, indent=2)

# %% [markdown]
# ## Export chosen student to docs/ (ONNX + tokenizer + calibration)
# Run scripts/export_model.py after selecting the final student here.

# %%
print("""
Next steps (run as scripts, not in-notebook):
  python export_model.py --checkpoint experiments/student_bigru.pt \\
      --arch bigru --vocab-size 30000 --out docs
Then benchmark:
  python -m src.inference.benchmark --model-dir docs
""")

