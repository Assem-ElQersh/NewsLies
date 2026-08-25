# %% [markdown]
# # 07 — AraBERTv0.2-base (E4) — CONTROLLED split comparison
#
# Phase B4: the SAME configuration is trained on the random split AND the
# source-disjoint split. The only experimental variable is the split.
# Context length comes from configs/arabert_v02.yaml; run this notebook three
# times with MAX_LENGTH in {128, 256, 512} for the context-length ablation.
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
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import TransformerDataset, map_labels
from src.data.preprocessing import build_preprocessor
from src.models.transformer_clf import TransformerClassifier
from src.training.trainer import Trainer
from src.training.losses import get_loss_fn
from src.training.scheduler import get_scheduler
from src.training.seed import seed_everything
from src.evaluation.metrics import print_metrics

CFG = yaml.safe_load(open("configs/arabert_v02.yaml"))

SPLIT = os.environ.get("NEWLIES_SPLIT", "rand")
MAX_LENGTH = int(os.environ.get("NEWLIES_MAXLEN", CFG["max_length"]))
SEED = int(os.environ.get("NEWLIES_SEED", CFG["seed"]))
MAX_TRAIN = CFG.get("max_train_articles")

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_parquet(f"data/splits/train_{SPLIT}.parquet")
val_df = pd.read_parquet(f"data/splits/val_{SPLIT}.parquet")
test_df = pd.read_parquet(f"data/splits/test_{SPLIT}.parquet")
train_df, inv_map = map_labels(train_df)
val_df, _ = map_labels(val_df)
test_df, _ = map_labels(test_df)
print(f"Split: {SPLIT} | max_len: {MAX_LENGTH} | seed: {SEED}")

if MAX_TRAIN and MAX_TRAIN < len(train_df):
    train_df = train_df.sample(n=MAX_TRAIN, random_state=SEED)

prep = build_preprocessor("transformer", model_name=CFG["hf_model_name"])
def prepared(df):
    return ((df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str))
            .map(prep))

tokenizer = AutoTokenizer.from_pretrained(CFG["hf_model_name"])
t_cfg = CFG["training"]

def make_loader(df, shuffle):
    ds = TransformerDataset(prepared(df), df["label_id"], tokenizer, max_len=MAX_LENGTH)
    return DataLoader(ds, batch_size=t_cfg["batch_size"], shuffle=shuffle)

train_loader = make_loader(train_df, True)
val_loader = make_loader(val_df, False)
test_loader = make_loader(test_df, False)

model = TransformerClassifier(CFG["hf_model_name"], num_classes=len(inv_map))
optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"])
total_steps = len(train_loader) * t_cfg["epochs"] // t_cfg["grad_accum_steps"]
scheduler = get_scheduler(optimizer,
                          warmup_steps=int(total_steps * t_cfg["scheduler"]["warmup_ratio"]),
                          total_steps=total_steps)
loss_fn = get_loss_fn(t_cfg["loss"])

trainer = Trainer(model, train_loader, val_loader, optimizer, loss_fn, device,
                  scheduler=scheduler, grad_accum_steps=t_cfg["grad_accum_steps"],
                  use_amp=t_cfg["use_amp"])
save_path = f"experiments/arabert_v02_{SPLIT}_len{MAX_LENGTH}_seed{SEED}.pt"
best = trainer.train(epochs=t_cfg["epochs"], save_path=save_path,
                     patience=t_cfg["patience"])

test_metrics = trainer.evaluate(test_loader)
print_metrics(test_metrics, f"AraBERTv0.2 ({SPLIT} TEST, len={MAX_LENGTH}, seed={SEED})")

import json
with open(f"experiments/arabert_v02_{SPLIT}_len{MAX_LENGTH}_seed{SEED}_metrics.json", "w") as f:
    json.dump({k: v for k, v in test_metrics.items() if k != "y_prob"},
              f, indent=2, default=str)

# %% [markdown]
# ## Controlled-comparison reminder
#
# Run with NEWLIES_SPLIT=rand and NEWLIES_SPLIT=src keeping every other knob
# identical. Report Macro-F1 drop (random → source-disjoint) as the primary
# leakage evidence. Do not compare these runs against historical numbers that
# used different budgets/lengths.
