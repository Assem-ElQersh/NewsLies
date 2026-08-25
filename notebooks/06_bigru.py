# %% [markdown]
# # 06 — BiGRU + Self-Attention (E3)
#
# Same split, same vocabulary policy, same budget as E2.
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

from src.data.dataset import ClassicalDataset, build_vocab, map_labels
from src.models.bigru_attention import BiGRUAttention
from src.training.trainer import Trainer
from src.training.losses import get_loss_fn
from src.training.seed import seed_everything
from src.evaluation.metrics import print_metrics

CFG = yaml.safe_load(open("configs/bigru.yaml"))
seed_everything(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_parquet("data/splits/train_rand.parquet")
val_df = pd.read_parquet("data/splits/val_rand.parquet")
test_df = pd.read_parquet("data/splits/test_rand.parquet")
train_df, inv_map = map_labels(train_df)
val_df, _ = map_labels(val_df)
test_df, _ = map_labels(test_df)

MAX_TRAIN = None
if MAX_TRAIN and MAX_TRAIN < len(train_df):
    train_df = train_df.sample(n=MAX_TRAIN, random_state=CFG["seed"])

vocab = build_vocab((train_df["title"] + " " + train_df["text"]),
                    vocab_size=CFG["vocab_size"], seed=CFG["seed"])

def make_ds(df):
    return ClassicalDataset(df["title"] + " " + df["text"], df["label_id"],
                            vocab, max_len=CFG["max_length"])

train_loader = DataLoader(make_ds(train_df), batch_size=CFG["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(make_ds(val_df), batch_size=CFG["training"]["batch_size"])
test_loader = DataLoader(make_ds(test_df), batch_size=CFG["training"]["batch_size"])

model = BiGRUAttention(vocab_size=len(vocab),
                       embed_dim=CFG["architecture"]["embed_dim"],
                       hidden_dim=CFG["architecture"]["hidden_dim"],
                       num_classes=len(inv_map))
opt = torch.optim.Adam(model.parameters(), lr=CFG["training"]["lr"])
loss_fn = get_loss_fn(CFG["training"]["loss"])

trainer = Trainer(model, train_loader, val_loader, opt, loss_fn, device,
                  use_amp=CFG["training"]["use_amp"])
best = trainer.train(epochs=CFG["training"]["epochs"],
                     save_path="experiments/bigru_best.pt",
                     patience=CFG["training"]["patience"])
print_metrics(trainer.evaluate(test_loader), "BiGRU+Attn (random-split TEST)")
