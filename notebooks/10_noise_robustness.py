# %% [markdown]
# # 10 — Label-Noise Robustness (Phase F)
#
# A: article-level random flips at 5/10/15%.
# B: source-level structured flips (whole publishers relabeled) — models the
#    actual structural noise of source-derived supervision.
#
# Compares CE vs label smoothing vs focal under each regime. HARD STOP: if all
# losses behave essentially identically, stop exploring losses.
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

import copy
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import TransformerDataset, map_labels
from src.models.transformer_clf import TransformerClassifier
from src.training.trainer import Trainer
from src.training.losses import get_loss_fn, class_weights_from
from src.training.seed import seed_everything
from src.evaluation.metrics import print_metrics

SEED = 42
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15]
LOSSES = ["ce", "label_smoothing", "focal"]
MAX_LEN = 128
BATCH = 16
EPOCHS = 2
MODEL_NAME = "aubmindlab/bert-base-arabertv02"

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_parquet("data/splits/train_rand.parquet")
val_df = pd.read_parquet("data/splits/val_rand.parquet")
test_df = pd.read_parquet("data/splits/test_rand.parquet")
train_df, inv_map = map_labels(train_df)
val_df, _ = map_labels(val_df)
test_df, _ = map_labels(test_df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
val_loader = DataLoader(TransformerDataset(train_df["title"] + " " + train_df["text"],
                                           val_df["label_id"], tokenizer, max_len=MAX_LEN),
                        batch_size=BATCH)
test_loader = DataLoader(TransformerDataset(test_df["title"] + " " + test_df["text"],
                                            test_df["label_id"], tokenizer, max_len=MAX_LEN),
                         batch_size=BATCH)


def article_noise(labels, rate):
    rng = np.random.RandomState(SEED)
    labels = labels.copy()
    n = len(labels)
    k = int(rate * n)
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        choices = [c for c in range(len(inv_map)) if c != labels.iloc[i]]
        labels.iloc[i] = rng.choice(choices)
    return labels


def source_noise(df_labels_sources, rate):
    df, labels = df_labels_sources
    rng = np.random.RandomState(SEED)
    out = labels.copy()
    sources = df["source"].unique()
    k = int(rate * len(sources))
    flip = rng.choice(sources, size=k, replace=False)
    for s in flip:
        mask = (df["source"] == s).to_numpy()
        old = out[mask].mode()[0]
        choices = [c for c in range(len(inv_map)) if c != old]
        new = rng.choice(choices)
        out.loc[mask] = new
        print(f"  flipped source {s}: label {old} -> {new} ({mask.sum()} articles)")
    return out


results = []
for regime in ("article", "source"):
    for rate in NOISE_LEVELS:
        if rate == 0.0 and regime == "source":
            continue
        print(f"\n=== Regime {regime} | noise {rate:.0%} ===")
        noisy = (article_noise(train_df["label_id"], rate) if regime == "article"
                 else source_noise((train_df, train_df["label_id"]), rate))
        train_noisy = train_df.copy()
        train_noisy["label_id"] = noisy.astype(int)

        train_loader = DataLoader(
            TransformerDataset(train_noisy["title"] + " " + train_noisy["text"],
                               train_noisy["label_id"], tokenizer, max_len=MAX_LEN),
            batch_size=BATCH, shuffle=True)

        for loss_name in LOSSES:
            seed_everything(SEED)
            model = TransformerClassifier(MODEL_NAME, num_classes=len(inv_map))
            weights = None
            if loss_name != "ce":
                weights = class_weights_from(train_noisy["label_id"].to_numpy(),
                                             len(inv_map), scheme="balanced").to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
            loss_fn = get_loss_fn(loss_name, class_weights=weights)
            trainer = Trainer(model, train_loader, val_loader, opt, loss_fn, device)
            best = trainer.train(epochs=EPOCHS,
                                 save_path=f"experiments/noise_{regime}_{rate}_{loss_name}.pt",
                                 patience=2)
            tm = trainer.evaluate(test_loader)
            results.append({"regime": regime, "noise": rate, "loss": loss_name,
                            "best_val_macro_f1": best,
                            "test_macro_f1": tm["macro_f1"]})
            print_metrics(tm, f"{regime}/{rate}/{loss_name}")

with open("experiments/noise_robustness.json", "w") as f:
    json.dump(results, f, indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, regime in zip(axes, ("article", "source")):
    sub = [r for r in results if r["regime"] == regime]
    for loss_name in LOSSES:
        pts = sorted([(r["noise"], r["test_macro_f1"]) for r in sub if r["loss"] == loss_name])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=loss_name)
    ax.set_title(regime); ax.set_xlabel("noise level"); ax.set_ylabel("Macro-F1")
axes[0].legend()
fig.tight_layout()
fig.savefig("experiments/noise_robustness.png", dpi=150)
print("\nHARD STOP check: if the three loss curves overlap within ~±0.5 F1 at every level, "
      "stop loss exploration and report equivalence.")
