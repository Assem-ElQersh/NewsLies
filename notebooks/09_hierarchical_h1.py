# %% [markdown]
# # 09 — Hierarchical H1 (frozen encoder + chunk attention)
#
# GATE (hard stop): H1 must beat the flat-512 transformer on the SAME
# source-disjoint split by a practically meaningful margin, otherwise STOP.
# Run only after the flat-512 baseline exists on this split.
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.data.dataset import map_labels
from src.models.hierarchical import HierarchicalClassifier, cache_body_embeddings
from src.training.losses import get_loss_fn
from src.training.seed import seed_everything
from src.evaluation.metrics import compute_metrics, print_metrics

CFG = yaml.safe_load(open("configs/hierarchical.yaml"))
seed_everything(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_parquet("data/splits/train_src.parquet")
val_df = pd.read_parquet("data/splits/val_src.parquet")
test_df = pd.read_parquet("data/splits/test_src.parquet")
train_df, inv_map = map_labels(train_df)
val_df, _ = map_labels(val_df)
test_df, _ = map_labels(test_df)

MAX_TRAIN = None
if MAX_TRAIN and MAX_TRAIN < len(train_df):
    train_df = train_df.sample(n=MAX_TRAIN, random_state=CFG["seed"])

encoder_name = CFG["encoder"]["hf_model_name"]
tokenizer = AutoTokenizer.from_pretrained(encoder_name)
CHUNK, STRIDE = CFG["body_chunking"]["chunk_len"], CFG["body_chunking"]["stride"]
TITLE_LEN = CFG["max_length_title"]

# %% [markdown]
# ## Cache frozen chunk embeddings once

# %%
def get_or_cache(df, tag):
    path = f"experiments/h1_body_{tag}.pt"
    if os.path.exists(path):
        blob = torch.load(path)
        return blob["embs"], blob["slices"]
    frozen = AutoModel.from_pretrained(encoder_name).to(device).eval()
    out = cache_body_embeddings(frozen, tokenizer,
                                df["text"].fillna("").astype(str).tolist(),
                                chunk_len=CHUNK, stride=STRIDE, device=device)
    torch.save({"embs": out[0], "slices": out[1]}, path)
    del frozen
    torch.cuda.empty_cache()
    return out

train_body = get_or_cache(train_df, "train")
val_body = get_or_cache(val_df, "val")
test_body = get_or_cache(test_df, "test")

# %%
class H1Dataset(Dataset):
    def __init__(self, df):
        enc = tokenizer(df["title"].fillna("").astype(str).tolist(),
                        truncation=True, max_length=TITLE_LEN,
                        padding="max_length", return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = df["label_id"].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return i


class H1Collator:
    """Batches title ids + per-document slice windows over the cached
    chunk-embedding matrix."""

    def __init__(self, dataset: H1Dataset, body_embs: torch.Tensor, slices):
        self.ds = dataset
        self.body_embs = body_embs
        self.slices = slices

    def __call__(self, indices):
        input_ids = self.ds.input_ids[list(indices)]
        mask = self.ds.attention_mask[list(indices)]
        labels = torch.tensor(self.ds.labels[list(indices)], dtype=torch.long)

        flat_idx = []
        rel_slices = []
        pos = 0
        for i in indices:
            s, e = self.slices[i]
            n = e - s
            flat_idx.append(torch.arange(s, e))
            rel_slices.append((pos, pos + n))
            pos += n
        flat_idx = torch.cat(flat_idx)
        body = self.body_embs[flat_idx]
        return {"input_ids": input_ids, "attention_mask": mask,
                "labels": labels, "body_embs": body, "slices": rel_slices}


class H1Model(nn.Module):
    def __init__(self, h1: HierarchicalClassifier):
        super().__init__()
        self.h1 = h1

    def forward(self, input_ids=None, attention_mask=None, body_embs=None,
                slices=None, **_):
        return self.h1(title_ids=input_ids.to(device),
                       title_mask=attention_mask.to(device),
                       cached_body_embs=body_embs, group_slices=slices)


def make_loader(df, body_cache, shuffle):
    ds = H1Dataset(df)
    coll = H1Collator(ds, body_cache[0], body_cache[1])
    return DataLoader(ds, batch_size=CFG["training"]["batch_size"], shuffle=shuffle,
                      collate_fn=coll), len(ds)


model = H1Model(HierarchicalClassifier(
    encoder_name, num_classes=len(inv_map),
    chunk_len=CHUNK, stride=STRIDE,
    dropout=CFG["architecture"]["dropout"],
    freeze_encoder=CFG["encoder"]["freeze"])).to(device)

train_loader, _ = make_loader(train_df, train_body, True)
val_loader, _ = make_loader(val_df, val_body, False)
test_loader, _ = make_loader(test_df, test_body, False)

t_cfg = CFG["training"]
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=t_cfg["lr"])
loss_fn = get_loss_fn(t_cfg["loss"])

history, best_f1, patience_left = [], -1.0, t_cfg["patience"]
best_state = None
for epoch in range(t_cfg["epochs"]):
    model.train()
    total = 0.0
    for batch in train_loader:
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=t_cfg["use_amp"]):
            logits = model(**batch)
            loss = loss_fn(logits.float(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(**batch)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            trues.extend(batch["labels"].tolist())
    m = compute_metrics(trues, preds)
    print(f"Epoch {epoch+1}: loss {total/len(train_loader):.4f} | val macro_f1 {m['macro_f1']:.4f}")
    if m["macro_f1"] > best_f1:
        best_f1 = m["macro_f1"]
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_left = t_cfg["patience"]
    else:
        patience_left -= 1
        if patience_left == 0:
            break

if best_state:
    model.load_state_dict(best_state)
torch.save({"model_state_dict": {k: v for k, v in model.state_dict().items()},
            "val_macro_f1": best_f1}, "experiments/h1_best.pt")

preds, trues = [], []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        logits = model(**batch)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        trues.extend(batch["labels"].tolist())
print_metrics(compute_metrics(trues, preds), "H1 (source-disjoint TEST)")
print("\nGATE: compare against flat-512 on the same split before any H2 work.")
