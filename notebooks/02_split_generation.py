# %% [markdown]
# # 02 — Split Generation (Phase A7)
#
# Generates random, source-disjoint, temporal and source+temporal splits from
# the audit-annotated corpus. Duplicate clusters are atomic units; the
# source-disjoint allocator enforces class-proportion tolerance.

# %%
import hashlib
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

import pandas as pd

from src.data.splits import (
    random_split,
    source_disjoint_split,
    temporal_split,
    source_temporal_split,
    split_fingerprint,
)

OUT = Path("data/splits")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# %%
annotated_path = OUT / "_annotated_full.parquet"
if annotated_path.exists():
    df = pd.read_parquet(annotated_path)
    print(f"Loaded annotated corpus: {len(df)} articles")
else:
    raise SystemExit("Run 01_data_audit.py first (annotated corpus missing).")

# %%
train_rand, val_rand, test_rand = random_split(df, seed=SEED)

# %%
train_src, val_src, test_src = source_disjoint_split(df, seed=SEED,
                                                     num_candidates=500,
                                                     class_tolerance=0.03)

# %% [markdown]
# ## Temporal splits — only if dates verified usable by 01_data_audit.

# %%
try:
    train_tmp, val_tmp, test_tmp = temporal_split(df)
    train_st, val_st, test_st = source_temporal_split(df, seed=SEED)
    temporal_ok = True
except ValueError as e:
    print(f"Temporal splits skipped: {e}")
    temporal_ok = False

# %%
def save_pair(frame, name):
    frame.to_parquet(OUT / f"{name}.parquet", index=False)

for name, fr in [
    ("train_rand", train_rand), ("val_rand", val_rand), ("test_rand", test_rand),
    ("train_src", train_src), ("val_src", val_src), ("test_src", test_src),
]:
    save_pair(fr, name)

if temporal_ok:
    for name, fr in [
        ("train_tmp", train_tmp), ("val_tmp", val_tmp), ("test_tmp", test_tmp),
        ("train_st", train_st), ("val_st", val_st), ("test_st", test_st),
    ]:
        save_pair(fr, name)

meta = {
    "seed": SEED,
    "corpus_articles": int(len(df)),
    "random": {"fingerprint": split_fingerprint(train_rand)},
    "source_disjoint": {
        "fingerprint": split_fingerprint(pd.concat([train_src, val_src, test_src])),
        "class_tolerance": 0.03,
        "num_candidates": 500,
        "train_sources": sorted(train_src["source"].unique().tolist()),
        "test_sources": sorted(test_src["source"].unique().tolist()),
    },
}
if temporal_ok:
    meta["temporal"] = {"fingerprint": split_fingerprint(train_tmp)}
    meta["source_temporal"] = {"fingerprint": split_fingerprint(train_st)}

with open(OUT / "split_manifest.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(json.dumps(meta, indent=2, default=str)[:1200])

# %% [markdown]
# Sanity: verify zero overlap and duplicate-cluster integrity on the disjoint split.

# %%
assert len(set(train_src["source"]) & set(test_src["source"])) == 0
assert len(set(val_src["source"]) & set(test_src["source"])) == 0

for name, parts in [("rand", (train_rand, val_rand, test_rand)),
                    ("src", (train_src, val_src, test_src))]:
    tagged = pd.concat(
        [p.assign(_split=s) for s, p in zip(("train", "val", "test"), parts)]
    )
    clustered = tagged[tagged["near_duplicate_cluster_id"] >= 0]
    n_cross = int(clustered.groupby("near_duplicate_cluster_id")["_split"].nunique().gt(1).sum())
    print(f"{name}: near-dup clusters crossing splits: {n_cross}")
