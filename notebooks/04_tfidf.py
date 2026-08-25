# %% [markdown]
# # 04 — TF-IDF + LinearSVM baseline (E1)
#
# Same development split as all other E-models. Controlled: no subsampling
# unless explicitly configured identically across models.

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

import joblib
import pandas as pd
import yaml
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.dataset import map_labels
from src.evaluation.metrics import compute_metrics, print_metrics
from src.training.registry import log_experiment, git_commit

CFG = yaml.safe_load(open("configs/tfidf.yaml"))
MAX_TRAIN = CFG.get("max_train_articles")

train_df = pd.read_parquet("data/splits/train_rand.parquet")
val_df = pd.read_parquet("data/splits/val_rand.parquet")
test_df = pd.read_parquet("data/splits/test_rand.parquet")
train_df, inv_map = map_labels(train_df)
val_df, _ = map_labels(val_df)
test_df, _ = map_labels(test_df)

if MAX_TRAIN and MAX_TRAIN < len(train_df):
    train_df = train_df.sample(n=MAX_TRAIN, random_state=42)

def join(d, include_title=True):
    t = d["title"].fillna("").astype(str) + " "
    return (t + d["text"].fillna("").astype(str)) if include_title else d["text"].fillna("").astype(str)

wv = TfidfVectorizer(**CFG["vectorizer"]["word"])
cv = TfidfVectorizer(**CFG["vectorizer"]["char"])

Xtr = hstack([wv.fit_transform(join(train_df)), cv.fit_transform(join(train_df))]).tocsr()
Xva = hstack([wv.transform(join(val_df)), cv.transform(join(val_df))]).tocsr()
ytr, yva = train_df["label_id"], val_df["label_id"]

clf = LinearSVC(C=CFG["classifier"]["C"],
                class_weight=CFG["classifier"]["class_weight"],
                max_iter=CFG["classifier"]["max_iter"])
clf.fit(Xtr, ytr)
preds_val = clf.predict(Xva)
metrics = compute_metrics(yva, preds_val)
print_metrics(metrics, "TF-IDF+SVM (val)")

joblib.dump({"word_vec": wv, "char_vec": cv, "clf": clf}, "experiments/tfidf_svm.joblib")
log_experiment({
    **{k: CFG.get(k) for k in ("experiment_id",)},
    "model": "tfidf_linear_svc", "split_type": "random", "split_seed": 42,
    "training_seed": 42, "max_length": None,
    "train_size": len(train_df), "validation_size": len(val_df), "test_size": len(test_df),
    "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    "git_commit": git_commit(),
})
