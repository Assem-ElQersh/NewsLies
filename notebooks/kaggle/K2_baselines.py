# %% [markdown]
# # K2 — Classical Baselines E0–E3 (Kaggle, GPU optional)
#
# Input: `afnd-splits-v2` dataset (stage-1 output).
# Output: metrics JSONs + LSTM/BiGRU checkpoints in /kaggle/working.
# Controlled: same random split, full training data, identical budget for E2/E3.

# %%
!if [ -d /kaggle/working/NewsLies/.git ]; then git -C /kaggle/working/NewsLies pull --ff-only; else git clone --depth 1 https://github.com/Assem-ElQersh/NewsLies.git /kaggle/working/NewsLies; fi
%cd /kaggle/working/NewsLies

# %%
import sys, json, os
sys.path.insert(0, "/kaggle/working/NewsLies")

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.pipelines import ctx
from src.data.dataset import ClassicalDataset, build_vocab, map_labels
from src.models.lstm import LSTMClassifier
from src.models.bigru_attention import BiGRUAttention
from src.training.trainer import Trainer
from src.training.losses import get_loss_fn
from src.training.seed import seed_everything
from src.evaluation.metrics import compute_metrics, print_metrics

env = ctx()
device = env["device"]
OUT = env["out_dir"]

train_df, val_df, test_df = [pd.read_parquet(env["splits_dir"] / f"{p}_rand.parquet")
                             for p in ("train", "val", "test")]
train_df, inv_map = map_labels(train_df)
val_df, _ = map_labels(val_df)
test_df, _ = map_labels(test_df)

results = {}

# %%
majority = train_df["label_id"].mode()[0]
preds = [majority] * len(val_df)
results["E0_majority_val"] = compute_metrics(val_df["label_id"], preds)["macro_f1"]
print(f"E0 majority Macro-F1 (val): {results['E0_majority_val']:.4f}")

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

join = lambda d: d["title"].fillna("").astype(str) + " " + d["text"].fillna("").astype(str)
wv = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
cv = TfidfVectorizer(max_features=50000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
Xtr = hstack([wv.fit_transform(join(train_df)), cv.fit_transform(join(train_df))]).tocsr()
Xva = hstack([wv.transform(join(val_df)), cv.transform(join(val_df))]).tocsr()
svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=3000).fit(Xtr, train_df["label_id"])
results["E1_tfidf_svm_val"] = compute_metrics(val_df["label_id"], svm.predict(Xva))["macro_f1"]
print(f"E1 TF-IDF+SVM Macro-F1 (val): {results['E1_tfidf_svm_val']:.4f}")

# %% [markdown]
# ## E2 LSTM and E3 BiGRU — identical data policy and budget

# %%
MAX_LEN = int(os.environ.get("NEWLIES_CLASSICAL_MAXLEN", 256))
BATCH = 64
EPOCHS = 5
vocab = build_vocab(train_df["title"] + " " + train_df["text"], vocab_size=30000, seed=42)
mk = lambda d, sh: DataLoader(ClassicalDataset(d["title"] + " " + d["text"], d["label_id"],
                                               vocab, max_len=MAX_LEN),
                              batch_size=BATCH, shuffle=sh)
loaders = (mk(train_df, True), mk(val_df, False), mk(test_df, False))

for name, cls, tag in (("E2_lstm", LSTMClassifier, "lstm"),
                       ("E3_bigru", BiGRUAttention, "bigru")):
    seed_everything(42)
    kwargs = dict(vocab_size=len(vocab), num_classes=3)
    if name == "E2_lstm":
        kwargs.update(embed_dim=100, hidden_dim=64)
    else:
        kwargs.update(embed_dim=300, hidden_dim=128)
    model = cls(**kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, *loaders[:2], opt, get_loss_fn("ce"), device)
    best = trainer.train(epochs=EPOCHS, save_path=str(OUT / f"{tag}_best.pt"), patience=3)
    tm = trainer.evaluate(loaders[2])
    results[name] = {"best_val_macro_f1": best,
                     "test": {k: v for k, v in tm.items() if isinstance(v, (int, float))}}
    print_metrics(tm, f"{name} (random TEST)")

json.dump(results, open(OUT / "baselines_results.json", "w"), indent=2, default=str)
print("\nSaved baselines_results.json")
