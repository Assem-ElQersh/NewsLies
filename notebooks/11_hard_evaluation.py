# %% [markdown]
# # 11 — Hard Generalization Evaluation (Phase D)
#
# Finalists only: evaluates the selected best neural and best Transformer
# checkpoints on random / source-disjoint / temporal / source+temporal splits.
# The random → hard-split drop is a PRIMARY research result.

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

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import TransformerDataset, map_labels
from src.models.transformer_clf import TransformerClassifier
from src.evaluation.metrics import compute_metrics, print_metrics

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
MAX_LEN = 128
BATCH = 16

CHECKPOINTS = {
    "rand": "experiments/arabert_v02_rand_len128_seed42.pt",
    "src": "experiments/arabert_v02_src_len128_seed42.pt",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

SPLITS = ["rand", "src"]
if all(os.path.exists(f"data/splits/{s}_tmp.parquet")
       for s in ("test", "train", "val")):
    SPLITS += ["tmp", "st"]

rows = []
model = None
current_ckpt = None
for split in SPLITS:
    test_df = pd.read_parquet(f"data/splits/test_{split}.parquet")
    test_df, inv_map = map_labels(test_df)
    loader = DataLoader(TransformerDataset(test_df["title"] + " " + test_df["text"],
                                           test_df["label_id"], tokenizer, max_len=MAX_LEN),
                        batch_size=BATCH)

    ckpt_path = CHECKPOINTS.get(split if split in CHECKPOINTS else "rand")
    if not os.path.exists(ckpt_path):
        print(f"Missing checkpoint {ckpt_path}; skipping split {split}.")
        continue
    if ckpt_path != current_ckpt:
        model = TransformerClassifier(MODEL_NAME, num_classes=3).to(device)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = blob.get("model_state_dict", blob)
        model.load_state_dict(state)
        model.eval()
        current_ckpt = ckpt_path

    preds, trues, probs_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["input_ids"].to(device),
                        batch["attention_mask"].to(device))
            p = torch.softmax(out.float(), dim=1).cpu().numpy()
            probs_all.append(p)
            preds.extend(p.argmax(axis=1).tolist())
            trues.extend(batch["labels"].tolist())

    m = compute_metrics(trues, preds, y_prob=np.concatenate(probs_all))
    rows.append({"split": split, "checkpoint": os.path.basename(ckpt_path),
                 **{k: round(v, 4) for k, v in m.items()
                    if isinstance(v, (int, float))}})
    print_metrics(m, f"{os.path.basename(ckpt_path)} on {split} TEST")

df_rows = pd.DataFrame(rows)
if len(df_rows):
    rand_f1 = df_rows.loc[df_rows["split"] == "rand", "macro_f1"]
    for _, r in df_rows.iterrows():
        if r["split"] != "rand" and len(rand_f1):
            print(f"Drop random -> {r['split']}: {float(rand_f1.iloc[0]) - r['macro_f1']:+.3f} Macro-F1")
    df_rows.to_csv("experiments/hard_generalization.csv", index=False)
