# %% [markdown]
# # 14 — Explainability (Phase G5)
#
# Integrated Gradients on 50 correct + 50 incorrect examples. Diagnostic only:
# attributions are NOT causal explanations, and lexical buckets are coarse.

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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import TransformerDataset, map_labels
from src.models.transformer_clf import TransformerClassifier
from src.evaluation.explainability import (
    integrated_gradients_attribution,
    attribution_bucket_summary,
)

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
MAX_LEN = 128
CHECKPOINT = os.environ.get("NEWLIES_CKPT", "experiments/arabert_v02_src_len128_seed42.pt")
N_PER_GROUP = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = TransformerClassifier(MODEL_NAME, num_classes=3).to(device)
blob = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(blob.get("model_state_dict", blob))
model.eval()

test_df = pd.read_parquet("data/splits/test_src.parquet")
test_df, _ = map_labels(test_df)
loader = DataLoader(TransformerDataset(test_df["title"] + " " + test_df["text"],
                                       test_df["label_id"], tokenizer, max_len=MAX_LEN),
                    batch_size=32)

preds, trues = [], []
with torch.no_grad():
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds.extend(out.argmax(dim=1).cpu().tolist())
        trues.extend(batch["labels"].tolist())

eval_df = test_df.reset_index(drop=True).iloc[:len(preds)].copy()
eval_df["pred_id"] = preds

correct_idx = eval_df[eval_df["pred_id"] == eval_df["label_id"]].index[:N_PER_GROUP]
wrong_idx = eval_df[eval_df["pred_id"] != eval_df["label_id"]].index[:N_PER_GROUP]

OUT = Path("experiments/explainability")
OUT.mkdir(exist_ok=True)

records = []
for tag, idx_list in (("correct", correct_idx), ("incorrect", wrong_idx)):
    bucket_totals = {}
    for i in idx_list:
        text = str(eval_df.at[i, "title"]) + " " + str(eval_df.at[i, "text"])
        res = integrated_gradients_attribution(model, tokenizer, text, device,
                                               max_length=MAX_LEN)
        records.append({"group": tag, "row": int(i),
                        "predicted_class": res["predicted_class"],
                        "probs": res["probs"]})
        mass = sum(t["attribution"] for t in res["tokens"]) or 1e-9
        for t in res["tokens"]:
            b = t["bucket"]
            bucket_totals[b] = bucket_totals.get(b, 0.0) + t["attribution"] / mass
    print(f"\n[{tag}] avg attribution share per article (n={len(idx_list)}):")
    for k, v in sorted(bucket_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s} {v / max(1, len(idx_list)):.3f}")

pd.DataFrame(records).to_csv(OUT / "ig_records.csv", index=False)
print("\nInterpretation contract: diagnostic evidence about WHERE the model looks;"
      " causal claims require ablations (artifact sanitization, split control).")
