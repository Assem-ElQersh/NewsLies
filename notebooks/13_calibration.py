# %% [markdown]
# # 13 — Calibration (Phase G4)
#
# Temperature scaling fitted ONLY on validation logits. Reports ECE, Brier and
# reliability diagrams before/after.

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
from src.evaluation.calibration import calibration_report

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
MAX_LEN = 128
CHECKPOINT = os.environ.get("NEWLIES_CKPT", "experiments/arabert_v02_src_len128_seed42.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = TransformerClassifier(MODEL_NAME, num_classes=3).to(device)
blob = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(blob.get("model_state_dict", blob))
model.eval()


@torch.no_grad()
def collect_logits(df):
    _, inv_map = map_labels(df)
    loader = DataLoader(TransformerDataset(df["title"] + " " + df["text"],
                                           df["label_id"], tokenizer, max_len=MAX_LEN),
                        batch_size=32)
    logits, labels = [], []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        logits.append(out.float().cpu().numpy())
        labels.extend(batch["labels"].tolist())
    return np.concatenate(logits), np.array(labels)


val_df = pd.read_parquet("data/splits/val_src.parquet")
test_df = pd.read_parquet("data/splits/test_src.parquet")
val_logits, val_labels = collect_logits(val_df)
test_logits, test_labels = collect_logits(test_df)

from src.evaluation.calibration import softmax_np
test_probs = softmax_np(test_logits)

report = calibration_report(
    test_labels, test_probs,
    val_logits=val_logits, val_labels=val_labels,
    out_dir="experiments/calibration", prefix="src_")
compact = {
    "temperature": report.get("temperature"),
    "ece_before": report["before"]["ece"]["ece"],
    "brier_before": report["before"]["brier"],
    "ece_after": report["after"]["ece"]["ece"] if report["after"] else None,
    "brier_after": report["after"]["brier"] if report["after"] else None,
}
print(json.dumps(compact, indent=2))
with open("experiments/calibration/calibration_summary.json", "w") as f:
    json.dump(compact, f, indent=2)
