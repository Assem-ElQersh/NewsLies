# %% [markdown]
# # K5 — Hard Evaluation + Statistics + Calibration + Explainability (Kaggle, GPU)
#
# Inputs: `afnd-splits-v2` + checkpoints and prediction .npz files produced by
# K3/K4 (attach them as a dataset, e.g. `newslies-checkpoints`).
# Output: hard-generalization table, grouped-bootstrap CIs, Δ comparisons,
# calibration bundle, Integrated-Gradients bucket summaries.

# %%
!if [ -d /kaggle/working/NewsLies/.git ]; then git -C /kaggle/working/NewsLies pull --ff-only; else git clone --depth 1 https://github.com/Assem-ElQersh/NewsLies.git /kaggle/working/NewsLies; fi
%cd /kaggle/working/NewsLies

# %%
import sys
sys.path.insert(0, "/kaggle/working/NewsLies")

PARAMS = {
    "TEACHER_MODEL": "aubmindlab/bert-base-arabertv02",
    "CHECKPOINT_GLOB": "/kaggle/input/newslies-checkpoints/*.pt",
}

# %% [markdown]
# ## 1. Hard generalization: one finalist checkpoint across all splits

# %%
import glob, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.pipelines import ctx
from src.data.dataset import TransformerDataset, map_labels
from src.models.transformer_clf import TransformerClassifier
from src.evaluation.metrics import compute_metrics

env = ctx()
device = env["device"]
MODEL = PARAMS["TEACHER_MODEL"]
tokenizer = AutoTokenizer.from_pretrained(MODEL)

ckpts = sorted(glob.glob(PARAMS["CHECKPOINT_GLOB"]))
print("Checkpoints found:", ckpts)

rows = []
for ckpt in ckpts:
    model = TransformerClassifier(MODEL, num_classes=3).to(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob.get("model_state_dict", blob))
    model.eval()

    for kind in ("rand", "src", "tmp", "st"):
        f = env["splits_dir"] / f"test_{kind}.parquet"
        if not f.exists():
            continue
        test_df = pd.read_parquet(f)
        test_df, _ = map_labels(test_df)
        loader = DataLoader(TransformerDataset(
            test_df["title"] + " " + test_df["text"], test_df["label_id"],
            tokenizer, max_len=128), batch_size=32)
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for b in loader:
                p = torch.softmax(model(b["input_ids"].to(device),
                                        b["attention_mask"].to(device)).float(), 1).cpu().numpy()
                probs.append(p); preds.extend(p.argmax(1).tolist()); trues.extend(b["labels"].tolist())
        m = compute_metrics(trues, preds, y_prob=np.concatenate(probs))
        rows.append({"checkpoint": ckpt.split("/")[-1], "split": kind,
                     "macro_f1": round(m["macro_f1"], 4),
                     "balanced_accuracy": round(m["balanced_accuracy"], 4),
                     "mcc": round(m["mcc"], 4)})

table = pd.DataFrame(rows)
table.to_csv(env["out_dir"] / "hard_generalization.csv", index=False)
print(table.to_string())

# %% [markdown]
# ## 2. Statistics: grouped bootstrap + Δ Macro-F1 (from saved prediction npz)

# %%
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.statistical import paired_model_comparison
from sklearn.metrics import f1_score

npzs = sorted(glob.glob("/kaggle/input/*/**/*_predictions.npz", recursive=True)) + \
       sorted(glob.glob(str(env["out_dir"] / "**/*_predictions.npz"), recursive=True))
print("Prediction files:", npzs)

if npzs:
    a = np.load(npzs[0])
    ci = bootstrap_ci(lambda t, p: f1_score(t, p, average="macro"),
                      a["y_true"], a["y_pred"], groups=a.get("groups"))
    json.dump(ci, open(env["out_dir"] / "bootstrap_ci.json", "w"), indent=2)
    print("Grouped bootstrap:", ci)

if len(npzs) >= 2:
    b = np.load(npzs[1])
    comp = paired_model_comparison(None, a["y_true"], a["y_pred"], b["y_pred"],
                                   groups=a.get("groups"))
    json.dump(comp, open(env["out_dir"] / "model_comparison.json", "w"), indent=2)
    print("Comparison:", comp)

# %% [markdown]
# ## 3. Calibration (temperature fitted on validation only)

# %%
best_ckpt = next((c for c in ckpts if "_src_" in c), ckpts[0] if ckpts else None)
if best_ckpt:
    from src.pipelines import calibrate_checkpoint
    summary = calibrate_checkpoint(best_ckpt, MODEL, split_kind="src")
    print(json.dumps(summary, indent=2))

# %% [markdown]
# ## 4. Integrated Gradients diagnostic (50 correct / 50 incorrect)

# %%
if best_ckpt:
    from src.evaluation.explainability import integrated_gradients_attribution

    model = TransformerClassifier(MODEL, num_classes=3).to(device)
    blob = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob.get("model_state_dict", blob)); model.eval()

    test_df = pd.read_parquet(env["splits_dir"] / "test_src.parquet")
    test_df, _ = map_labels(test_df)
    sample = test_df.sample(n=min(100, len(test_df)), random_state=42)

    records = []
    for i, row in sample.iterrows():
        text = str(row["title"]) + " " + str(row["text"])
        try:
            res = integrated_gradients_attribution(model, tokenizer, text, device, max_length=128)
        except Exception as e:
            print("IG skipped:", e); break
        mass = sum(t["attribution"] for t in res["tokens"]) or 1e-9
        buckets = {}
        for t in res["tokens"]:
            buckets[t["bucket"]] = buckets.get(t["bucket"], 0) + t["attribution"] / mass
        correct = int(res["predicted_class"] == row["label_id"])
        records.append({"row": int(i), "correct": correct, **buckets})
    if records:
        ig_df = pd.DataFrame(records).fillna(0)
        ig_df.groupby("correct").mean(numeric_only=True).to_csv(
            env["out_dir"] / "ig_bucket_summary.csv")
        print(ig_df.groupby("correct").mean(numeric_only=True).head())
        print("\nDiagnostic only — NOT causal proof.")
