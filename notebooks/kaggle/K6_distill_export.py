# %% [markdown]
# # K6 — Distillation + Deployment Export (Kaggle, GPU)
#
# Input: `afnd-splits-v2` + the FROZEN teacher checkpoint.
# Output: student metrics (random AND source-disjoint) + `docs/` bundle
# (model.onnx + tokenizer + calibration.json) ready for GitHub Pages.

# %%
!if [ -d /kaggle/working/NewsLies/.git ]; then git -C /kaggle/working/NewsLies pull --ff-only; else git clone --depth 1 https://github.com/Assem-ElQersh/NewsLies.git /kaggle/working/NewsLies; fi
%cd /kaggle/working/NewsLies

# %%
!pip install -q arabert pyarabic onnx onnxruntime

# %%
import sys
sys.path.insert(0, "/kaggle/working/NewsLies")

PARAMS = {
    "TEACHER_CKPT": "/kaggle/input/newslies-checkpoints/arabert_v02_src_len128_seed42.pt",
    "TEACHER_MODEL": "aubmindlab/bert-base-arabertv02",
    "ALPHA": 0.5,
    "T": 4.0,
    "EPOCHS": 5,
}

# %%
from src.pipelines import ctx, distill_bigru_student

results = distill_bigru_student(
    PARAMS["TEACHER_CKPT"], PARAMS["TEACHER_MODEL"],
    alpha=PARAMS["ALPHA"], T=PARAMS["T"], epochs=PARAMS["EPOCHS"])
print(results)

# %% [markdown]
# ## Acceptance gate
# The student's rand→src drop must not be materially worse than the teacher's.
# If it is, reject the student regardless of size savings and revisit
# alpha/T or student capacity — do NOT ship it.

# %%
drop = results["rand"]["test_macro_f1"] - results["src"]["test_macro_f1"]
print(f"Student random->disjoint drop: {drop:+.4f}")
print("Compare with the teacher drop from K5 hard_generalization.csv before accepting.")

# %% [markdown]
# ## Export deployment bundle to /kaggle/working/docs_bundle

# %%
import os, json, shutil
from src.pipelines import ctx

env = ctx()
bundle = env["out_dir"] / "docs_bundle"
os.makedirs(bundle, exist_ok=True)

from src.models.bigru_attention import BiGRUAttention
from src.data.dataset import build_vocab
import pandas as pd
import torch

train_df = pd.read_parquet(env["splits_dir"] / "train_rand.parquet")
vocab = build_vocab(train_df["title"] + " " + train_df["text"], vocab_size=30000)
student = BiGRUAttention(vocab_size=len(vocab), num_classes=3)
blob = torch.load(env["out_dir"] / "student_bigru_src.pt", map_location="cpu", weights_only=False)
student.load_state_dict(blob.get("model_state_dict", blob))
student.eval()

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = student

    def forward(self, input_ids):
        return self.net(input_ids)

torch.onnx.export(Wrapper(), (torch.zeros((1, 128), dtype=torch.long),),
                  str(bundle / "model.onnx"),
                  input_names=["input_ids"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "batch"}}, opset_version=14)

calib = env["out_dir"] / "calibration" / "calibration.json"
if calib.exists():
    shutil.copy(calib, bundle / "calibration.json")
else:
    json.dump({"temperature": 1.0,
               "note": "fit via K5 calibration step on validation data"}, open(bundle / "calibration.json", "w"))

print("Bundle contents:", os.listdir(bundle))
print("""
NEXT: download docs_bundle -> replace repo docs/ model artifacts ->
run `python -m src.inference.benchmark --model-dir docs` locally -> update README metrics.
""")
