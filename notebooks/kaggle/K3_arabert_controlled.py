# %% [markdown]
# # K3 — AraBERTv0.2 Controlled Grid (Kaggle, GPU)
#
# The controlled leakage experiment. ONE variable changes per run — everything
# else is fixed inside `src.pipelines.train_transformer`.
#
# ## How to run the grid
# Duplicate this notebook and edit ONLY the PARAMS cell:
#
# | purpose            | SPLIT | MAX_LEN | SEED |
# |--------------------|-------|---------|------|
# | mandatory pair     | rand  | 128     | 42   |
# |                    | src   | 128     | 42   |
# | context ablation   | rand/src | 256, 512 | 42 |
# | final seeds        | chosen split | chosen len | 13/42/71/101/202 |
#
# Input: `afnd-splits-v2`. Output: metrics + predictions + checkpoint.

# %%
!if [ -d /kaggle/working/NewsLies/.git ]; then git -C /kaggle/working/NewsLies pull --ff-only; else git clone --depth 1 https://github.com/Assem-ElQersh/NewsLies.git /kaggle/working/NewsLies; fi
%cd /kaggle/working/NewsLies

# %%
!pip install -q arabert pyarabic

# %%
import sys
sys.path.insert(0, "/kaggle/working/NewsLies")

PARAMS = {
    "SPLIT": "rand",          # 'rand' | 'src' | 'tmp' | 'st'
    "MAX_LEN": 128,           # 128 | 256 | 512
    "SEED": 42,
    "MODEL": "aubmindlab/bert-base-arabertv02",   # AraBERTv0.2-base (NOT v2)
    "EPOCHS": 3,
    "BATCH_SIZE": 16,
    "LR": 2e-5,
    "GRAD_ACCUM": 1,
    "LOSS": "ce",
}

# %%
from src.pipelines import ctx, train_transformer

payload = train_transformer(
    model_name=PARAMS["MODEL"],
    split_kind=PARAMS["SPLIT"],
    max_length=PARAMS["MAX_LEN"],
    seed=PARAMS["SEED"],
    epochs=PARAMS["EPOCHS"],
    batch_size=PARAMS["BATCH_SIZE"],
    lr=PARAMS["LR"],
    grad_accum=PARAMS["GRAD_ACCUM"],
    loss_type=PARAMS["LOSS"],
)
print(payload)

# %% [markdown]
# Interpretation contract:
# - Compare ONLY cells differing in the declared variable.
# - The rand→src Macro-F1 drop at fixed (len=128, seed=42) is THE primary
#   leakage evidence. Historical numbers used different budgets and are
#   superseded by this controlled pair.
