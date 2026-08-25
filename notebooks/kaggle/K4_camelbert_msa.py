# %% [markdown]
# # K4 — CAMeLBERT-MSA (Kaggle, GPU)
#
# Same protocol as K3 with the CAMeLBERT checkpoint; compare against AraBERT
# only within identical SPLIT/MAX_LEN/SEED via Δ-Macro-F1 CIs in K5.

# %%
!if [ -d /kaggle/working/NewsLies/.git ]; then git -C /kaggle/working/NewsLies pull --ff-only; else git clone --depth 1 https://github.com/Assem-ElQersh/NewsLies.git /kaggle/working/NewsLies; fi
%cd /kaggle/working/NewsLies

# %%
import sys
sys.path.insert(0, "/kaggle/working/NewsLies")

PARAMS = {
    "SPLIT": "rand",
    "MAX_LEN": 128,
    "SEED": 42,
    "MODEL": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
}

# %%
from src.pipelines import train_transformer
print(train_transformer(model_name=PARAMS["MODEL"], split_kind=PARAMS["SPLIT"],
                        max_length=PARAMS["MAX_LEN"], seed=PARAMS["SEED"]))
