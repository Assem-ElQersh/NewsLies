# %% [markdown]
# # 03 — Source Leakage Investigation (Phase B)
#
# B1: text → source publisher-identifiability probe (word+char TF-IDF → LinearSVC).
# B2: article length vs label.
#
# Interpretation contract: a high probe score means article text contains strong
# publisher-identifying information — a shortcut OPPORTUNITY. It does not by
# itself prove what the credibility classifier learned; that requires the
# controlled split comparison in 07.

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

from src.evaluation.analysis import test_text_to_source_leakage, check_length_shortcut

OUT = Path("experiments/leakage")
OUT.mkdir(parents=True, exist_ok=True)

train_rand = pd.read_parquet("data/splits/train_rand.parquet")
test_rand = pd.read_parquet("data/splits/test_rand.parquet")

# %% [markdown]
# ## B1 — Publisher-identifiability probe

# %%
probe = test_text_to_source_leakage(
    train_rand, test_rand,
    max_train_articles=150000,
)
print(json.dumps(probe["confusion_top_pairs"][:5], ensure_ascii=False, indent=2))

with open(OUT / "text_to_source_probe.json", "w") as f:
    json.dump(probe, f, indent=2, default=str)

# %% [markdown]
# ## B2 — Length shortcut

# %%
length_stats = check_length_shortcut(pd.concat([train_rand, test_rand]))
with open(OUT / "length_shortcut.json", "w") as f:
    json.dump(length_stats, f, indent=2)

# %%
print("""
Interpretation (do not overstate):
- The probe quantifies how identifiable publishers are from text alone.
- The causal question — 'does the credibility model rely on these cues?' — is
  answered by training the SAME configuration on random vs source-disjoint
  splits and comparing the drop (see notebooks 07 + 11).
""")
