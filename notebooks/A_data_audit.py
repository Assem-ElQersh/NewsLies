# %%
import os
import sys

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_afnd
from src.data.cleaner import clean_afnd
from src.data.splits import random_split, source_disjoint_split, rolling_temporal_evaluation
from src.evaluation.analysis import check_label_generation, test_text_to_source_leakage, check_length_shortcut, check_date_reliability

# %%
# Clean up legacy TF script
tf_script = os.path.join(os.path.dirname(__file__), '..', 'train_tensorflow.py')
if os.path.exists(tf_script):
    os.remove(tf_script)
    print("Deleted legacy train_tensorflow.py")

# %%
print("Loading AFND Dataset...")
df = load_afnd()

# %%
print("\nPhase A1: Source-Label Mechanism (Sanity Check)")
check_label_generation(df)

# %%
print("\nPhase A0: Data Audit & Cleaning")
df_clean = clean_afnd(df)

# %%
print("\nPhase A3: Date Reliability & Length Shortcut")
check_date_reliability(df_clean)
check_length_shortcut(df_clean)

# %%
print("\nPhase A4: Split Generation")
train_rand, val_rand, test_rand = random_split(df_clean)
train_src, val_src, test_src = source_disjoint_split(df_clean)
train_temp, val_temp, test_temp = rolling_temporal_evaluation(df_clean)

# %%
print("\nPhase A2: Text to Source Leakage")
# We test leakage on the random split to see if text predicts source.
test_text_to_source_leakage(train_rand, test_rand)

# %%
print("\nSaving splits to data/splits/...")
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data', 'splits'), exist_ok=True)
train_rand.to_parquet('data/splits/train_rand.parquet')
val_rand.to_parquet('data/splits/val_rand.parquet')
test_rand.to_parquet('data/splits/test_rand.parquet')

train_src.to_parquet('data/splits/train_src.parquet')
val_src.to_parquet('data/splits/val_src.parquet')
test_src.to_parquet('data/splits/test_src.parquet')

print("Phase A Complete.")
