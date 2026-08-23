import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import TransformerDataset, ClassicalDataset
from src.models.transformer_clf import TransformerClassifier
from src.models.bigru_attention import BiGRUAttention
from src.training.trainer import Trainer
from src.training.losses import get_loss_fn
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs('experiments/taxonomy', exist_ok=True)

# 1. Load Source-Disjoint Splits
print("\nLoading source-disjoint splits generated in Phase A...")
train_df = pd.read_parquet('data/splits/train_src.parquet')
val_df = pd.read_parquet('data/splits/val_src.parquet')
test_df = pd.read_parquet('data/splits/test_src.parquet')

# Build Label Map based on training labels to ensure consistency
label_map = {l: i for i, l in enumerate(train_df['label'].unique())}
reverse_label_map = {i: l for l, i in label_map.items()}

for df in [train_df, val_df, test_df]:
    df['label_id'] = df['label'].map(label_map)

print(f"Disjoint Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

loss_fn = get_loss_fn("ce")
MAX_LEN = 128
BATCH_SIZE = 8

# 2. Phase C: Train AraBERT on Source-Disjoint
print("\n--- E6: AraBERTv0.2-base (Source-Disjoint) ---")
from transformers import AutoTokenizer
model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample to 50k to save time on local GPU (sufficient to show performance collapse)
train_sample = train_df.sample(n=min(len(train_df), 50000), random_state=42)

train_trans_ds = TransformerDataset(train_sample['title'] + " " + train_sample['text'], train_sample['label_id'], tokenizer, max_len=MAX_LEN)
val_trans_ds = TransformerDataset(val_df['title'] + " " + val_df['text'], val_df['label_id'], tokenizer, max_len=MAX_LEN)
test_trans_ds = TransformerDataset(test_df['title'] + " " + test_df['text'], test_df['label_id'], tokenizer, max_len=MAX_LEN)

train_trans_loader = DataLoader(train_trans_ds, batch_size=BATCH_SIZE, shuffle=True) 
val_trans_loader = DataLoader(val_trans_ds, batch_size=BATCH_SIZE, shuffle=False)
test_trans_loader = DataLoader(test_trans_ds, batch_size=BATCH_SIZE, shuffle=False)

arabert = TransformerClassifier(model_name=model_name, num_classes=len(label_map))
optimizer_arabert = torch.optim.AdamW(arabert.parameters(), lr=2e-5)

trainer_arabert = Trainer(arabert, train_trans_loader, val_trans_loader, optimizer_arabert, loss_fn, device)
print("Training AraBERT on disjoint split...")
trainer_arabert.train(epochs=2, save_path="experiments/arabert_disjoint.pt")

print("Evaluating AraBERT on disjoint TEST split...")
test_metrics = trainer_arabert.evaluate(test_trans_loader)
print(f"AraBERT Disjoint Test Macro-F1: {test_metrics['macro_f1']:.4f}")

# 3. Generate Predictions for Error Taxonomy (Phase F)
print("\n--- Phase F: Error Taxonomy Extraction ---")
arabert.eval()
all_preds = []
with torch.no_grad():
    for batch in test_trans_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = arabert(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)

test_df['pred_id'] = all_preds
test_df['pred'] = test_df['pred_id'].map(reverse_label_map)

# Extract errors
errors_df = test_df[test_df['label'] != test_df['pred']]
print(f"Total Errors Found: {len(errors_df)} out of {len(test_df)}")

# Sample 500 errors for manual taxonomy
sampled_errors = errors_df.sample(n=min(len(errors_df), 500), random_state=42)
sampled_errors.to_csv('experiments/taxonomy/arabert_errors_sample.csv', index=False)
print("Saved 500 sampled errors to experiments/taxonomy/arabert_errors_sample.csv")

print("\nPhase C & F Execution Complete.")
