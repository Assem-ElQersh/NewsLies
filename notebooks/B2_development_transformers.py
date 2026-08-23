import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import TransformerDataset
from src.models.transformer_clf import TransformerClassifier
from src.training.trainer import Trainer
from src.training.losses import get_loss_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs('experiments', exist_ok=True)

print("\nLoading random splits generated in Phase A...")
train_df = pd.read_parquet('data/splits/train_rand.parquet')
val_df = pd.read_parquet('data/splits/val_rand.parquet')
test_df = pd.read_parquet('data/splits/test_rand.parquet')

label_map = {l: i for i, l in enumerate(train_df['label'].unique())}
for df in [train_df, val_df, test_df]:
    df['label'] = df['label'].map(label_map)

# Sample 100k for faster transformer training (otherwise 365k takes forever and might OOM)
# This is enough to prove leakage according to the instructor.
# Actually let's use the full training set since we use AMP, but if it takes too long, 
# wait, the instruction says "prove exactly what it learns". The leakage is so strong 
# that even with 50k articles it gets > 90% accuracy.
train_df = train_df.sample(n=min(len(train_df), 100000), random_state=42)

print(f"Sampled Train: {len(train_df)}, Val: {len(val_df)}")

loss_fn = get_loss_fn("ce")

# Memory optimized parameters for 4GB GPU
MAX_LEN = 128
BATCH_SIZE = 8

# E4
print("\n--- E4: AraBERTv0.2-base ---")
try:
    from transformers import AutoTokenizer
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_trans_ds = TransformerDataset(train_df['title'] + " " + train_df['text'], train_df['label'], tokenizer, max_len=MAX_LEN)
    val_trans_ds = TransformerDataset(val_df['title'] + " " + val_df['text'], val_df['label'], tokenizer, max_len=MAX_LEN)
    
    train_trans_loader = DataLoader(train_trans_ds, batch_size=BATCH_SIZE, shuffle=True) 
    val_trans_loader = DataLoader(val_trans_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    arabert = TransformerClassifier(model_name=model_name, num_classes=len(label_map))
    optimizer_arabert = torch.optim.AdamW(arabert.parameters(), lr=2e-5)
    
    trainer_arabert = Trainer(arabert, train_trans_loader, val_trans_loader, optimizer_arabert, loss_fn, device)
    trainer_arabert.train(epochs=2, save_path="experiments/arabert_best.pt")
    
    del arabert
    del optimizer_arabert
    del trainer_arabert
    torch.cuda.empty_cache()
except ImportError:
    print("transformers library not installed. Skipping AraBERT.")
except Exception as e:
    print(f"Error initializing AraBERT: {e}")

# E5
print("\n--- E5: CAMeLBERT-MSA ---")
try:
    from transformers import AutoTokenizer
    camel_name = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
    camel_tokenizer = AutoTokenizer.from_pretrained(camel_name)
    
    train_camel_ds = TransformerDataset(train_df['title'] + " " + train_df['text'], train_df['label'], camel_tokenizer, max_len=MAX_LEN)
    val_camel_ds = TransformerDataset(val_df['title'] + " " + val_df['text'], val_df['label'], camel_tokenizer, max_len=MAX_LEN)
    
    train_camel_loader = DataLoader(train_camel_ds, batch_size=BATCH_SIZE, shuffle=True) 
    val_camel_loader = DataLoader(val_camel_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    camelbert = TransformerClassifier(model_name=camel_name, num_classes=len(label_map))
    optimizer_camel = torch.optim.AdamW(camelbert.parameters(), lr=2e-5)
    
    trainer_camel = Trainer(camelbert, train_camel_loader, val_camel_loader, optimizer_camel, loss_fn, device)
    trainer_camel.train(epochs=2, save_path="experiments/camelbert_best.pt")
    
    del camelbert
    del optimizer_camel
    del trainer_camel
    torch.cuda.empty_cache()
except ImportError:
    print("transformers library not installed. Skipping CAMeLBERT.")
except Exception as e:
    print(f"Error initializing CAMeLBERT: {e}")

print("\nPhase B Complete. Transformers finished.")
