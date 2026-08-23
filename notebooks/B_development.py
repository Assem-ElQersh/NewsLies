# %%
import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import ClassicalDataset, TransformerDataset
from src.models.lstm import LSTMClassifier
from src.models.bigru_attention import BiGRUAttention
from src.models.transformer_clf import TransformerClassifier
from src.training.trainer import Trainer
from src.training.losses import get_loss_fn
from src.evaluation.metrics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs('experiments', exist_ok=True)

# %%
print("\nLoading random splits generated in Phase A...")
try:
    train_df = pd.read_parquet('data/splits/train_rand.parquet')
    val_df = pd.read_parquet('data/splits/val_rand.parquet')
    test_df = pd.read_parquet('data/splits/test_rand.parquet')
except FileNotFoundError:
    print("Run notebook A_data_audit.py first to generate the random splits.")
    sys.exit(1)

label_map = {l: i for i, l in enumerate(train_df['label'].unique())}
for df in [train_df, val_df, test_df]:
    df['label'] = df['label'].map(label_map)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# %% 
print("\n--- E0: Majority Classifier ---")
majority_class = train_df['label'].mode()[0]
preds = [majority_class] * len(val_df)
metrics = compute_metrics(val_df['label'], preds)
print(f"Majority Classifier Val Macro-F1: {metrics['macro_f1']:.4f}")

# %% 
print("\n--- E1: TF-IDF + LinearSVM ---")
# Using a small subset to prevent memory crash during vec fit if RAM is low, 
# but SVM can handle sparse matrices well.
vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train = vec.fit_transform(train_df['title'] + " " + train_df['text'])
y_train = train_df['label']
X_val = vec.transform(val_df['title'] + " " + val_df['text'])
y_val = val_df['label']

svm = LinearSVC(max_iter=1000, class_weight='balanced')
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_val)
svm_metrics = compute_metrics(y_val, svm_preds)
print(f"TF-IDF + SVM Val Macro-F1: {svm_metrics['macro_f1']:.4f}")

# %%
print("\nBuilding vocabulary for classical neural models...")
from collections import Counter
# To save memory/time, sample 100k articles for vocab building
sample_text = train_df.sample(min(100000, len(train_df)))
words = " ".join(sample_text['title'] + " " + sample_text['text']).split()
most_common = Counter(words).most_common(29998) # Leave room for PAD and UNK
vocab = {word: i+2 for i, (word, _) in enumerate(most_common)}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

train_class_ds = ClassicalDataset(train_df['title'] + " " + train_df['text'], train_df['label'], vocab, max_len=256)
val_class_ds = ClassicalDataset(val_df['title'] + " " + val_df['text'], val_df['label'], vocab, max_len=256)

train_class_loader = DataLoader(train_class_ds, batch_size=64, shuffle=True)
val_class_loader = DataLoader(val_class_ds, batch_size=64, shuffle=False)
loss_fn = get_loss_fn("ce")

# %%
print("\n--- E2: LSTM Baseline ---")
lstm_model = LSTMClassifier(vocab_size=len(vocab), num_classes=len(label_map))
optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
trainer_lstm = Trainer(lstm_model, train_class_loader, val_class_loader, optimizer_lstm, loss_fn, device)
trainer_lstm.train(epochs=5, save_path="experiments/lstm_best.pt")

# %%
print("\n--- E3: BiGRU + Self-Attention ---")
bigru_model = BiGRUAttention(vocab_size=len(vocab), num_classes=len(label_map))
optimizer_bigru = torch.optim.Adam(bigru_model.parameters(), lr=1e-3)
trainer_bigru = Trainer(bigru_model, train_class_loader, val_class_loader, optimizer_bigru, loss_fn, device)
trainer_bigru.train(epochs=5, save_path="experiments/bigru_best.pt")

# %%
print("\n--- E4: AraBERTv0.2-base ---")
try:
    from transformers import AutoTokenizer
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 512 tokens might OOM on smaller GPUs, change to 256 or lower batch size if needed
    train_trans_ds = TransformerDataset(train_df['title'] + " " + train_df['text'], train_df['label'], tokenizer, max_len=512)
    val_trans_ds = TransformerDataset(val_df['title'] + " " + val_df['text'], val_df['label'], tokenizer, max_len=512)
    
    train_trans_loader = DataLoader(train_trans_ds, batch_size=16, shuffle=True) 
    val_trans_loader = DataLoader(val_trans_ds, batch_size=16, shuffle=False)
    
    arabert = TransformerClassifier(model_name=model_name, num_classes=len(label_map))
    optimizer_arabert = torch.optim.AdamW(arabert.parameters(), lr=2e-5)
    
    trainer_arabert = Trainer(arabert, train_trans_loader, val_trans_loader, optimizer_arabert, loss_fn, device)
    trainer_arabert.train(epochs=4, save_path="experiments/arabert_best.pt")
except ImportError:
    print("transformers library not installed. Skipping AraBERT.")
except Exception as e:
    print(f"Error initializing AraBERT: {e}")

# %%
print("\n--- E5: CAMeLBERT-MSA ---")
try:
    from transformers import AutoTokenizer
    camel_name = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
    camel_tokenizer = AutoTokenizer.from_pretrained(camel_name)
    
    train_camel_ds = TransformerDataset(train_df['title'] + " " + train_df['text'], train_df['label'], camel_tokenizer, max_len=512)
    val_camel_ds = TransformerDataset(val_df['title'] + " " + val_df['text'], val_df['label'], camel_tokenizer, max_len=512)
    
    train_camel_loader = DataLoader(train_camel_ds, batch_size=16, shuffle=True) 
    val_camel_loader = DataLoader(val_camel_ds, batch_size=16, shuffle=False)
    
    camelbert = TransformerClassifier(model_name=camel_name, num_classes=len(label_map))
    optimizer_camel = torch.optim.AdamW(camelbert.parameters(), lr=2e-5)
    
    trainer_camel = Trainer(camelbert, train_camel_loader, val_camel_loader, optimizer_camel, loss_fn, device)
    trainer_camel.train(epochs=4, save_path="experiments/camelbert_best.pt")
except ImportError:
    print("transformers library not installed. Skipping CAMeLBERT.")
except Exception as e:
    print(f"Error initializing CAMeLBERT: {e}")

print("\nPhase B Complete. Review Macro-F1 scores to determine the best Neural baseline and best Transformer.")
