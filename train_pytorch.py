"""
Arabic Fake News Detection — PyTorch LSTM (GPU-accelerated)
------------------------------------------------------------
Automatically uses your NVIDIA GPU (RTX 3050) if available.

Usage:
    python train_pytorch.py [--data-dir data/AFND]

Dataset directory must contain:
    <data-dir>/sources.json
    <data-dir>/Dataset/<source-name>/scraped_articles.json

Download from:
    https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd
    kaggle datasets download -d murtadhayaseen/arabic-fake-news-dataset-afnd -p data --unzip
"""

import os
import sys
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_VOCAB     = 10_000
MAX_SEQ_LEN   = 128
EMBEDDING_DIM = 100
LSTM_UNITS    = 64
DROPOUT       = 0.2
BATCH_SIZE    = 64
EPOCHS        = 10
LEARNING_RATE = 1e-4
TEST_SIZE     = 0.2
VAL_SIZE      = 0.2   # fraction of train set used for validation
RANDOM_STATE  = 42
PATIENCE      = 2     # early stopping patience
OUTPUT_DIR    = "outputs"
PAD_IDX       = 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str = ""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def ensure_nltk_data():
    try:
        stopwords.words("arabic")
    except LookupError:
        log("Downloading NLTK stopwords...")
        nltk.download("stopwords")


def load_data(data_dir: str) -> pd.DataFrame:
    sources_path = os.path.join(data_dir, "sources.json")
    dataset_dir  = os.path.join(data_dir, "Dataset")

    if not os.path.exists(sources_path):
        raise FileNotFoundError(
            f"sources.json not found at: {sources_path}\n"
            "Make sure you extracted the AFND dataset to the correct location.\n"
            "Download: kaggle datasets download -d murtadhayaseen/arabic-fake-news-dataset-afnd -p data --unzip"
        )

    log(f"Loading sources from:  {sources_path}")
    with open(sources_path, "r", encoding="utf-8") as f:
        sources_data = json.load(f)
    sources_df = pd.DataFrame(list(sources_data.items()), columns=["source", "label"])
    log(f"  {len(sources_df)} sources  |  labels: {dict(sources_df['label'].value_counts())}")

    log(f"Loading articles from: {dataset_dir}")
    articles_data = []
    missing = 0
    for source in tqdm(sources_df["source"], desc="  Reading sources", unit="src", file=sys.stdout):
        articles_path = os.path.join(dataset_dir, source, "scraped_articles.json")
        if not os.path.exists(articles_path):
            missing += 1
            continue
        with open(articles_path, "r", encoding="utf-8") as f:
            source_dict = json.load(f)
        for article in source_dict.get("articles", []):
            article["source"] = source
            articles_data.append(article)

    if missing:
        log(f"  Warning: {missing} source director(ies) not found — skipped.")

    articles_df = pd.DataFrame(articles_data)
    merged = pd.merge(articles_df, sources_df, on="source", how="inner")
    log(f"  Total articles loaded: {len(merged):,}")
    return merged


def preprocess_text(series: pd.Series) -> pd.Series:
    log("Preprocessing text (lowercase → stopword removal → stemming)...")
    stop_words = set(stopwords.words("arabic"))
    stemmer    = ISRIStemmer()

    def clean(text):
        text   = str(text).lower()
        tokens = [w for w in text.split() if w not in stop_words]
        tokens = [stemmer.stem(w) for w in tokens]
        return " ".join(tokens)

    tqdm.pandas(desc="  Cleaning", unit="art", file=sys.stdout)
    return series.progress_apply(clean)


def plot_label_distribution(df: pd.DataFrame):
    counts = {
        "Credible":     len(df[df["label"] == "credible"]),
        "Not Credible": len(df[df["label"] == "not credible"]),
        "Undecided":    len(df[df["label"] == "undecided"]),
    }
    colors = ["#2ecc71", "#e67e22", "#95a5a6"]
    plt.figure(figsize=(10, 5))
    plt.bar(counts.keys(), counts.values(), color=colors)
    plt.title("Distribution of Articles", size=15)
    plt.xlabel("Article Type", size=13)
    plt.ylabel("# of Articles", size=13)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "label_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    log(f"Saved {out}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_matrix_pytorch.png")
    plt.savefig(out, dpi=150)
    plt.close()
    log(f"Saved {out}")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """Minimal word-level tokenizer (mirrors Keras Tokenizer behaviour)."""

    def __init__(self, num_words: int):
        self.num_words  = num_words
        self.word2idx: dict[str, int] = {}

    def fit(self, texts):
        counter: Counter = Counter()
        for text in tqdm(texts, desc="  Building vocab", unit="doc", file=sys.stdout):
            counter.update(str(text).split())
        # Reserve 0 for padding, assign indices by frequency
        for i, (word, _) in enumerate(counter.most_common(self.num_words - 1), start=1):
            self.word2idx[word] = i

    def encode(self, text: str) -> list[int]:
        return [self.word2idx[w] for w in str(text).split() if w in self.word2idx]

    def texts_to_sequences(self, texts) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    @property
    def vocab_size(self) -> int:
        return min(len(self.word2idx) + 1, self.num_words)


def pad_sequences(sequences: list[list[int]], maxlen: int) -> np.ndarray:
    out = np.zeros((len(sequences), maxlen), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = seq[:maxlen]
        out[i, : len(seq)] = seq
    return out


# ---------------------------------------------------------------------------
# PyTorch Dataset & Model
# ---------------------------------------------------------------------------

class NewsDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels    = torch.tensor(labels,    dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMClassifier(nn.Module):
    """
    Embedding → SpatialDropout1D → LSTM(64, return_seq) → LSTM(64) → Linear(n_classes)
    Mirrors the TensorFlow architecture in train.py.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.embedding        = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.spatial_dropout  = nn.Dropout(dropout)          # applied to embedding output
        self.lstm1            = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.lstm2            = nn.LSTM(hidden,    hidden, batch_first=True)
        self.fc               = nn.Linear(hidden, n_classes)

    def forward(self, x):
        emb = self.embedding(x)                              # (B, T, E)
        emb = self.spatial_dropout(emb)                      # (B, T, E)
        out, _ = self.lstm1(emb)                             # (B, T, H)
        _, (hn, _) = self.lstm2(out)                         # hn: (1, B, H)
        logits = self.fc(hn.squeeze(0))                      # (B, n_classes)
        return logits


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience   = patience
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    desc = "  train" if train else "  val  "

    with torch.set_grad_enabled(train):
        for seqs, labels in tqdm(loader, desc=desc, leave=False, file=sys.stdout):
            seqs   = seqs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(seqs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += len(labels)

    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: str):
    ensure_nltk_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        log(f"\n{'='*60}")
        log(f"  Arabic Fake News Detector — PyTorch LSTM")
        log(f"  Device  : GPU — {gpu_name}")
    else:
        log(f"\n{'='*60}")
        log(f"  Arabic Fake News Detector — PyTorch LSTM")
        log(f"  Device  : CPU  (no CUDA GPU found)")
    log(f"  Data dir: {data_dir}")
    log(f"{'='*60}\n")

    # 1. Load data
    df = load_data(data_dir)

    # 2. Label distribution chart
    plot_label_distribution(df)

    # 3. Preprocess
    X = preprocess_text(df["text"])
    y = df["label"]

    # 4. Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)
    log(f"Classes : {list(label_encoder.classes_)}")

    # 5. Train / test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X.values, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    log(f"Split   : {len(X_train_raw):,} train  |  {len(X_test_raw):,} test")

    # 6. Train / val split (from train portion)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )
    log(f"         {len(X_train_raw):,} train  |  {len(X_val_raw):,} val")

    # 7. Tokenize & pad
    log("\nBuilding vocabulary...")
    tokenizer = SimpleTokenizer(num_words=MAX_VOCAB)
    tokenizer.fit(X_train_raw)
    log(f"  Vocab size: {tokenizer.vocab_size:,}  |  Sequence length: {MAX_SEQ_LEN}")

    log("Padding sequences...")
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), MAX_SEQ_LEN)
    X_val_pad   = pad_sequences(tokenizer.texts_to_sequences(X_val_raw),   MAX_SEQ_LEN)
    X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test_raw),  MAX_SEQ_LEN)

    # 8. DataLoaders
    pin = device.type == "cuda"
    train_loader = DataLoader(NewsDataset(X_train_pad, y_train), batch_size=BATCH_SIZE, shuffle=True,  pin_memory=pin, num_workers=2)
    val_loader   = DataLoader(NewsDataset(X_val_pad,   y_val),   batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin, num_workers=2)
    test_loader  = DataLoader(NewsDataset(X_test_pad,  y_test),  batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin, num_workers=2)

    # 9. Build model
    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBEDDING_DIM,
        hidden=LSTM_UNITS,
        n_classes=n_classes,
        dropout=DROPOUT,
    ).to(device)

    log(f"\nModel architecture:")
    log(str(model))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable parameters: {total_params:,}\n")

    criterion    = nn.CrossEntropyLoss()
    optimizer    = Adam(model.parameters(), lr=LEARNING_RATE)
    early_stop   = EarlyStopping(patience=PATIENCE)

    # 10. Training loop
    log("Training...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        log(f"  Epoch {epoch:2d}/{EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        if early_stop.step(vl_loss, model):
            log(f"  Early stopping triggered (patience={PATIENCE}). Restoring best weights.")
            early_stop.restore(model)
            break

    # 11. Evaluate
    log("\nEvaluating on test set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in tqdm(test_loader, desc="  Testing", file=sys.stdout):
            preds = model(seqs.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_preds)
    log(f"\nTest Accuracy: {accuracy_score(all_labels, y_pred):.4f}")
    log(classification_report(all_labels, y_pred, target_names=label_encoder.classes_))

    # 12. Confusion matrix
    plot_confusion_matrix(all_labels, y_pred, label_encoder.classes_)

    # 13. Save model
    model_path = os.path.join(OUTPUT_DIR, "newslies_lstm.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_word2idx": tokenizer.word2idx,
        "label_classes": list(label_encoder.classes_),
        "config": {
            "vocab_size":   tokenizer.vocab_size,
            "embed_dim":    EMBEDDING_DIM,
            "hidden":       LSTM_UNITS,
            "n_classes":    n_classes,
            "dropout":      DROPOUT,
            "max_seq_len":  MAX_SEQ_LEN,
        },
    }, model_path)
    log(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic Fake News Detector — PyTorch LSTM (GPU)")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("AFND_DATA_DIR", "data/AFND"),
        help="Path to the AFND dataset root (contains sources.json + Dataset/). Default: data/AFND",
    )
    args = parser.parse_args()
    main(args.data_dir)
