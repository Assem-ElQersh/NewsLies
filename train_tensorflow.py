"""
Arabic Fake News Detection — TensorFlow/Keras LSTM
----------------------------------------------------
Runs on CPU (TF 2.14 requires CUDA 11.x; use train_pytorch.py for GPU).

Usage:
    python train.py [--data-dir data/AFND]

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

import pandas as pd
import numpy as np
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
RANDOM_STATE  = 42
OUTPUT_DIR    = "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    print(msg, flush=True)


def ensure_nltk_data():
    try:
        stopwords.words("arabic")
    except LookupError:
        log("Downloading NLTK stopwords...")
        nltk.download("stopwords")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    out = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    log(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: str):
    ensure_nltk_data()
    ensure_output_dir()

    # Device info — TF 2.14 needs CUDA 11.x; RTX 3050 with CUDA 13 won't be used
    gpus = tf.config.list_physical_devices("GPU")
    device_info = f"GPU ({gpus[0].name})" if gpus else "CPU  (TF 2.14 requires CUDA 11.x — use train_pytorch.py for GPU)"
    log(f"\n{'='*60}")
    log(f"  Arabic Fake News Detector — TensorFlow LSTM")
    log(f"  Device  : {device_info}")
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
    log(f"Classes: {list(label_encoder.classes_)}", )

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log(f"Split   : {len(X_train):,} train  |  {len(X_test):,} test", )

    # 6. Tokenize & pad
    log("Tokenizing and padding sequences...", )
    tokenizer = Tokenizer(num_words=MAX_VOCAB)
    tokenizer.fit_on_texts(X_train)
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_SEQ_LEN, padding="post")
    X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_SEQ_LEN, padding="post")
    log(f"  Vocab size: {len(tokenizer.word_index):,}  |  Sequence length: {MAX_SEQ_LEN}", )

    # 7. Build model
    num_classes = len(label_encoder.classes_)
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LEN),
        SpatialDropout1D(DROPOUT),
        LSTM(LSTM_UNITS, return_sequences=True),
        LSTM(LSTM_UNITS),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        metrics=["accuracy"],
    )
    log("")
    model.summary()
    log("")

    # 8. Train
    log("Training...", )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
    )
    model.fit(
        X_train_pad, y_train,
        epochs=EPOCHS,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stop],
    )

    # 9. Evaluate
    log("\nEvaluating...", )
    y_pred = np.argmax(model.predict(X_test_pad), axis=1)
    log(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}", )
    log(classification_report(y_test, y_pred, target_names=label_encoder.classes_), )

    # 10. Confusion matrix
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)

    # 11. Save model
    model_path = os.path.join(OUTPUT_DIR, "newslies_lstm.keras")
    model.save(model_path)
    log(f"Model saved to {model_path}", )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic Fake News Detector — TensorFlow LSTM (CPU)")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("AFND_DATA_DIR", "data/AFND"),
        help="Path to the AFND dataset root (contains sources.json + Dataset/). Default: data/AFND",
    )
    args = parser.parse_args()
    main(args.data_dir)
