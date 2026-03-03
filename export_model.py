"""
Export the trained PyTorch model to ONNX for use in the browser (GitHub Pages).

Run once after training:
    python export_model.py

Outputs written to docs/:
    docs/model.onnx      — ONNX model (~4 MB, runs in browser via onnxruntime-web)
    docs/vocab.json      — word → token-index mapping
    docs/stopwords.json  — Arabic stopwords list
"""

import os
import json
import sys

import torch
import torch.nn as nn
import nltk
from nltk.corpus import stopwords

# Ensure the project root is on the path so we can import from train_pytorch
sys.path.insert(0, os.path.dirname(__file__))
from train_pytorch import LSTMClassifier

CHECKPOINT  = "outputs/newslies_lstm.pt"
OUTPUT_DIR  = "docs"
ONNX_PATH   = os.path.join(OUTPUT_DIR, "model.onnx")
VOCAB_PATH  = os.path.join(OUTPUT_DIR, "vocab.json")
SW_PATH     = os.path.join(OUTPUT_DIR, "stopwords.json")


def load_model(checkpoint_path: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    cfg    = ckpt["config"]
    model  = LSTMClassifier(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden=cfg["hidden"],
        n_classes=cfg["n_classes"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Classes       : {ckpt['label_classes']}")
    print(f"  Vocab size    : {cfg['vocab_size']:,}")
    print(f"  Max seq len   : {cfg['max_seq_len']}")
    return model, ckpt


def export_onnx(model: nn.Module, seq_len: int, path: str):
    print(f"Exporting ONNX → {path}")
    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "logits":    {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"  Saved ({size_mb:.1f} MB)")


def export_vocab(ckpt: dict, path: str):
    print(f"Exporting vocab → {path}")
    vocab = {
        "word2idx":     ckpt["tokenizer_word2idx"],
        "label_classes": ckpt["label_classes"],
        "max_seq_len":  ckpt["config"]["max_seq_len"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved ({size_kb:.0f} KB, {len(vocab['word2idx']):,} tokens)")


def export_stopwords(path: str):
    print(f"Exporting Arabic stopwords → {path}")
    try:
        sw = list(stopwords.words("arabic"))
    except LookupError:
        nltk.download("stopwords")
        sw = list(stopwords.words("arabic"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sw, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  Saved ({len(sw)} stopwords)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        print("Run train_pytorch.py first to generate the model.")
        sys.exit(1)

    model, ckpt = load_model(CHECKPOINT)
    export_onnx(model, ckpt["config"]["max_seq_len"], ONNX_PATH)
    export_vocab(ckpt, VOCAB_PATH)
    export_stopwords(SW_PATH)

    print("\nDone! Files written to docs/")
    print("Next steps:")
    print("  1. git add docs/ && git push")
    print("  2. GitHub → Settings → Pages → Source: main branch, /docs folder")
    print("  3. Visit https://<your-username>.github.io/NewsLies/")


if __name__ == "__main__":
    main()
