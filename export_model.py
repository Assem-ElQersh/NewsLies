"""Export a trained NewsLies model to the deployment bundle under docs/.

Examples:
    python export_model.py --checkpoint experiments/student_bigru.pt --arch bigru \
        --vocab-size 30000 --out docs --calibration experiments/calibration/calibration.json
    python export_model.py --checkpoint experiments/arabert_v02_rand_len128_seed42.pt \
        --arch transformer --model-name aubmindlab/bert-base-arabertv02 --out docs
"""
import argparse
import json
import os
import shutil
import sys

import torch


def export_bigru(checkpoint, vocab_size, out_dir, max_len=128):
    from src.models.bigru_attention import BiGRUAttention
    from src.data.dataset import build_vocab
    import pandas as pd

    model = BiGRUAttention(vocab_size=vocab_size, num_classes=3)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob.get("model_state_dict", blob)
    model.load_state_dict(state)
    model.eval()

    train_df = pd.read_parquet("data/splits/train_rand.parquet")
    vocab = build_vocab(train_df["title"] + " " + train_df["text"],
                        vocab_size=vocab_size)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = model
            self.vocab = vocab
            self.max_len = max_len

        def forward(self, input_ids):
            return self.net(input_ids)

    dummy = torch.zeros((1, max_len), dtype=torch.long)
    path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        Wrapper(), (dummy,), path,
        input_names=["input_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}},
        opset_version=14,
    )
    return path


def export_transformer(checkpoint, model_name, out_dir, max_len=128):
    from src.models.transformer_clf import TransformerClassifier
    from transformers import AutoTokenizer

    model = TransformerClassifier(model_name, num_classes=3)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(blob.get("model_state_dict", blob))
    model.eval()

    tok = AutoTokenizer.from_pretrained(model_name)
    dummy = tok("نص تجريبي", truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt")

    path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), path,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=14,
    )
    tok.save_pretrained(out_dir)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", choices=["bigru", "transformer"], required=True)
    parser.add_argument("--model-name", default=None,
                        help="HF checkpoint for --arch transformer")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--out", default="docs")
    parser.add_argument("--calibration", default=None,
                        help="calibration.json containing fitted temperature")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.arch == "bigru":
        path = export_bigru(args.checkpoint, args.vocab_size, args.out, args.max_len)
    else:
        if not args.model_name:
            raise SystemExit("--model-name is required for --arch transformer")
        path = export_transformer(args.checkpoint, args.model_name, args.out, args.max_len)

    size_mb = os.path.getsize(path) / 1e6
    print(f"Exported {path} ({size_mb:.2f} MB)")

    calib_src = args.calibration or os.path.join("experiments", "calibration",
                                                 "calibration.json")
    if os.path.exists(calib_src):
        shutil.copy(calib_src, os.path.join(args.out, "calibration.json"))
        print(f"Copied calibration from {calib_src}")
    else:
        with open(os.path.join(args.out, "calibration.json"), "w") as f:
            json.dump({"temperature": 1.0,
                       "note": "default; fit on validation data via 13_calibration"}, f)
        print("No calibration found; wrote temperature=1.0 placeholder")


if __name__ == "__main__":
    main()
