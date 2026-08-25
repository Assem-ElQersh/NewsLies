import json
import os
import time

import numpy as np


class CredibilityPipeline:
    """ONNX-browser-parity inference pipeline for the deployed student model.

    Expects docs/model.onnx, tokenizer files and optional calibration.json with
    a temperature fitted on validation data only.
    """

    def __init__(self, model_dir: str = "docs", max_length: int = 128,
                 providers=None):
        import onnxruntime as ort

        self.model_dir = model_dir
        self.max_length = max_length
        so = ort.SessionOptions()
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            sess_options=so,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

        calib_path = os.path.join(model_dir, "calibration.json")
        self.temperature = 1.0
        if os.path.exists(calib_path):
            with open(calib_path, encoding="utf-8") as f:
                self.temperature = float(json.load(f)["temperature"])

    @staticmethod
    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def predict(self, text: str) -> dict:
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length)
        ids = list(enc["input_ids"])[:self.max_length]
        mask = [1] * len(ids)
        pad = self.max_length - len(ids)
        ids += [0] * pad
        mask += [0] * pad

        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([ids], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array([mask], dtype=np.int64)
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros((1, self.max_length), dtype=np.int64)

        logits = self.session.run(None, feed)[0][0]
        probs = self.softmax(logits / self.temperature)

        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = float(np.log(len(probs)))
        return {
            "probs": probs.tolist(),
            "predicted_class": int(probs.argmax()),
            "confidence": float(probs.max()),
            "uncertainty": entropy / max_entropy,
            "temperature": self.temperature,
        }

    def benchmark(self, texts, warmup: int = 5) -> dict:
        for t in texts[:warmup]:
            self.predict(t)
        latencies = []
        for t in texts:
            start = time.perf_counter()
            self.predict(t)
            latencies.append((time.perf_counter() - start) * 1000)
        latencies = np.array(latencies)
        size_mb = os.path.getsize(os.path.join(self.model_dir, "model.onnx")) / 1e6
        data_path = os.path.join(self.model_dir, "model.onnx.data")
        if os.path.exists(data_path):
            size_mb += os.path.getsize(data_path) / 1e6
        return {
            "model_size_mb": round(size_mb, 2),
            "latency_ms_mean": round(float(latencies.mean()), 2),
            "latency_ms_p50": round(float(np.percentile(latencies, 50)), 2),
            "latency_ms_p95": round(float(np.percentile(latencies, 95)), 2),
            "n_texts": len(texts),
            "providers": self.session.get_providers(),
        }
