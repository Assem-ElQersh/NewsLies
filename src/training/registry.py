import hashlib
import json
import os
import time
from datetime import datetime, timezone

REGISTRY_PATH = os.path.join("experiments", "registry.json")

SCHEMA_FIELDS = [
    "experiment_id", "model", "model_version", "dataset", "dataset_fingerprint",
    "cleaning_version", "split_type", "split_seed", "training_seed",
    "tokenizer", "max_length", "train_size", "validation_size", "test_size",
    "loss", "optimizer", "learning_rate", "scheduler", "epochs", "best_epoch",
    "metrics", "checkpoint_hash", "git_commit", "runtime_seconds", "gpu",
    "split_fingerprint", "configuration_hash",
]


def git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def configuration_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def checkpoint_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def log_experiment(entry: dict, registry_path: str = REGISTRY_PATH) -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: entry.get(k) for k in SCHEMA_FIELDS},
    }
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    if os.path.exists(registry_path):
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"version": 2, "experiments": {}}
    exp_id = record.get("experiment_id")
    if not exp_id:
        prefix = datetime.now(timezone.utc).strftime("E%Y%m%d%H%M%S")
        existing = [k for k in registry["experiments"] if k.startswith(prefix)]
        exp_id = f"{prefix}_{len(existing):02d}"
        record["experiment_id"] = exp_id
    registry["experiments"][exp_id] = record
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"Registry: logged experiment {exp_id}")
    return record


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.seconds = time.time() - self.start
