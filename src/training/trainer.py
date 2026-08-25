import json
import os

import numpy as np
import torch
from torch.amp import autocast, GradScaler

from src.evaluation.metrics import compute_metrics
from src.training.registry import Timer, configuration_hash, git_commit, log_experiment


class Trainer:
    """Unified training loop for classical and transformer models.

    Features: mixed precision, gradient accumulation, gradient clipping,
    warmup+cosine scheduler, early stopping on validation Macro-F1 (default),
    checkpointing with full metric metadata.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device,
                 scheduler=None, grad_accum_steps=1, max_grad_norm=1.0,
                 monitor="macro_f1", monitor_mode="max", use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_grad_norm = max_grad_norm
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.history = []

    def _unpack(self, batch):
        if isinstance(batch, dict) and "input_ids" in batch:
            return (
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["labels"].to(self.device),
                "transformer",
            )
        inputs, labels = batch
        return inputs.to(self.device), None, labels.to(self.device), "classical"

    def train_epoch(self):
        self.model.train()
        total_loss, n_batches = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            input_ids, mask, labels, kind = self._unpack(batch)
            with autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(input_ids, mask) if kind == "transformer" else self.model(input_ids)
                loss = self.loss_fn(outputs, labels) / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            input_ids, mask, labels, kind = self._unpack(batch)
            with autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(input_ids, mask) if kind == "transformer" else self.model(input_ids)
                loss = self.loss_fn(outputs, labels)
            total_loss += loss.item()
            n_batches += 1
            probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds),
                                  y_prob=np.array(all_probs) if all_probs else None)
        metrics["loss"] = total_loss / max(1, n_batches)
        metrics["y_true"] = np.array(all_labels)
        metrics["y_pred"] = np.array(all_preds)
        metrics["y_prob"] = np.array(all_probs) if all_probs else None
        return metrics

    def save_checkpoint(self, path, epoch, val_metrics):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "val_metrics": {k: v for k, v in val_metrics.items()
                            if isinstance(v, (int, float, str))},
        }, path)

    def train(self, epochs, save_path="best_model.pt", patience=3, registry_entry=None):
        best_score = -float("inf") if self.monitor_mode == "max" else float("inf")
        patience_counter, best_epoch = 0, 0

        for epoch in range(epochs):
            with Timer() as t:
                train_loss = self.train_epoch()
                val_metrics = self.evaluate(self.val_loader)

            score = val_metrics[self.monitor]
            improved = score > best_score if self.monitor_mode == "max" else score < best_score
            marker = ""
            if improved:
                best_score, best_epoch = score, epoch + 1
                self.save_checkpoint(save_path, epoch + 1, val_metrics)
                patience_counter = 0
                marker = " *"
            else:
                patience_counter += 1

            row = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "seconds": round(t.seconds, 1),
            }
            self.history.append(row)
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} "
                  f"| Val Loss: {val_metrics['loss']:.4f} "
                  f"| Val {self.monitor}: {score:.4f}{marker}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best epoch {best_epoch}).")
                break

        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if registry_entry:
            entry = dict(registry_entry)
            entry.update({
                "epochs": epochs,
                "best_epoch": best_epoch,
                "metrics": {"best_val_" + self.monitor: best_score},
                "checkpoint_hash": None,
                "git_commit": git_commit(),
                "runtime_seconds": sum(h["seconds"] for h in self.history),
            })
            try:
                log_experiment(entry)
            except Exception as e:
                print(f"Registry logging skipped: {e}")

        return best_score
