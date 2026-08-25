import torch
import torch.nn as nn

from src.training.losses import distillation_loss


class SmallTransformerStudent(nn.Module):
    """Compact HF-encoder student. Pass any small checkpoint name via config;
    verify availability before benchmarking (no model is invented here)."""

    def __init__(self, model_name: str, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


def freeze_bottom_layers(model: nn.Module, n_layers_to_freeze: int):
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layer", None) or getattr(encoder, "encoder", None)
    layers = getattr(layers, "layer", None) if layers is not None else None
    if layers is not None:
        for layer in list(layers)[:n_layers_to_freeze]:
            for p in layer.parameters():
                p.requires_grad = False


class DistillationTrainer:
    """alpha * CE(hard) + (1-alpha) * KL(student || teacher) * T^2.

    Teacher must produce logits under the same tokenization as the student.
    """

    def __init__(self, student, teacher, train_loader, val_loader, device,
                 alpha=0.5, T=4.0, lr=2e-3, max_grad_norm=1.0, use_amp=True,
                 monitor="macro_f1"):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.alpha = alpha
        self.T = T
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad], lr=lr)
        self.monitor = monitor
        self.history = []
        self.save_path = None

    def _batch_fields(self, batch):
        return (batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["labels"].to(self.device))

    def train_epoch(self):
        self.student.train()
        total = 0.0
        for batch in self.train_loader:
            ids, mask, labels = self._batch_fields(batch)
            with torch.no_grad():
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    teacher_logits = self.teacher(ids, mask).float()
            with torch.autocast(self.device.type, enabled=self.use_amp):
                student_logits = self.student(ids, mask).float()
            loss, _, _ = distillation_loss(
                student_logits, teacher_logits, labels, self.alpha, self.T)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
        return total / max(1, len(self.train_loader))

    @torch.no_grad()
    def evaluate(self, loader):
        from src.evaluation.metrics import compute_metrics

        self.student.eval()
        preds, labels_all = [], []
        for batch in loader:
            ids, mask, labels = self._batch_fields(batch)
            out = self.student(ids, mask)
            preds.extend(out.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
        return compute_metrics(labels_all, preds)

    def fit(self, epochs, save_path="student_best.pt", patience=3):
        self.save_path = save_path
        best = -float("inf")
        patience_counter = 0
        best_state = None
        for epoch in range(epochs):
            tl = self.train_epoch()
            m = self.evaluate(self.val_loader)
            score = m[self.monitor]
            improved = score > best
            if improved:
                best = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.student.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"[distill] Epoch {epoch + 1}/{epochs} | loss {tl:.4f} | "
                  f"val {self.monitor} {score:.4f}{' *' if improved else ''}")
            if patience_counter >= patience:
                print(f"Early stopping (best epoch {epoch + 1 - patience_counter}).")
                break
        if best_state is not None:
            self.student.load_state_dict(best_state)
            if save_path:
                torch.save({"model_state_dict": best_state,
                            "val_" + self.monitor: best}, save_path)
        return best
