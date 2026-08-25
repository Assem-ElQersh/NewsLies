import json
import os

import numpy as np
import torch


def expected_calibration_error(y_true, probs, n_bins: int = 15) -> dict:
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == np.asarray(y_true)).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf, bin_acc = conf[mask].mean(), correct[mask].mean()
        ece += mask.mean() * abs(bin_conf - bin_acc)
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": int(mask.sum()),
                     "confidence": float(bin_conf), "accuracy": float(bin_acc)})
    return {"ece": float(ece), "bins": rows}


def brier_score(y_true, probs) -> float:
    y_true = np.asarray(y_true)
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_temperature(logits, labels, max_iter: int = 200):
    """Fit a single temperature on VALIDATION logits by NLL minimization.
    Never call this with test data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = torch.as_tensor(np.asarray(logits), dtype=torch.float64, device=device)
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    log_T = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T).clamp(1e-2, 100.0)
        loss = torch.nn.functional.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).item())


def reliability_diagram(y_true, probs, out_path: str, title: str = "Reliability Diagram", n_bins: int = 15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = expected_calibration_error(y_true, probs, n_bins)
    rows = report["bins"]
    centers = [(r["bin_low"] + r["bin_high"]) / 2 for r in rows]
    accs = [r["accuracy"] for r in rows]
    counts = [r["count"] for r in rows]

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.bar(centers, accs, width=1 / n_bins, edgecolor="0.3", alpha=0.8, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}\nECE = {report['ece']:.4f}")
    ax.legend(fontsize=8)

    ax2 = ax.twinx()
    ax2.bar(centers, counts, width=1 / n_bins, alpha=0.18, color="gray")
    ax2.set_yscale("log")
    ax2.set_ylabel("Count (log)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return report


def calibration_report(y_true, probs, val_logits=None, val_labels=None,
                       out_dir: str = None, prefix: str = "") -> dict:
    """Full calibration analysis; optional temperature scaling fitted on val."""
    out = {
        "before": {
            "ece": expected_calibration_error(y_true, probs),
            "brier": brier_score(y_true, probs),
        },
    }
    if val_logits is not None and val_labels is not None:
        T = fit_temperature(val_logits, val_labels)
        scaled = softmax_np(np.asarray(probs_to_logits(probs)) / T)
        out["temperature"] = T
        out["after"] = {
            "ece": expected_calibration_error(y_true, scaled),
            "brier": brier_score(y_true, scaled),
        }
        out["calibrated_probs"] = scaled
    else:
        out["after"] = None

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        diagram_data = out["before"]
        reliability_diagram(y_true, probs, os.path.join(out_dir, f"{prefix}reliability_before.png"))
        if out.get("after") is not None:
            reliability_diagram(y_true, out["calibrated_probs"],
                                os.path.join(out_dir, f"{prefix}reliability_after.png"),
                                title="After temperature scaling")
            with open(os.path.join(out_dir, "calibration.json"), "w", encoding="utf-8") as f:
                json.dump({"temperature": out["temperature"]}, f, indent=2)
    return {k: v for k, v in out.items() if k != "calibrated_probs"}


def probs_to_logits(probs, eps=1e-12):
    probs = np.clip(np.asarray(probs), eps, 1 - eps)
    return np.log(probs)


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
