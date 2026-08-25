import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)


def compute_metrics(y_true, y_pred, y_prob=None, label_names=None) -> dict:
    """Primary metric: Macro-F1. Returns the full reporting bundle.

    Backward-compatible keys (macro_f1 / balanced_accuracy / mcc / accuracy) are
    preserved; weighted_f1, per-class report and confusion matrix are added.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_prob is not None:
        try:
            from sklearn.metrics import log_loss
            metrics["log_loss"] = float(log_loss(y_true, np.asarray(y_prob),
                                                 labels=list(range(np.asarray(y_prob).shape[1]))))
        except Exception:
            pass

    from sklearn.metrics import classification_report
    labels_sorted = sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)
    names = label_names or [str(l) for l in labels_sorted]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["per_class"] = {name: report[str(l)] for name, l in zip(names, labels_sorted)
                            if str(l) in report}
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=labels_sorted
    ).tolist()
    return metrics


def print_metrics(metrics: dict, title: str = ""):
    if title:
        print(f"--- {title} ---")
    for k in ("macro_f1", "weighted_f1", "balanced_accuracy", "mcc", "accuracy"):
        if k in metrics:
            print(f"{k:>18}: {metrics[k]:.4f}")
