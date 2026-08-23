from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, accuracy_score

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes primary and secondary evaluation metrics.
    Macro-F1 is the primary metric for model selection.
    """
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred)
    }
    return metrics
