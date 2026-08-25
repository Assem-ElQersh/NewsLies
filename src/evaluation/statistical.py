import numpy as np

from src.evaluation.bootstrap import bootstrap_ci


def paired_model_comparison(metric_a, y_true, pred_a, pred_b, n_boot: int = 1000,
                            seed: int = 42, alpha: float = 0.05, groups=None) -> dict:
    """Delta Macro-F1 between model A and model B with a bootstrap 95% CI.

    Both predictions must be on the same evaluation rows. The CI, not a
    p-value, is the primary result.
    """
    from sklearn.metrics import f1_score

    delta_point = f1_score(y_true, pred_a, average="macro") - f1_score(y_true, pred_b, average="macro")
    rng = np.random.RandomState(seed)
    deltas = []
    n = len(y_true)

    unit_idx = None
    if groups is not None:
        units = {}
        for i, g in enumerate(groups):
            units.setdefault(g, []).append(i)
        unit_idx = [np.array(v) for v in units.values()]

    for _ in range(n_boot):
        if unit_idx is not None:
            chosen = rng.choice(len(unit_idx), size=len(unit_idx), replace=True)
            idx = np.concatenate([unit_idx[c] for c in chosen])
        else:
            idx = rng.randint(0, n, size=n)
        d = (f1_score(y_true[idx], np.asarray(pred_a)[idx], average="macro")
             - f1_score(y_true[idx], np.asarray(pred_b)[idx], average="macro"))
        deltas.append(d)

    deltas = np.array(deltas)
    ci_low = float(np.percentile(deltas, 100 * alpha / 2))
    ci_high = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    crosses_zero = ci_low <= 0 <= ci_high

    if abs(delta_point) < 0.005 or crosses_zero:
        interpretation = "No convincing evidence of superiority."
    elif delta_point > 0:
        interpretation = f"Model A outperforms Model B by {delta_point:.3f} Macro-F1 (CI excludes zero)."
    else:
        interpretation = f"Model B outperforms Model A by {-delta_point:.3f} Macro-F1 (CI excludes zero)."

    return {
        "delta_macro_f1": float(delta_point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_excludes_zero": not crosses_zero,
        "interpretation": interpretation,
        "resampling_unit": "group" if groups is not None else "article",
    }


def multi_seed_summary(results_per_seed: list) -> dict:
    """results_per_seed: list of dicts containing at least {'seed', 'macro_f1'}."""
    scores = np.array([r["macro_f1"] for r in results_per_seed])
    return {
        "n_seeds": len(scores),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "per_seed": results_per_seed,
    }
