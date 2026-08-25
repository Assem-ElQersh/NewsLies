import numpy as np


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42, groups: np.ndarray = None) -> dict:
    """Bootstrap CI for any metric.

    groups: optional array defining resampling units. When provided (e.g.
    source IDs in a source-disjoint evaluation), clusters of articles are
    resampled instead of individual articles — the unit is reported explicitly.
    """
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)

    if groups is not None:
        units = {}
        for i, g in enumerate(groups):
            units.setdefault(g, []).append(i)
        unit_list = [np.array(v) for v in units.values()]
    else:
        unit_list = None

    scores = []
    for _ in range(n_boot):
        if unit_list is not None:
            chosen = rng.choice(len(unit_list), size=len(unit_list), replace=True)
            idx = np.concatenate([unit_list[c] for c in chosen])
        else:
            idx = rng.randint(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "ci_low": float(np.percentile(scores, 100 * alpha / 2)),
        "ci_high": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "n_boot": n_boot,
        "resampling_unit": "group" if unit_list is not None else "article",
        "n_groups": len(unit_list) if unit_list is not None else None,
    }
