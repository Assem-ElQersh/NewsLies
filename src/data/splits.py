import hashlib
import json
from typing import Tuple

import numpy as np
import pandas as pd

TARGET_RATIOS = {"train": 0.64, "val": 0.16, "test": 0.20}


def split_fingerprint(df: pd.DataFrame, extra: dict = None) -> str:
    payload = {
        "n": int(len(df)),
        "sources": df["source"].value_counts().sort_index().to_dict() if "source" in df else None,
        "label_counts": df["label"].value_counts().to_dict(),
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _unit_keys(df: pd.DataFrame) -> pd.Series:
    """Constraint units: near-duplicate clusters and exact-duplicate groups must
    never be separated across splits. Un-grouped rows are their own unit."""
    keys = []
    for i in range(len(df)):
        nd = df["near_duplicate_cluster_id"].iloc[i] if "near_duplicate_cluster_id" in df else -1
        dup = df["dup_group_id"].iloc[i] if "dup_group_id" in df else -1
        if isinstance(nd, (int, np.integer)) and nd >= 0:
            keys.append(f"nd:{nd}")
        elif isinstance(dup, str) and dup.startswith("dup_"):
            keys.append(dup)
        else:
            keys.append(f"row:{df.index[i]}")
    return pd.Series(keys, index=df.index)


def _assign_units_to_splits(units: pd.Series, labels: pd.Series, seed: int,
                            ratios: dict = TARGET_RATIOS) -> pd.Series:
    rng = np.random.RandomState(seed)
    assignment = pd.Series(index=units.index, dtype=object)
    for label in sorted(labels.unique(), key=str):
        idx = labels[labels == label].index
        seen_units, ordered = set(), []
        for i in idx:
            u = units.at[i]
            if u not in seen_units:
                seen_units.add(u)
                ordered.append(u)
        rng.shuffle(ordered)
        unit_sizes = {u: int((units.loc[idx] == u).sum()) for u in ordered}
        target_total = len(idx)
        counts = {s: 0 for s in ratios}
        for u in ordered:
            deficits = {s: ratios[s] * target_total - counts[s] for s in ratios}
            best = max(deficits, key=deficits.get)
            assignment.loc[units.loc[idx][units.loc[idx] == u].index] = best
            counts[best] += unit_sizes[u]
    return assignment


def random_split(df: pd.DataFrame, seed: int = 42,
                 ratios: dict = TARGET_RATIOS) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """64/16/20 stratified random split over duplicate-safe units."""
    units = _unit_keys(df)
    assignment = _assign_units_to_splits(units, df["label"], seed, ratios)
    parts = []
    for name in ("train", "val", "test"):
        part = df[assignment == name].copy()
        part["split"] = name
        parts.append(part)
    train, val, test = parts
    print(f"Random split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return (
        train.drop(columns=["split"]),
        val.drop(columns=["split"]),
        test.drop(columns=["split"]),
    )


def source_disjoint_split(
    df: pd.DataFrame,
    seed: int = 42,
    num_candidates: int = 500,
    class_tolerance: float = 0.03,
    ratios: dict = TARGET_RATIOS,
):
    """Constrained repeated search for zero-source-overlap allocation.

    Each candidate assigns whole sources to train/val/test while balancing
    article count, source count and class proportions simultaneously.
    Candidates whose per-split class proportions deviate more than
    `class_tolerance` from the global distribution are rejected.
    """
    rng_master = np.random.RandomState(seed)
    source_totals = df.groupby("source").size()
    src_label_counts = {}
    for (src, lbl), c in df.groupby(["source", "label"]).size().items():
        src_label_counts.setdefault(src, {})[lbl] = int(c)
    global_label_dist = df["label"].value_counts(normalize=True).to_dict()
    total = int(source_totals.sum())
    n_sources = len(source_totals)

    best_score, best_alloc, n_valid = float("inf"), None, 0
    for _ in range(num_candidates):
        shuffled = list(source_totals.index)
        rng_master.shuffle(shuffled)
        art = {s: 0 for s in ratios}
        nsrc = {s: 0 for s in ratios}
        lc = {s: {} for s in ratios}
        alloc = {}
        for src in shuffled:
            c = int(source_totals[src])
            scored = {}
            for s in ratios:
                cls_dev = sum(
                    abs(lc[s].get(lbl, 0) / max(1, art[s]) - gp) for lbl, gp in global_label_dist.items()
                )
                scored[s] = (
                    ratios[s] - art[s] / total
                    + 0.25 * (ratios[s] - nsrc[s] / n_sources)
                    + 0.5 * cls_dev
                )
            chosen = max(scored, key=scored.get)
            alloc[src] = chosen
            art[chosen] += c
            nsrc[chosen] += 1
            for lbl, cnt in src_label_counts[src].items():
                lc[chosen][lbl] = lc[chosen].get(lbl, 0) + cnt

        valid = all(
            abs(lc[s].get(lbl, 0) / max(1, art[s]) - gp) <= class_tolerance
            for s in ratios for lbl, gp in global_label_dist.items()
        )
        if not valid:
            continue
        n_valid += 1
        art_dev = sum(abs(art[s] / total - ratios[s]) for s in ratios)
        src_dev = sum(abs(nsrc[s] / n_sources - ratios[s]) for s in ratios)
        cls_dev = sum(
            abs(lc[s].get(lbl, 0) / max(1, art[s]) - gp)
            for s in ratios for lbl, gp in global_label_dist.items()
        )
        score = art_dev + 0.25 * src_dev + 0.5 * cls_dev
        if score < best_score:
            best_score, best_alloc = score, dict(alloc)

    if best_alloc is None:
        raise RuntimeError(
            f"No allocation satisfied class_tolerance={class_tolerance}; "
            "relax tolerance or raise num_candidates."
        )
    print(f"Valid allocations: {n_valid}/{num_candidates} | best score: {best_score:.4f}")

    df = df.copy()
    df["split"] = df["source"].map(best_alloc)
    train = df[df["split"] == "train"].drop(columns=["split"])
    val = df[df["split"] == "val"].drop(columns=["split"])
    test = df[df["split"] == "test"].drop(columns=["split"])
    for name, part in (("train", train), ("val", val), ("test", test)):
        dist = part["label"].value_counts(normalize=True).round(3).to_dict()
        print(f"  {name}: {len(part)} articles | {part['source'].nunique()} sources | classes {dist}")
    assert len(set(train["source"]) & set(test["source"])) == 0
    assert len(set(train["source"]) & set(val["source"])) == 0
    return train, val, test


def temporal_split(df: pd.DataFrame, date_col: str = "published_date",
                   ratios=(0.70, 0.10, 0.20), min_coverage: float = 0.40):
    """Chronological split on verifiably dated articles.

    Dates that are missing, unparseable or in the future are excluded from the
    timeline; the split then runs on the dated subset and its coverage is
    reported explicitly so no one mistakes it for a full-corpus benchmark.
    Refuses to run below `min_coverage`.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    valid = df[date_col].notna() & (df[date_col] <= now + pd.Timedelta(days=1))
    coverage = valid.mean()
    if coverage < min_coverage:
        raise ValueError(
            f"Temporal split rejected: only {coverage:.1%} of articles have a "
            f"usable (parseable, non-future) date (< {min_coverage:.0%}). "
            "Document the failure instead of splitting blindly."
        )
    print(f"Temporal coverage: {coverage:.1%} of corpus has usable dates; "
          f"splitting the dated subset ({int(valid.sum())} articles).")
    df = df[valid].sort_values(date_col).reset_index(drop=True)
    n = len(df)
    t1, t2 = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
    train, val, test = df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()
    print(f"Temporal split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"  boundaries: {train[date_col].max()} | {val[date_col].max()} | {test[date_col].max()}")
    return train, val, test


def source_temporal_split(
    df: pd.DataFrame,
    date_col: str = "published_date",
    seed: int = 42,
    num_candidates: int = 300,
    test_article_target: float = 0.20,
):
    """Hardest benchmark: unseen sources AND later dates.

    Design:
      1. Whole sources are allocated to TEST until ~test_article_target articles.
      2. Articles of training sources are ordered chronologically:
         earliest ~85% -> train, latest ~15% -> val.
      3. TEST keeps only unseen-source articles dated at or after the start of
         the validation window, so test is strictly future relative to train
         and its publishers were never seen.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df = df[df[date_col].notna() & (df[date_col] <= now + pd.Timedelta(days=1))]
    if df.empty:
        raise ValueError("No usable dates for source+temporal split.")
    df = df.sort_values(date_col).reset_index(drop=True)
    print(f"Source+temporal: using {len(df)} dated articles "
          f"({len(df)/max(1,len(df)):.0%} of input kept after date validity filter)")

    source_totals = df.groupby("source").size()
    order = list(source_totals.sample(frac=1.0, random_state=rng.randint(2**31 - 1)).sort_values().index)
    total = int(source_totals.sum())
    test_sources, acc = set(), 0
    for src in order:
        if acc / total >= test_article_target:
            break
        test_sources.add(src)
        acc += int(source_totals[src])

    non_test = df[~df["source"].isin(test_sources)]
    cutoff = non_test[date_col].quantile(0.85)
    train = non_test[non_test[date_col] < cutoff]
    val = non_test[non_test[date_col] >= cutoff]
    val_start = val[date_col].min()
    test = df[df["source"].isin(test_sources) & (df[date_col] >= val_start)]

    print(f"Source+temporal split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"  Test sources: {len(test_sources)} (unseen), val window starts {val_start}")
    print(f"  Class dists: train {train['label'].value_counts(normalize=True).round(3).to_dict()}, "
          f"test {test['label'].value_counts(normalize=True).round(3).to_dict()}")
    return train, val, test
