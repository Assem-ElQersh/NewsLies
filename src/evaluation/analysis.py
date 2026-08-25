import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report


def assert_source_label_consistency(df: pd.DataFrame) -> dict:
    """Sanity check on dataset construction (NOT a leakage discovery):
    every source must map to exactly one label, since AFND labels were assigned
    at publisher level via Misbar."""
    n_labels_per_source = df.groupby("source")["label"].nunique()
    bad = n_labels_per_source[n_labels_per_source > 1]
    n_sources = df["source"].nunique()
    result = {
        "sources_total": int(n_sources),
        "sources_with_multiple_labels": int(len(bad)),
        "consistent": bool(len(bad) == 0),
    }
    if len(bad) == 0:
        print(f"Sanity check passed: {n_sources}/{n_sources} sources have exactly one label.")
    else:
        print(f"WARNING: {len(bad)} sources carry multiple labels: {bad.index.tolist()[:10]}")
    return result


def test_text_to_source_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                max_features_word: int = 30000,
                                max_features_char: int = 50000,
                                max_train_articles: int = None,
                                seed: int = 42) -> dict:
    """Publisher-identifiability probe.

    TF-IDF word + char n-grams -> LinearSVC over source IDs.
    Interpretation contract: a high score means article text contains strong
    publisher-identifying information, creating a shortcut opportunity. It does
    NOT by itself prove the credibility classifier only memorizes sources —
    that conclusion requires the controlled random vs source-disjoint training
    comparison (Phase B4).
    """
    from sklearn.svm import LinearSVC

    if max_train_articles and max_train_articles < len(train_df):
        train_df = train_df.sample(n=max_train_articles, random_state=seed)

    def join(d):
        return (d["title"].fillna("").astype(str) + " " + d["text"].fillna("").astype(str))

    print(f"Fitting TF-IDF + LinearSVC source probe "
          f"({len(train_df)} train / {len(test_df)} eval articles)...")
    word_vec = TfidfVectorizer(max_features=max_features_word, ngram_range=(1, 2), sublinear_tf=True)
    char_vec = TfidfVectorizer(max_features=max_features_char, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)

    Xw_tr = word_vec.fit_transform(join(train_df))
    Xc_tr = char_vec.fit_transform(join(train_df))
    from scipy.sparse import hstack
    X_tr = hstack([Xw_tr, Xc_tr]).tocsr()

    y_tr = train_df["source"].values
    clf = LinearSVC(class_weight=None, C=1.0, max_iter=3000)
    clf.fit(X_tr, y_tr)

    X_te = hstack([
        word_vec.transform(join(test_df)),
        char_vec.transform(join(test_df)),
    ]).tocsr()
    y_te = test_df["source"].values

    scores = clf.decision_function(X_te)
    top1 = scores.argmax(axis=1)
    order = np.argsort(-scores, axis=1)[:, :5]
    top5 = np.mean([y_te[i] in order[i] for i in range(len(y_te))])
    acc = float((top1 == y_te).mean())
    macro_f1 = float(f1_score(y_te, top1, average="macro"))

    n_classes = len(np.unique(y_tr))
    chance = 1.0 / max(1, n_classes)
    print(f"Source prediction accuracy: {acc:.4f} (chance ≈ {chance:.4f}, {n_classes} classes)")
    print(f"Source prediction top-5 accuracy: {top5:.4f}")
    print(f"Source prediction macro-F1: {macro_f1:.4f}")

    return {
        "accuracy": acc,
        "top5_accuracy": float(top5),
        "macro_f1": macro_f1,
        "n_classes": int(n_classes),
        "chance_accuracy": chance,
        "train_size": len(y_tr),
        "eval_size": len(y_te),
        "confusion_top_pairs": _top_confusions(y_te, top1),
    }


def _top_confusions(y_true, y_pred, k: int = 15):
    pairs = pd.DataFrame({"true": y_true, "pred": y_pred})
    mis = pairs[pairs["true"] != pairs["pred"]]
    grouped = mis.groupby(["true", "pred"]).size().sort_values(ascending=False).head(k)
    return [{"true": t, "predicted": p, "count": int(c)} for (t, p), c in grouped.items()]


def check_length_shortcut(df: pd.DataFrame) -> dict:
    """Is article length alone predictive of the label?"""
    df = df.copy()
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    stats = {}
    for label in sorted(df["label"].unique(), key=str):
        s = df.loc[df["label"] == label, "word_count"]
        stats[str(label)] = {"mean": round(float(s.mean()), 1), "median": float(s.median())}
        print(f"Avg length for {label}: {s.mean():.1f} words (median {s.median():.0f})")
    return stats


def check_date_reliability(df: pd.DataFrame, date_col: str = "published_date") -> dict:
    """Audited directly from raw data — replaces the unverified historical claim."""
    total = len(df)
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    missing = int(dates.isna().sum())
    parsed_valid = int((~dates.isna()).sum())
    future = int(((dates > pd.Timestamp.now(tz="UTC")) & dates.notna()).sum())

    report = {
        "total_articles": total,
        "missing_or_invalid_dates": missing,
        "missing_pct": missing / max(1, total),
        "parseable_dates": parsed_valid,
        "future_dates": future,
    }
    valid = dates.dropna()
    if len(valid):
        report["date_min"] = str(valid.min())
        report["date_max"] = str(valid.max())
        report["unique_dates"] = int(valid.nunique())
        report["monthly_coverage"] = {
            str(period): int(cnt)
            for period, cnt in valid.dt.to_period("M").value_counts().sort_index().items()
        }
    print(f"Dates audited: {parsed_valid}/{total} parseable ({report['missing_pct']:.2%} missing/invalid)")
    if len(valid):
        print(f"Range: {valid.min().date()} → {valid.max().date()} | unique: {report['unique_dates']}")
    return report


def classification_report_full(y_true, y_pred, label_names=None) -> dict:
    """Full metric bundle for experiment reporting."""
    labels_sorted = sorted(set(y_true) | set(y_pred), key=str)
    names = label_names or [str(l) for l in labels_sorted]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report
