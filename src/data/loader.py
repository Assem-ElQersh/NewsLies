import hashlib
import json
import os
from collections import Counter

import pandas as pd


def _candidate_paths() -> list:
    return [
        "data/AFND/AFND",
        "/kaggle/input/arabic-fake-news-dataset-afnd",
        "/kaggle/input/arabic-fake-news-dataset-afnd/AFND",
        "/kaggle/input/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd/AFND",
    ]


def get_dataset_path(base_path: str = None) -> str:
    """Locate the AFND root (directory containing sources.json), trying known layouts.

    Kaggle mounts differ across dataset uploads; we detect rather than assume.
    """
    candidates = [base_path] if base_path else _candidate_paths()
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "sources.json")):
            return cand
    if not base_path and os.path.isdir("/kaggle/input"):
        for dirpath, dirnames, files in os.walk("/kaggle/input"):
            if "sources.json" in files:
                return dirpath
    raise FileNotFoundError(
        "AFND not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + " — pass base_path explicitly or place the dataset under data/AFND/AFND."
    )


def load_afnd(base_path: str = None) -> pd.DataFrame:
    """Load AFND preserving all raw fields present in the article objects.

    The raw per-article JSON keys are: 'title', 'text', 'published date'
    (space-separated). We never fabricate a field: missing values stay NaN/NaT.
    """
    base = get_dataset_path(base_path)
    with open(os.path.join(base, "sources.json"), encoding="utf-8") as f:
        sources_mapping = json.load(f)

    records = []
    missing_sources = 0
    label_counts = Counter()
    key_variants = Counter()

    for source_id, label in sources_mapping.items():
        label_counts[label] += 1
        articles_file = os.path.join(base, "Dataset", str(source_id), "scraped_articles.json")
        if not os.path.exists(articles_file):
            missing_sources += 1
            continue
        with open(articles_file, encoding="utf-8") as f:
            data = json.load(f)
        articles = data["articles"] if isinstance(data, dict) and "articles" in data else data

        for article in articles:
            date_raw = article.get("published date")
            if date_raw is None:
                date_raw = article.get("published_date")
                key_variants["published_date_fallback"] += 1
            key_variants["published date" if "published date" in article else ("published_date" if "published_date" in article else "no_date_field")] += 1
            records.append({
                "source": str(source_id),
                "label": label,
                "title": article.get("title"),
                "text": article.get("text"),
                "published_date": date_raw,
            })

    df = pd.DataFrame(records)
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)

    n = len(df)
    print("--- AFND Loader Summary ---")
    print(f"Base path: {base}")
    print(f"Articles: {n}")
    print(f"Sources in mapping: {len(sources_mapping)} | source dirs missing: {missing_sources}")
    print(f"Sources loaded: {df['source'].nunique()}")
    print(f"Articles per label: {df['label'].value_counts().to_dict()}")
    print(f"Missing title: {df['title'].isna().sum()} ({df['title'].isna().mean():.2%})")
    print(f"Missing text: {df['text'].isna().sum()} ({df['text'].isna().mean():.2%})")
    print(f"Missing/unparseable published_date: {df['published_date'].isna().sum()} ({df['published_date'].isna().mean():.2%})")
    valid_dates = df["published_date"].dropna()
    print(f"Unique dates: {valid_dates.nunique()}")
    if len(valid_dates):
        print(f"Date range: {valid_dates.min()} → {valid_dates.max()}")
    print(f"Raw key variants observed: {dict(key_variants)}")
    return df


if __name__ == "__main__":
    load_afnd()
