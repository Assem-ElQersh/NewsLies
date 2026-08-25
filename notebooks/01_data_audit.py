# %% [markdown]
# # 01 — Raw Data Audit (Phase A)
#
# Audits the AFND dataset directly from raw article objects. Replaces the
# historical unverified claim of "100% missing publication dates".
# Outputs: experiments/data_audit/report.json, report.md, plots.

# %%
import json
import os
import sys
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
for p in [HERE] + list(HERE.parents):
    if (p / "src" / "data" / "loader.py").exists():
        ROOT = p
        break
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_afnd
from src.data.cleaner import prepare_corpus
from src.data.duplicates import exact_duplicate_audit, near_duplicate_audit_fast
from src.evaluation.analysis import (
    assert_source_label_consistency,
    check_date_reliability,
    check_length_shortcut,
)

OUT = Path("experiments/data_audit")
OUT.mkdir(parents=True, exist_ok=True)

# %%
df_raw = load_afnd()

# %%
sanity = assert_source_label_consistency(df_raw)

# %%
date_report = check_date_reliability(df_raw)

# %%
length_stats = check_length_shortcut(df_raw)

# %%
print("Applying conservative cleaning (no Arabic character folding)...")
df = prepare_corpus(df_raw)

# %%
exact_rep = exact_duplicate_audit(df, assign=True)
df = exact_rep["frame"]

# %% [markdown]
# ## Near-duplicate audit
# Full corpus via vectorized signatures + LSH banding + exact Jaccard.
# Set SAMPLE to an integer for a quicker exploratory pass; None = full data.

# %%
SAMPLE = None
nd_rep = near_duplicate_audit_fast(
    df, threshold=0.8, num_perm=128, rows_per_band=8, sample=SAMPLE,
    persist_path="experiments/data_audit/near_dup_clusters.parquet",
)
df = nd_rep["frame"]

# %%
def plot_and_save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.5))
df["label"].value_counts().plot(kind="bar", ax=ax, color=["#22c55e", "#ef4444", "#f59e0b"])
ax.set_title("Class distribution"); ax.set_ylabel("articles")
plot_and_save(fig, "class_distribution.png")

fig, ax = plt.subplots(figsize=(6, 3.5))
src_counts = df.groupby("source").size().sort_values(ascending=False)
sns.histplot(src_counts, bins=40, ax=ax)
ax.set_title("Articles per source"); ax.set_xlabel("articles")
plot_and_save(fig, "source_count_distribution.png")

fig, ax = plt.subplots(figsize=(6, 3.5))
wl = df["text"].str.split().str.len()
sns.histplot(wl.clip(upper=1500), bins=50, ax=ax)
ax.set_title("Article body length (words, clipped at 1500)")
plot_and_save(fig, "body_length_distribution.png")

fig, ax = plt.subplots(figsize=(6, 3.5))
tl = df["title"].str.split().str.len()
sns.histplot(tl, bins=40, ax=ax)
ax.set_title("Title length (words)")
plot_and_save(fig, "title_length_distribution.png")

if date_report.get("monthly_coverage"):
    months = pd.Series(date_report["monthly_coverage"])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    months.plot(ax=ax)
    ax.set_title("Articles per month"); ax.set_xlabel("month")
    plot_and_save(fig, "publication_date_distribution.png")

# %%
report = {
    "total_articles": int(len(df_raw)),
    "after_cleaning": int(len(df)),
    "source_label_sanity": sanity,
    "dates": {k: v for k, v in date_report.items() if k != "monthly_coverage"},
    "monthly_articles_top": dict(list(date_report.get("monthly_coverage", {}).items())[:60]),
    "article_lengths_by_label": length_stats,
    "exact_duplicates": {k: v for k, v in exact_rep.items() if k != "frame"},
    "near_duplicates": {k: v for k, v in nd_rep.items() if k != "frame"},
    "sources_total": int(df["source"].nunique()),
}
with open(OUT / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

md = f"""# AFND Raw Data Audit

- Raw articles: **{report['total_articles']:,}** → after conservative cleaning: **{report['after_cleaning']:,}**
- Sources: {report['sources_total']} — label consistency: {'OK' if sanity['consistent'] else 'VIOLATION'}
- Publication dates: **{date_report['parseable_dates']:,}/{report['total_articles']:,} parseable ({1 - date_report['missing_pct']:.2%})**, range {date_report.get('date_min','?')} → {date_report.get('date_max','?')}
- Exact duplicates: {exact_rep['exact_duplicate_articles']:,} redundant articles in {exact_rep['duplicate_groups']:,} groups ({exact_rep['exact_duplicate_pct']:.2%})
- Near-duplicate clusters: {nd_rep['near_duplicate_clusters']:,} covering {nd_rep['articles_in_clusters']:,} articles; {nd_rep['redundant_articles_if_keep_first']:,} redundant under keep-first policy
- Duplicate handling: audited and annotated (`dup_group_id`, `near_duplicate_cluster_id`); nothing silently deleted. Splits treat clusters as atomic units.

## Interpretation contract
The near-duplicate and leakage audits describe *dataset structure*; they do not by themselves prove what any classifier learned. Causal claims require the controlled random-vs-disjoint training comparison (notebook 03/07).
"""
with open(OUT / "report.md", "w", encoding="utf-8") as f:
    f.write(md)
print(md)

# %%
df.to_parquet("data/splits/_annotated_full.parquet", index=False)
print("Saved audit-annotated corpus to data/splits/_annotated_full.parquet")
