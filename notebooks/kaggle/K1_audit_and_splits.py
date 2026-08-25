# %% [markdown]
# # K1 — Audit + Splits (Kaggle, CPU)
#
# Stage 1 of the Kaggle pipeline. Input: the AFND raw dataset.
# Output (in /kaggle/working): annotated corpus, all split parquets,
# split manifest, full data-audit reports and plots.
#
# After it finishes: download the output and re-upload as dataset
# `afnd-splits-v2` for stages 2–6.

# %%
!if [ -d /kaggle/working/NewsLies/.git ]; then git -C /kaggle/working/NewsLies pull --ff-only; else git clone --depth 1 https://github.com/Assem-ElQersh/NewsLies.git /kaggle/working/NewsLies; fi
%cd /kaggle/working/NewsLies

# %%
!pip install -q datasketch arabert pyarabic

# %%
import sys
sys.path.insert(0, "/kaggle/working/NewsLies")

from src.pipelines import run_audit, generate_splits

# %% [markdown]
# ## 1. Full raw-data audit
# `sample_near_dup=None` audits near-duplicates on the FULL corpus (~30-60 min CPU).
# Set an integer for a quicker exploratory pass.

# %%
report = run_audit(
    base_path=None,
    out_dir="/kaggle/working/experiments/data_audit",
    sample_near_dup=None,
)

# %% [markdown]
# ## 2. Generate all four split families
# random / source-disjoint / temporal / source+temporal — duplicate-cluster-safe,
# class-tolerance-constrained.

# %%
meta = generate_splits(seed=42, out_dir="/kaggle/working")

# %% [markdown]
# ## 3. Summary

# %%
print("Audit highlights:")
print("  dates parseable:", report["dates"]["parseable_dates"], "/", report["total_articles"],
      f"({1 - report['dates']['missing_pct']:.2%})")
print("  exact duplicate articles:", report["exact_duplicates"]["exact_duplicate_articles"])
print("  near-dup clusters:", report["near_duplicates"]["near_duplicate_clusters"])
print()
import json
manifest = json.load(open("/kaggle/working/split_manifest.json"))
print(json.dumps(manifest, indent=2, default=str)[:2000])
print("""
NEXT: download /kaggle/working output -> upload as Kaggle dataset 'afnd-splits-v2'.
""")
