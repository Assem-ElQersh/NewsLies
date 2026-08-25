"""Shared experiment pipeline used by both numbered notebooks (local dev) and
Kaggle drivers (notebooks/kaggle/*). Every function is self-contained,
deterministic given its seed, and writes artifacts + registry entries.

Kaggle-aware path resolution:
  SPLITS_DIR : $NEWLIES_SPLITS_DIR | repo data/splits | /kaggle/input/afnd-splits-v2
  OUT_DIR    : /kaggle/working (Kaggle) | experiments/ (local)
"""
import json
import os
from collections import Counter
from pathlib import Path


def find_root() -> Path:
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    for p in [here] + list(here.parents):
        if (p / "src" / "data" / "loader.py").exists():
            return p
    raise RuntimeError("repo root not found")


ROOT = find_root()


def ctx(require_splits=False):
    root = ROOT
    os.chdir(root)
    if str(root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(root))

    out_dir = Path("/kaggle/working") if os.path.exists("/kaggle/working") else root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        os.environ.get("NEWLIES_SPLITS_DIR"),
        str(root / "data" / "splits"),
        "/kaggle/input/afnd-splits-v2",
        "/kaggle/input/afnd-splits",
    ]
    splits_dir = next((c for c in candidates if c and Path(c, "train_rand.parquet").exists()),
                      None)
    if splits_dir is None and require_splits:
        raise FileNotFoundError(
            "Split parquets not found. Attach 'afnd-splits-v2' dataset on Kaggle "
            "or set NEWLIES_SPLITS_DIR (run stage K1 first).")
    return {"root": root, "out_dir": out_dir,
            "splits_dir": Path(splits_dir) if splits_dir else None,
            "device": __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")}


def _load_splits(splits_dir: Path, kinds=("rand", "src")):
    import pandas as pd
    from src.data.dataset import map_labels

    out = {}
    for kind in kinds:
        parts = []
        for part in ("train", "val", "test"):
            f = splits_dir / f"{part}_{kind}.parquet"
            if f.exists():
                parts.append(pd.read_parquet(f))
        if len(parts) == 3:
            mapped = [map_labels(d)[0] for d in parts]
            out[kind] = tuple(mapped)
    return out


# ---------------------------------------------------------------- audits ----
def run_audit(base_path=None, out_dir="experiments/data_audit", sample_near_dup=None,
              resume_near_dup=True):
    """Full raw-data audit: dates, duplicates (exact + near-dup), plots, reports."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from src.data.loader import load_afnd
    from src.data.cleaner import prepare_corpus
    from src.data.duplicates import exact_duplicate_audit, near_duplicate_audit_fast
    from src.evaluation.analysis import (assert_source_label_consistency,
                                         check_date_reliability, check_length_shortcut)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_raw = load_afnd(base_path)
    sanity = assert_source_label_consistency(df_raw)
    date_report = check_date_reliability(df_raw)
    length_stats = check_length_shortcut(df_raw)

    df = prepare_corpus(df_raw)
    exact_rep = exact_duplicate_audit(df, assign=True)
    df = exact_rep["frame"]

    nd_cache = out / "near_dup_clusters.parquet"
    if resume_near_dup and nd_cache.exists():
        print(f"Resuming near-dup annotations from cache: {nd_cache}")
        cached = pd.read_parquet(nd_cache)
        assert len(cached) == len(df), (
            f"Cache length {len(cached)} != current corpus {len(df)}; "
            "delete the cache file to force a full recomputation.")
        df = df.copy()
        df["near_duplicate_cluster_id"] = cached["near_duplicate_cluster_id"].to_numpy()
        sizes = Counter(cached["near_duplicate_cluster_id"].tolist())
        multi = {cid: n for cid, n in sizes.items() if cid >= 0 and n > 1}
        nd_rep = {
            "articles_audited": len(df), "method": "resumed_from_cache",
            "lsh_candidate_pairs": None, "verified_pairs": None,
            "near_duplicate_clusters": len(multi),
            "articles_in_clusters": int(sum(multi.values())),
            "redundant_articles_if_keep_first": int(sum(n - 1 for n in multi.values())),
            "threshold": 0.8, "num_perm": 128,
        }
        print(f"Near-dup clusters (cached): {len(multi)} covering "
              f"{nd_rep['articles_in_clusters']} articles.")
    else:
        nd_rep = near_duplicate_audit_fast(df, threshold=0.8, num_perm=128,
                                           rows_per_band=8, sample=sample_near_dup,
                                           persist_path=str(nd_cache))
    df = nd_rep.get("frame", df)

    def plot(fig, name):
        fig.tight_layout(); fig.savefig(out / name, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    df["label"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Class distribution"); plot(fig, "class_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(df.groupby("source").size(), bins=40, ax=ax)
    ax.set_title("Articles per source"); plot(fig, "source_count_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(df["text"].str.split().str.len().clip(upper=1500), bins=50, ax=ax)
    ax.set_title("Body length (words)"); plot(fig, "body_length_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(df["title"].str.split().str.len(), bins=40, ax=ax)
    ax.set_title("Title length (words)"); plot(fig, "title_length_distribution.png")

    if date_report.get("monthly_coverage"):
        months = pd.Series(date_report["monthly_coverage"])
        fig, ax = plt.subplots(figsize=(9, 3.2))
        months.plot(ax=ax); ax.set_title("Articles per month")
        plot(fig, "publication_date_distribution.png")

    report = {
        "total_articles": int(len(df_raw)),
        "after_cleaning": int(len(df)),
        "source_label_sanity": sanity,
        "dates": {k: v for k, v in date_report.items() if k != "monthly_coverage"},
        "article_lengths_by_label": length_stats,
        "exact_duplicates": {k: v for k, v in exact_rep.items() if k != "frame"},
        "near_duplicates": {k: v for k, v in nd_rep.items() if k != "frame"},
        "sources_total": int(df["source"].nunique()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "report.md").write_text(f"""# AFND Raw Data Audit

- Raw articles: {report['total_articles']:,} -> after cleaning {report['after_cleaning']:,}
- Sources: {report['sources_total']} (label consistency: {'OK' if sanity['consistent'] else 'VIOLATION'})
- Dates: {date_report['parseable_dates']:,}/{report['total_articles']:,} parseable
  ({date_report['missing_pct']:.2%} missing/invalid, {date_report.get('future_dates',0):,} future-dated);
  range {date_report.get('date_min','?')} -> {date_report.get('date_max','?')}
- Exact duplicates: {exact_rep['exact_duplicate_articles']:,} redundant in {exact_rep['duplicate_groups']:,} groups ({exact_rep['exact_duplicate_pct']:.2%})
- Near-duplicate clusters: {nd_rep['near_duplicate_clusters']:,} covering {nd_rep['articles_in_clusters']:,} articles

Duplicates are annotated, never silently deleted; splits treat clusters as atomic units.
""")
    annotated_path = Path(out_dir).parent / "_annotated_full.parquet"
    df.to_parquet(annotated_path, index=False)
    print(f"Audit complete. Annotated corpus: {annotated_path}")
    return report


# ---------------------------------------------------------------- splits ----
def generate_splits(seed=42, out_dir=None, num_candidates=500, class_tolerance=0.03):
    """Generates all four split families + manifest from the annotated corpus."""
    import json as _json
    import pandas as pd

    from src.data.splits import (random_split, source_disjoint_split, temporal_split,
                                 source_temporal_split, split_fingerprint)

    out_dir = Path(out_dir or (ctx()["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_annotated = [
        out_dir / "_annotated_full.parquet",
        out_dir / "experiments" / "_annotated_full.parquet",
        out_dir / "experiments" / "data_audit" / "_annotated_full.parquet",
        out_dir / "data_audit" / "_annotated_full.parquet",
        out_dir.parent / "_annotated_full.parquet",
        ROOT / "data" / "splits" / "_annotated_full.parquet",
    ]
    annotated = next((c for c in candidates_annotated if c.exists()), None)
    if annotated is None:
        raise FileNotFoundError(
            f"_annotated_full.parquet not found in {[str(c) for c in candidates_annotated]}. "
            "Run run_audit (stage K1 step 1) first.")
    df = pd.read_parquet(annotated)

    train_rand, val_rand, test_rand = random_split(df, seed=seed)
    train_src, val_src, test_src = source_disjoint_split(
        df, seed=seed, num_candidates=num_candidates, class_tolerance=class_tolerance)

    meta = {"seed": seed,
            "random": {"fingerprint": split_fingerprint(train_rand)},
            "source_disjoint": {
                "fingerprint": split_fingerprint(pd.concat([train_src, val_src, test_src])),
                "class_tolerance": class_tolerance,
                "num_candidates": num_candidates,
                "test_sources": sorted(test_src["source"].unique().tolist())}}
    temporal_ok = False
    try:
        train_tmp, val_tmp, test_tmp = temporal_split(df)
        train_st, val_st, test_st = source_temporal_split(df, seed=seed)
        temporal_ok = True
    except ValueError as e:
        print(f"Temporal skipped: {e}")

    frames = [("rand", train_rand, val_rand, test_rand),
              ("src", train_src, val_src, test_src)]
    if temporal_ok:
        frames += [("tmp", train_tmp, val_tmp, test_tmp),
                   ("st", train_st, val_st, test_st)]
        meta["temporal"] = {"fingerprint": split_fingerprint(train_tmp)}
        meta["source_temporal"] = {"fingerprint": split_fingerprint(train_st)}

    for kind, tr, va, te in frames:
        for name, fr in (("train", tr), ("val", va), ("test", te)):
            fr.to_parquet(out_dir / f"{name}_{kind}.parquet", index=False)

    (out_dir / "split_manifest.json").write_text(_json.dumps(meta, indent=2, default=str))
    print("Splits written to", out_dir)
    return meta


# ------------------------------------------------------------- training -----
def train_transformer(model_name, split_kind, max_length, seed=42, epochs=3,
                      batch_size=16, lr=2e-5, grad_accum=1, loss_type="ce",
                      max_train_articles=None, save_tag=None, log_registry=True):
    """Controlled transformer run. Only declared args vary between runs."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from src.data.dataset import TransformerDataset
    from src.data.preprocessing import build_preprocessor
    from src.models.transformer_clf import TransformerClassifier
    from src.training.trainer import Trainer
    from src.training.losses import get_loss_fn
    from src.training.scheduler import get_scheduler
    from src.training.seed import seed_everything
    from src.evaluation.metrics import compute_metrics

    env = ctx(require_splits=True)
    device = env["device"]
    seed_everything(seed)

    train_df, val_df, test_df = _load_splits(env["splits_dir"], kinds=(split_kind,))[split_kind]
    if max_train_articles and max_train_articles < len(train_df):
        train_df = train_df.sample(n=max_train_articles, random_state=seed)

    prep = build_preprocessor("transformer", model_name=model_name)

    def text_col(d):
        s = d["title"].fillna("").astype(str) + " " + d["text"].fillna("").astype(str)
        return s.map(prep)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mk = lambda d, sh: DataLoader(TransformerDataset(text_col(d), d["label_id"], tokenizer,
                                                     max_len=max_length),
                                  batch_size=batch_size, shuffle=sh)
    train_loader, val_loader, test_loader = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    model = TransformerClassifier(model_name, num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs // grad_accum
    scheduler = get_scheduler(optimizer, warmup_steps=int(0.06 * total_steps),
                              total_steps=total_steps)
    loss_fn = get_loss_fn(loss_type)

    tag = save_tag or f"{model_name.split('/')[-1]}_{split_kind}_len{max_length}_seed{seed}"
    trainer = Trainer(model, train_loader, val_loader, optimizer, loss_fn, device,
                      scheduler=scheduler, grad_accum_steps=grad_accum)
    best_val = trainer.train(epochs=epochs, save_path=str(env["out_dir"] / f"{tag}.pt"))

    tm = trainer.evaluate(test_loader)
    payload = {"tag": tag, "model": model_name, "split": split_kind,
               "max_length": max_length, "seed": seed, "epochs": epochs,
               "best_val_macro_f1": best_val,
               "metrics": {k: v for k, v in tm.items()
                           if isinstance(v, (int, float))}}

    import numpy as np
    np.savez_compressed(env["out_dir"] / f"{tag}_predictions.npz",
                        y_true=np.asarray(tm["y_true"]), y_pred=np.asarray(tm["y_pred"]),
                        groups=test_df["source"].to_numpy())
    with open(env["out_dir"] / f"{tag}_metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    if log_registry:
        from src.training.registry import log_experiment, git_commit, configuration_hash
        log_experiment({
            "experiment_id": tag, "model": model_name, "dataset": "AFND",
            "split_type": split_kind, "split_seed": 42, "training_seed": seed,
            "tokenizer": model_name, "max_length": max_length,
            "train_size": len(train_df), "validation_size": len(val_df),
            "test_size": len(test_df), "loss": loss_type, "optimizer": "adamw",
            "learning_rate": lr, "scheduler": "cosine_warmup", "epochs": epochs,
            "metrics": payload["metrics"], "git_commit": git_commit(),
            "gpu": __import__("torch").cuda.get_device_name(0)
                   if __import__("torch").cuda.is_available() else "cpu",
            "configuration_hash": configuration_hash(payload),
        })
    return payload


# ---------------------------------------------------------- calibration ----
def calibrate_checkpoint(checkpoint, model_name, split_kind="src", out_dir=None):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from src.data.dataset import TransformerDataset, map_labels
    from src.models.transformer_clf import TransformerClassifier
    from src.evaluation.calibration import calibration_report

    env = ctx(require_splits=True)
    device = env["device"]
    out_dir = Path(out_dir or env["out_dir"] / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TransformerClassifier(model_name, num_classes=3).to(device)
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(blob.get("model_state_dict", blob))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def logits_of(path_or_df):
        df = pd.read_parquet(path_or_df) if isinstance(path_or_df, (str, Path)) else path_or_df
        df, _ = map_labels(df)
        loader = DataLoader(TransformerDataset(df["title"] + " " + df["text"],
                                               df["label_id"], tokenizer, max_len=128),
                            batch_size=32)
        ls, ys = [], []
        with torch.no_grad():
            for b in loader:
                ls.append(model(b["input_ids"].to(device),
                                b["attention_mask"].to(device)).float().cpu().numpy())
                ys.extend(b["labels"].tolist())
        return np.concatenate(ls), np.array(ys)

    val_logits, val_labels = logits_of(env["splits_dir"] / f"val_{split_kind}.parquet")
    test_logits, test_labels = logits_of(env["splits_dir"] / f"test_{split_kind}.parquet")
    from src.evaluation.calibration import softmax_np
    report = calibration_report(test_labels, softmax_np(test_logits),
                                val_logits=val_logits, val_labels=val_labels,
                                out_dir=str(out_dir), prefix=f"{split_kind}_")
    compact = {"temperature": report.get("temperature"),
               "ece_before": report["before"]["ece"]["ece"],
               "brier_before": report["before"]["brier"],
               "ece_after": report["after"]["ece"]["ece"] if report["after"] else None}
    (out_dir / "calibration_summary.json").write_text(json.dumps(compact, indent=2))
    return compact


# ---------------------------------------------------------- distillation ---
def distill_bigru_student(teacher_ckpt, teacher_name, alpha=0.5, T=4.0, epochs=5,
                          batch_size=32, lr=2e-3, vocab_size=30000, seed=42):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from src.data.dataset import TransformerDataset, ClassicalDataset, build_vocab
    from src.models.bigru_attention import BiGRUAttention
    from src.models.distillation import DistillationTrainer
    from src.models.transformer_clf import TransformerClassifier
    from src.evaluation.metrics import compute_metrics

    env = ctx(require_splits=True)
    device = env["device"]
    teacher = TransformerClassifier(teacher_name, num_classes=3)
    blob = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(blob.get("model_state_dict", blob))

    results = {}
    for kind in ("rand", "src"):
        train_df, val_df, test_df = _load_splits(env["splits_dir"], kinds=(kind,))[kind]

        tok_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        t_train = DataLoader(TransformerDataset(train_df["title"] + " " + train_df["text"],
                                                train_df["label_id"], tok_tokenizer,
                                                max_len=128),
                             batch_size=batch_size, shuffle=True)
        t_val = DataLoader(TransformerDataset(val_df["title"] + " " + val_df["text"],
                                              val_df["label_id"], tok_tokenizer, max_len=128),
                           batch_size=batch_size)

        vocab = build_vocab(train_df["title"] + " " + train_df["text"],
                            vocab_size=vocab_size, seed=seed)
        c_mk = lambda d, sh: DataLoader(ClassicalDataset(d["title"] + " " + d["text"],
                                                         d["label_id"], vocab, max_len=128),
                                        batch_size=batch_size, shuffle=sh)

        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = BiGRUAttention(vocab_size=len(vocab), num_classes=3)

            def forward(self, input_ids, attention_mask=None, **_):
                return self.net(input_ids)

        torch.manual_seed(seed)
        student = Student()
        distiller = DistillationTrainer(student, teacher, t_train, t_val, device,
                                        alpha=alpha, T=T, lr=lr)
        best = distiller.fit(epochs=epochs,
                             save_path=str(env["out_dir"] / f"student_bigru_{kind}.pt"))
        m = distiller.evaluate(c_mk(test_df, False))
        results[kind] = {"best_val_macro_f1": best, "test_macro_f1": m["macro_f1"]}
        print(f"[distill:{kind}] test macro_f1={m['macro_f1']:.4f}")

    (env["out_dir"] / "distillation_results.json").write_text(json.dumps({
        "alpha": alpha, "T": T, "student": "BiGRU+attention", **results}, indent=2))
    return results
