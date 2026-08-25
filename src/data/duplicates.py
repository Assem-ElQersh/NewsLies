import hashlib
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _normalize_for_hash(text) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def exact_duplicate_audit(df: pd.DataFrame, assign: bool = True) -> dict:
    """Independent audit of exact duplicates (normalized title + normalized body → SHA-256).

    AFND's authors report deduplication during construction; this verifies it.
    Nothing is deleted from the raw data here — a `dup_group_id` column can be
    attached so the split stage can keep duplicate groups within one split.
    """
    keys = (df["title"].map(_normalize_for_hash) + " \u2016 " + df["text"].map(_normalize_for_hash)).map(_sha256)
    counts = Counter(keys)
    n_dupes = int(sum(c - 1 for c in counts.values() if c > 1))
    n_groups = int(sum(1 for c in counts.values() if c > 1))
    report = {
        "total_articles": len(df),
        "exact_duplicate_articles": n_dupes,
        "exact_duplicate_pct": n_dupes / max(1, len(df)),
        "duplicate_groups": n_groups,
    }
    print(f"Exact duplicates: {n_dupes} redundant articles in {n_groups} groups ({report['exact_duplicate_pct']:.2%}).")

    frame = None
    if assign:
        frame = df.copy()
        frame["dup_group_id"] = [f"dup_{k[:12]}" if counts[k] > 1 else -1 for k in keys]
    report["frame"] = frame
    return report


def _shingles(text: str, k: int = 5):
    words = text.split()
    return (" ".join(words[i:i + k]) for i in range(len(words) - k + 1))


def _token_hash(word: str) -> int:
    import zlib
    return zlib.crc32(word.encode("utf-8")) & 0xFFFFFFFF


def _doc_shingles_uint64(text: str, k: int = 5) -> np.ndarray:
    words = str(text).split()
    if len(words) < k:
        return np.empty(0, dtype=np.uint64)
    h = np.array([_token_hash(w) for w in words], dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(h, k)
    acc = np.zeros(w.shape[0], dtype=np.uint64)
    C = np.uint64(0x100000001B3)
    for j in range(k):
        acc = acc * C + w[:, j]
    return acc


_ODD_MULT = np.uint64(0x9E3779B97F4A7C15)
_PRIME_STEP = np.uint64(0xC2B2AE3D27D4EB4F)


def _signatures_for_docs(texts, num_perm: int, k: int):
    """Vectorized permutation-signatures: sig_p(d) = min_x (M_p * x + S_p) over
    uint64 wrapping arithmetic (odd multiplier M_p, stride S_p)."""
    sigs = np.full((len(texts), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    mult = (_ODD_MULT * (np.arange(num_perm, dtype=np.uint64) + 1))
    step = (_PRIME_STEP * (np.arange(num_perm, dtype=np.uint64) + np.uint64(7919)))
    for i, t in enumerate(texts):
        x = _doc_shingles_uint64(t, k)
        lengths[i] = len(x)
        if len(x) == 0:
            continue
        vals = np.multiply.outer(mult, x) + step[:, None]
        sigs[i] = vals.min(axis=1)
    return sigs, lengths


def near_duplicate_audit(
    df: pd.DataFrame,
    threshold: float = 0.8,
    num_perm: int = 128,
    sample: int = None,
    seed: int = 42,
    persist_path: str = None,
):
    """MinHash/LSH candidate retrieval → exact Jaccard verification → connected components.

    Fixes the previous order-dependent behaviour: LSH results are treated only as
    candidates; membership is decided by verified Jaccard similarity over 5-word
    shingles, and clusters are the connected components of the verified-pair graph.
    Returns the frame with a `near_duplicate_cluster_id` column (-1 = singleton).
    """
    from datasketch import MinHash, MinHashLSH

    work = df.reset_index(drop=True)
    if sample and sample < len(work):
        work = work.sample(n=sample, random_state=seed).reset_index(drop=True)
        print(f"Near-dup audit running on a seeded sample of {len(work)} articles.")

    texts = (work["title"].fillna("").astype(str) + " " + work["text"].fillna("").astype(str)).tolist()

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    shingle_cache = {}
    for idx, text in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        sh = list(_shingles(text))
        shingle_cache[idx] = set(sh)
        for s in sh:
            m.update(s.encode("utf8"))
        lsh.insert(str(idx), m)
        minhashes[str(idx)] = m

    candidate_pairs = set()
    for idx in range(len(work)):
        for other in lsh.query(minhashes[str(idx)]):
            j = int(other)
            if j != idx:
                candidate_pairs.add((min(idx, j), max(idx, j)))
    print(f"LSH candidate pairs: {len(candidate_pairs)}")

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        return inter / (len(a) + len(b) - inter)

    verified_pairs = [(i, j) for i, j in candidate_pairs if jaccard(shingle_cache[i], shingle_cache[j]) >= threshold]
    print(f"Jaccard-verified pairs (>= {threshold}): {len(verified_pairs)}")

    parent = list(range(len(work)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in verified_pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    cluster_of = {}
    next_id = 0
    cluster_ids = []
    members = defaultdict(list)
    for idx in range(len(work)):
        root = find(idx)
        if root not in cluster_of:
            cluster_of[root] = next_id
            next_id += 1
        cid = cluster_of[root]
        cluster_ids.append(cid)
        members[cid].append(idx)

    multi = {cid: mem for cid, mem in members.items() if len(mem) > 1}
    work["near_duplicate_cluster_id"] = [-1 if len(members[c]) == 1 else c for c in cluster_ids]

    report = {
        "articles_audited": len(work),
        "lsh_candidate_pairs": len(candidate_pairs),
        "verified_pairs": len(verified_pairs),
        "near_duplicate_clusters": len(multi),
        "articles_in_clusters": int(sum(len(m) for m in multi.values())),
        "redundant_articles_if_keep_first": int(sum(len(m) - 1 for m in multi.values())),
        "threshold": threshold,
        "num_perm": num_perm,
    }
    print(f"Near-duplicate clusters (>1 member): {len(multi)}, covering {report['articles_in_clusters']} articles.")

    if persist_path:
        import os
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        work[["near_duplicate_cluster_id"]].to_csv(persist_path)
        pd.DataFrame(
            [{"a": i, "b": j} for i, j in verified_pairs]
        ).to_parquet(persist_path.replace(".csv", "_pairs.parquet"), index=False)

    report["frame"] = work
    return report


def _banded_candidates(sigs: np.ndarray, rows_per_band: int = 8,
                       max_bucket: int = 50) -> set:
    n, num_perm = sigs.shape
    bands = []
    for b0 in range(0, num_perm - rows_per_band + 1, rows_per_band):
        bands.append((sigs[:, b0:b0 + rows_per_band]).copy())
    pairs = set()
    skipped_buckets = 0
    for band in bands:
        keys = {}
        for i in range(n):
            key = band[i].tobytes()
            bucket = keys.get(key)
            if bucket is None:
                keys[key] = [i]
            else:
                if len(bucket) >= max_bucket:
                    skipped_buckets += 1
                    continue
                for j in bucket:
                    pairs.add((j, i) if j < i else (i, j))
                bucket.append(i)
    return pairs


def near_duplicate_audit_fast(
    df: pd.DataFrame,
    threshold: float = 0.8,
    num_perm: int = 128,
    rows_per_band: int = 8,
    sample: int = None,
    seed: int = 42,
    persist_path: str = None,
):
    """Full-corpus near-duplicate audit using vectorized permutation signatures.

    Pipeline: 5-word shingles (64-bit hashed) -> min-signatures -> LSH banding
    candidate pairs -> EXACT Jaccard verification on recomputed shingles ->
    connected components -> cluster ids. Signature estimation never decides
    membership by itself; only verified pairs do.
    """
    work = df.reset_index(drop=True)
    if sample and sample < len(work):
        work = work.sample(n=sample, random_state=seed).reset_index(drop=True)
        print(f"Fast near-dup audit on seeded sample of {len(work)} articles.")

    texts = (work["title"].fillna("").astype(str) + " " +
             work["text"].fillna("").astype(str)).tolist()

    print("Computing permutation signatures...")
    sigs, lengths = _signatures_for_docs(texts, num_perm, k=5)
    empty = lengths == 0
    print(f"Signatures done. Empty/short docs: {int(empty.sum())}")

    print("LSH banding for candidates...")
    candidate_pairs = _banded_candidates(sigs[~empty], rows_per_band)
    remap = np.flatnonzero(~empty)
    candidate_pairs = {(int(remap[i]), int(remap[j])) for i, j in candidate_pairs}
    print(f"Candidate pairs from {num_perm//rows_per_band} bands: {len(candidate_pairs)}")

    def jaccard_exact(i: int, j: int) -> float:
        a = set(_doc_shingles_uint64(texts[i]).tolist())
        b = set(_doc_shingles_uint64(texts[j]).tolist())
        union = len(a | b)
        return (len(a & b) / union) if union else 0.0

    verified_pairs = [(i, j) for i, j in sorted(candidate_pairs)
                      if jaccard_exact(i, j) >= threshold]
    print(f"Jaccard-verified pairs (>= {threshold}): {len(verified_pairs)}")

    parent = list(range(len(work)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in verified_pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    from collections import defaultdict
    members = defaultdict(list)
    for idx in range(len(work)):
        members[find(idx)].append(idx)

    root_to_cluster, next_id = {}, 0
    cluster_ids = np.full(len(work), -1, dtype=np.int64)
    multi = 0
    articles_in_clusters = 0
    for idx in range(len(work)):
        r = find(idx)
        grp = members[r]
        if len(grp) > 1:
            if r not in root_to_cluster:
                root_to_cluster[r] = next_id
                next_id += 1
            cluster_ids[idx] = root_to_cluster[r]
    multi = next_id
    articles_in_clusters = int((cluster_ids >= 0).sum())
    redundant = sum(len(members[r]) - 1 for r in set(find(i) for i in range(len(work)))
                    if len(members[find(i)]) > 1)

    report = {
        "articles_audited": len(work),
        "method": "vectorized_signatures+banded_lsh+exact_jaccard",
        "lsh_candidate_pairs": len(candidate_pairs),
        "verified_pairs": len(verified_pairs),
        "near_duplicate_clusters": multi,
        "articles_in_clusters": articles_in_clusters,
        "redundant_articles_if_keep_first": redundant,
        "threshold": threshold,
        "num_perm": num_perm,
        "rows_per_band": rows_per_band,
    }
    print(f"Near-dup clusters (>1 member): {multi} covering {articles_in_clusters} articles "
          f"({redundant} redundant).")

    work["near_duplicate_cluster_id"] = cluster_ids
    if persist_path:
        import os
        os.makedirs(os.path.dirname(persist_path) or ".", exist_ok=True)
        work[["near_duplicate_cluster_id"]].to_parquet(persist_path)
        pd.DataFrame(verified_pairs, columns=["a", "b"]).to_parquet(
            persist_path.replace(".parquet", "_pairs.parquet"), index=False)

    report["frame"] = work
    return report
