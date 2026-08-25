import re
import unicodedata

import pandas as pd

_HTML_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPEATED_PUNCT_RE = re.compile(r"([^\w\s])\1+")


def conservative_clean(text: str) -> str:
    """Unicode normalization, HTML stripping, control-char removal, whitespace collapse.

    Deliberately does NOT fold Arabic characters (ة→ه, ى→ي, أ→ا):
    character-level normalization is delegated to model-specific preprocessing.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = _HTML_RE.sub(" ", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return " ".join(text.split())


def html_density(text: str) -> float:
    text = str(text)
    if not text:
        return 0.0
    html_len = sum(len(m.group()) for m in _HTML_RE.finditer(text))
    return html_len / len(text)


def filter_junk(df: pd.DataFrame, min_words: int = 10, max_html_density: float = 0.3) -> pd.DataFrame:
    initial_len = len(df)
    df = df[df["text"].fillna("").str.strip().astype(bool)]
    df = df[df["text"].apply(lambda x: len(str(x).split()) >= min_words)]
    df = df[df["text"].apply(html_density) <= max_html_density]
    dropped = initial_len - len(df)
    print(f"Junk filter: dropped {dropped} articles ({dropped / max(1, initial_len):.2%}).")
    return df


def apply_conservative_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].map(conservative_clean)
    df["text"] = df["text"].map(conservative_clean)
    return df


def prepare_corpus(df: pd.DataFrame, **filter_kwargs) -> pd.DataFrame:
    df = filter_junk(df, **filter_kwargs)
    df = apply_conservative_cleaning(df)
    return df.reset_index(drop=True)
