import re
import warnings

import pandas as pd


def basic_arabic_clean(text: str) -> str:
    """Unicode-safe cleanup only. No Arabic character folding here."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return " ".join(text.split())


class ClassicalPreprocessor:
    """For TF-IDF / LSTM / BiGRU.

    Supports lowercase, optional stopword removal, optional ISRI stemming.
    Aggressive normalization stays opt-in via config, never default.
    """

    def __init__(self, lowercase: bool = False, remove_stopwords: bool = False,
                 stem: bool = False, stopwords_path: str = None):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self._stemmer = None
        self.stopwords = set()
        if remove_stopwords:
            import json
            import os
            path = stopwords_path or os.path.join("docs", "stopwords.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self.stopwords = set(data if isinstance(data, list) else data.get("stopwords", []))
            else:
                warnings.warn(f"Stopwords file not found at {path}; proceeding without stopword removal.")
                self.remove_stopwords = False
        if stem:
            try:
                from nltk.stem.isri import ISRIStemmer
                self._stemmer = ISRIStemmer()
            except ImportError:
                warnings.warn("nltk unavailable; ISRI stemming disabled.")
                self.stem = False

    def tokenize(self, text: str):
        tokens = basic_arabic_clean(text).split()
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        if self.stem and self._stemmer is not None:
            tokens = [self._stemmer.stem(t) for t in tokens]
        return tokens

    def __call__(self, text: str) -> str:
        return " ".join(self.tokenize(text))


class TransformerPreprocessor:
    """For AraBERT / CAMeLBERT: conservative cleanup + model-specific official
    preprocessing + native pretrained tokenization (applied later in the Dataset).

    No ISRI stemming. No stopword removal. No Arabic character folding unless
    the model's own official preprocessor does it (AraBERTv0.2 does not).
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.arabert_prep = None
        if model_name == "aubmindlab/bert-base-arabertv02":
            try:
                from arabert.preprocess import ArabertPreprocessor
                self.arabert_prep = ArabertPreprocessor(model_name=model_name)
            except Exception as e:
                warnings.warn(f"arabert package unavailable ({e}); using basic cleanup fallback.")

    def __call__(self, text: str) -> str:
        text = basic_arabic_clean(text)
        if self.arabert_prep:
            return self.arabert_prep.preprocess(text)
        return text


def build_preprocessor(mode: str, model_name: str = None, **kwargs):
    if mode == "classical":
        return ClassicalPreprocessor(**kwargs)
    if mode == "transformer":
        return TransformerPreprocessor(model_name=model_name)
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def preprocess_frame(df: pd.DataFrame, mode: str, model_name: str = None,
                     include_title: bool = True, **kwargs) -> pd.Series:
    prep = build_preprocessor(mode, model_name, **kwargs)
    title = df["title"].fillna("").astype(str) + " "
    body = df["text"].fillna("").astype(str)
    raw = (title + body) if include_title else body
    return raw.map(prep)


class TextPreprocessor(TransformerPreprocessor):
    """Backward-compatible alias for older notebooks."""
