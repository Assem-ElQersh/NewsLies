import torch
from torch.utils.data import Dataset

LABEL_ORDER = ["credible", "not credible", "undecided"]


def build_label_map(labels) -> dict:
    """Deterministic label→id mapping independent of row order."""
    unique = sorted(set(str(l) for l in labels), key=LABEL_ORDER.index)
    return {l: i for i, l in enumerate(unique)}


def map_labels(df):
    lm = build_label_map(df["label"])
    df = df.copy()
    df["label_id"] = df["label"].map(lm)
    return df, {v: k for k, v in lm.items()}


class ClassicalDataset(Dataset):
    """Dataset for LSTM and BiGRU baselines."""

    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx]).split()
        token_ids = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in text[:self.max_len]]
        if len(token_ids) < self.max_len:
            token_ids += [0] * (self.max_len - len(token_ids))
        return (torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long))


class TransformerDataset(Dataset):
    """Dataset for AraBERT / CAMeLBERT / hierarchical fine-tuning."""

    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts.iloc[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long),
        }


def build_vocab(texts, vocab_size: int = 30000, seed: int = 42, sample_n: int = 100000):
    from collections import Counter

    if sample_n and len(texts) > sample_n:
        texts = texts.sample(n=sample_n, random_state=seed)
    words = " ".join(texts.astype(str)).split()
    most_common = Counter(words).most_common(vocab_size - 2)
    vocab = {w: i + 2 for i, (w, _) in enumerate(most_common)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab
