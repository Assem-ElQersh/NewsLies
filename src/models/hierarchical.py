import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedBodyEncoder(nn.Module):
    """Splits tokenized body into overlapping windows, encodes each chunk with a
    FROZEN transformer, and pools chunk embeddings with trainable attention."""

    def __init__(self, hidden_size: int = 768, chunk_len: int = 512, stride: int = 256):
        super().__init__()
        self.chunk_len = chunk_len
        self.stride = stride
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def split_ids(self, input_ids, attention_mask):
        """input_ids: [B, L] -> padded chunks [total_chunks, chunk_len], counts."""
        n, s = self.chunk_len, self.stride
        chunks_ids, chunks_mask, counts = [], [], []
        for ids, mask in zip(input_ids, attention_mask):
            starts = list(range(0, max(1, int(mask.sum().item())), s))
            kept = []
            for st in starts:
                seg_ids = ids[st:st + n]
                if len(seg_ids) == 0:
                    continue
                kept.append((seg_ids, mask[st:st + n]))
                if len(seg_ids) < n:
                    break
            if not kept:
                kept = [(ids[:n], mask[:n])]
            counts.append(len(kept))
            for cids, cmask in kept:
                chunks_ids.append(cids)
                chunks_mask.append(cmask)
        max_len = max(len(c) for c in chunks_ids)
        padded_ids = torch.full((len(chunks_ids), max_len), 0, dtype=chunks_ids[0].dtype)
        padded_mask = torch.zeros_like(padded_ids)
        for i, (c, m) in enumerate(zip(chunks_ids, chunks_mask)):
            padded_ids[i, :len(c)] = c
            padded_mask[i, :len(m)] = m
        return padded_ids.to(input_ids.device), padded_mask.to(attention_mask.device), counts

    def pool_from_groups(self, chunk_embeddings, group_slices):
        """group_slices: list of (start, end) ranges per document."""
        pooled = []
        attn_scores = self.attention(chunk_embeddings).squeeze(-1)
        for start, end in group_slices:
            scores = attn_scores[start:end]
            weights = F.softmax(scores, dim=0)
            emb = chunk_embeddings[start:end]
            pooled.append((weights.unsqueeze(1) * emb).sum(dim=0))
        return torch.stack(pooled)


class HierarchicalClassifier(nn.Module):
    """H1: frozen title encoder + frozen chunked body encoder + cross-chunk
    attention pooling + classifier head. Attention and head are trainable."""

    def __init__(self, encoder_name_or_model, num_classes: int = 3,
                 chunk_len: int = 512, stride: int = 256, dropout: float = 0.1,
                 freeze_encoder: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name_or_model)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        hidden = self.encoder.config.hidden_size
        self.chunker = ChunkedBodyEncoder(hidden_size=hidden, chunk_len=chunk_len, stride=stride)
        self.title_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def _encode_passages(self, input_ids, attention_mask, batch_size: int = 16):
        embs = []
        ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            for i in range(0, input_ids.size(0), batch_size):
                out = self.encoder(input_ids=input_ids[i:i + batch_size],
                                   attention_mask=attention_mask[i:i + batch_size])
                embs.append(out.last_hidden_state[:, 0, :].float().cpu())
        return torch.cat(embs, dim=0)

    def forward(self, title_ids=None, title_mask=None, body_ids=None, body_mask=None,
                cached_body_embs=None, group_slices=None):
        if title_ids is None:
            raise ValueError("title_ids required")
        ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            t_out = self.encoder(input_ids=title_ids, attention_mask=title_mask)
        title_emb = t_out.last_hidden_state[:, 0, :]

        if cached_body_embs is not None:
            body_doc = self.chunker.pool_from_groups(
                cached_body_embs.to(title_emb.device), group_slices)
        elif body_ids is not None:
            chunk_ids, chunk_mask, counts = self.chunker.split_ids(body_ids, body_mask)
            all_chunk_embs = self._encode_passages(chunk_ids, chunk_mask)
            slices, pos = [], 0
            for c in counts:
                slices.append((pos, pos + c))
                pos += c
            body_doc = self.chunker.pool_from_groups(
                all_chunk_embs.to(title_emb.device), slices)
        else:
            raise ValueError("body_ids or cached_body_embs required")

        fused = torch.cat([self.dropout(self.title_proj(title_emb)), self.dropout(body_doc)], dim=1)
        return self.classifier(fused)


@torch.no_grad()
def cache_body_embeddings(encoder, tokenizer, texts, chunk_len: int = 512, stride: int = 256,
                          encode_batch: int = 16, device="cuda", progress_every: int = 500,
                          max_input_tokens: int = 2048):
    """Pre-compute frozen chunk embeddings for a corpus. Returns (tensor, slices)."""
    chunker = ChunkedBodyEncoder(chunk_len=chunk_len, stride=stride)
    encoder.eval()
    all_embs, slices = [], []
    pos = 0
    for n_done, text in enumerate(texts):
        enc = tokenizer(str(text), truncation=True, max_length=max_input_tokens,
                        return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        cids, cmask, _ = chunker.split_ids(ids.unsqueeze(0).to(device),
                                           mask.unsqueeze(0).to(device))
        embs = []
        for i in range(0, cids.size(0), encode_batch):
            out = encoder(input_ids=cids[i:i + encode_batch],
                          attention_mask=cmask[i:i + encode_batch])
            embs.append(out.last_hidden_state[:, 0, :].float().cpu())
        block = torch.cat(embs, dim=0)
        all_embs.append(block)
        slices.append((pos, pos + block.size(0)))
        pos += block.size(0)
        if (n_done + 1) % progress_every == 0:
            print(f"Cached embeddings {n_done + 1}/{len(texts)}")
    return torch.cat(all_embs, dim=0), slices
