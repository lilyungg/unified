import torch
import torch.nn as nn
import xxhash
import numpy as np


def prehash(values: np.ndarray, seeds: tuple, emb_levels: int,
            feature_id: str = "") -> np.ndarray:
    out = np.empty((len(values), len(seeds)), dtype=np.int64)
    prefix = feature_id + ":" if feature_id else ""
    for j, seed in enumerate(seeds):
        out[:, j] = np.vectorize(
            lambda v, s=seed: xxhash.xxh32(prefix + str(v), s).intdigest() % emb_levels
        )(values)
    return out


def build_vocabs(df, cols: list) -> dict:
    vocabs = {}
    for col in cols:
        vals = sorted(df[col].unique().to_list(), key=str)
        vocabs[col] = {str(v): i + 1 for i, v in enumerate(vals)}
    return vocabs


def preencode(df, cols: list, vocabs: dict) -> np.ndarray:
    parts = []
    for col in cols:
        vocab = vocabs[col]
        encoded = np.array(
            [vocab.get(str(v), 0) for v in df[col].to_numpy()], dtype=np.int64
        )
        parts.append(encoded.reshape(-1, 1))
    return np.concatenate(parts, axis=1)


def prehash_split(df, cols: list, levels: list) -> np.ndarray:
    parts = []
    for col, lev in zip(cols, levels):
        hashed = np.vectorize(
            lambda v: xxhash.xxh32(str(v), 0).intdigest() % lev
        )(df[col].to_numpy())
        parts.append(hashed.reshape(-1, 1))
    return np.concatenate(parts, axis=1)


class CollisionlessEmbedding(nn.Module):
    def __init__(self, vocab_sizes: list, emb_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vs + 1, emb_dim) for vs in vocab_sizes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=1)


class NonMultiplexedEmbedding(nn.Module):
    def __init__(self, vocab_sizes: list, total_emb_levels: int, emb_dim: int):
        super().__init__()
        total_vocab = sum(vocab_sizes)
        self.levels = [
            max(1, round(vs / total_vocab * total_emb_levels)) for vs in vocab_sizes
        ]
        self.tables = nn.ModuleList(
            [nn.Embedding(lev, emb_dim) for lev in self.levels]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [tbl(x[:, i]) for i, tbl in enumerate(self.tables)]
        return torch.cat(parts, dim=1)


class UnifiedEmbedding(nn.Module):
    def __init__(self, emb_levels: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(emb_levels, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x).reshape(x.size(0), -1)
