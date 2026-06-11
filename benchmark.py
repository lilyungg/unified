import time
import numpy as np
import polars as pl
from torch.utils.data import DataLoader

from ue       import (UnifiedEmbedding, NonMultiplexedEmbedding, CollisionlessEmbedding,
                      prehash, prehash_split, build_vocabs, preencode)
from models   import SimpleMLP, DCNV2LowRank
from data     import EmbDataset
from train    import train_model, evaluate


DATASET_CFG = {
    "movielens": {"emb_dim": 30, "emb_levels": 13_653},
    "avazu":     {"emb_dim": 32, "emb_levels": 26_542},
    "criteo":    {"emb_dim": 39, "emb_levels": 83_886},
}


def _make_loaders(data: np.ndarray, labels: np.ndarray,
                  tr, va, te, batch_size: int) -> tuple:
    kw = dict(num_workers=4, pin_memory=True)
    return (
        DataLoader(EmbDataset(data[tr], labels[tr]), batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(EmbDataset(data[va], labels[va]), batch_size=batch_size, shuffle=False, **kw),
        DataLoader(EmbDataset(data[te], labels[te]), batch_size=batch_size, shuffle=False, **kw),
    )


def run_dataset(
    name:       str,
    df:         pl.DataFrame,
    labels:     np.ndarray,
    tr, va, te,
    device,
    batch_size: int   = 65536,
    lr:         float = 1e-3,
    max_epochs: int   = 30,
    patience:   int   = 5,
    rank:       int   = 64,
) -> dict:
    cfg        = DATASET_CFG[name]
    emb_dim    = cfg["emb_dim"]
    emb_levels = cfg["emb_levels"]
    cols       = df.columns
    emb_out    = len(cols) * emb_dim

    vocabs = build_vocabs(df, cols)
    vs     = [len(vocabs[c]) for c in cols]
    nm_mod = NonMultiplexedEmbedding(vs, emb_levels, emb_dim)

    hash_data = np.concatenate(
        [prehash(df[c].to_numpy(), (0,), emb_levels, feature_id=c) for c in cols], axis=1)
    nm_data   = prehash_split(df, cols, nm_mod.levels)
    cl_data   = preencode(df, cols, vocabs)

    hl = _make_loaders(hash_data, labels, tr, va, te, batch_size)
    nl = _make_loaders(nm_data,   labels, tr, va, te, batch_size)
    cl = _make_loaders(cl_data,   labels, tr, va, te, batch_size)

    experiments = [
        ("Non-multiplex + MLP",
         SimpleMLP(NonMultiplexedEmbedding(vs, emb_levels, emb_dim), emb_out), nl),
        ("Non-multiplex + LR-DCN",
         DCNV2LowRank(NonMultiplexedEmbedding(vs, emb_levels, emb_dim), emb_out, rank=rank), nl),
        ("Multiplex + MLP",
         SimpleMLP(UnifiedEmbedding(emb_levels, emb_dim), emb_out), hl),
        ("Multiplex + LR-DCN",
         DCNV2LowRank(UnifiedEmbedding(emb_levels, emb_dim), emb_out, rank=rank), hl),
        ("Collisionless + MLP",
         SimpleMLP(CollisionlessEmbedding(vs, emb_dim), emb_out), cl),
        ("Collisionless + LR-DCN",
         DCNV2LowRank(CollisionlessEmbedding(vs, emb_dim), emb_out, rank=rank), cl),
    ]

    results = {}
    for exp_name, model, (tr_l, va_l, te_l) in experiments:
        t0    = time.time()
        model = train_model(model, tr_l, va_l, device,
                            lr=lr, max_epochs=max_epochs, patience=patience)
        results[exp_name] = {
            "auc":      round(evaluate(model, te_l, device), 4),
            "n_params": sum(p.numel() for p in model.parameters()),
            "time_sec": int(time.time() - t0),
        }

    return results
