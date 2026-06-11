import copy
import datetime
import json
import pathlib
import time
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from ue import (UnifiedEmbedding, CollisionlessEmbedding, NonMultiplexedEmbedding,
                prehash, prehash_split, build_vocabs, preencode)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE   = 2048
LR           = 1e-3
EPOCHS       = 10
ML1M_PATH    = pathlib.Path("/Users/m.filatov/recsys_research/ml-1m")
AVAZU_PATH   = pathlib.Path("/Users/m.filatov/recsys_research/avazu-ctr-prediction/train.gz")
AVAZU_NROWS  = 1_000_000
CRITEO_NROWS = 1_000_000
LOG_DIR      = pathlib.Path("experiment_logs")

DATASET_CFG = {
    "movielens": {"emb_dim": 30, "emb_levels": 13_653},
    "avazu":     {"emb_dim": 32, "emb_levels": 26_542},
    "criteo":    {"emb_dim": 39, "emb_levels": 83_886},
}


def load_movielens(path: pathlib.Path):
    user_info = {}
    with open(path / "users.dat") as f:
        for line in f:
            uid, gender, age, occ, zip_ = line.strip().split("::")
            user_info[uid] = (gender, age, occ, zip_)
    rows = []
    with open(path / "ratings.dat") as f:
        for line in f:
            uid, mid, rating, _ = line.strip().split("::")
            gender, age, occ, zip_ = user_info[uid]
            rows.append({"user_id": uid, "movie_id": mid, "gender": gender,
                         "age": age, "occupation": occ, "zip": zip_,
                         "label": int(int(rating) >= 3)})
    df = pl.DataFrame(rows)
    labels = df["label"].to_numpy().astype(np.float32)
    return _split(df.select(["user_id", "movie_id", "gender", "age", "occupation", "zip"]), labels)


def load_avazu(path: pathlib.Path, n_rows: int = AVAZU_NROWS):
    df = pl.read_csv(path, n_rows=n_rows, infer_schema_length=10_000,
                     schema_overrides={"id": pl.Utf8})
    labels = df["click"].cast(pl.Float32).to_numpy()
    cat_cols = [c for c in df.columns if c not in ("id", "click")]
    return _split(df.select(cat_cols).with_columns(pl.all().cast(pl.Utf8)), labels)


def load_criteo(n_rows: int = CRITEO_NROWS):
    from datasets import load_dataset
    ds = load_dataset("reczoo/Criteo_x4", split="train").select(range(n_rows))
    df = pl.DataFrame(ds.to_dict())
    labels = df["Label"].cast(pl.Float32).to_numpy()
    cat_cols = [f"C{i}" for i in range(1, 27)]
    return _split(df.select(cat_cols).with_columns(pl.all().cast(pl.Utf8).fill_null("nan")), labels)


def _split(df: pl.DataFrame, labels: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    t, v = int(0.8 * len(df)), int(0.9 * len(df))
    return df, labels, idx[:t], idx[t:v], idx[v:]


class EmbDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).long()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class SimpleNN(nn.Module):
    def __init__(self, emb_module: nn.Module, emb_out_dim: int):
        super().__init__()
        self.emb = emb_module
        self.mlp = nn.Sequential(
            nn.Linear(emb_out_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),          nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.mlp(self.emb(x)).squeeze(1)


class DCNV2(nn.Module):
    def __init__(self, emb_module: nn.Module, emb_out_dim: int,
                 num_cross: int = 2, dnn_dims: tuple = (128, 64),
                 dropout: float = 0.1, reg_weight: float = 1e-5):
        super().__init__()
        dim = emb_out_dim
        self.reg_weight = reg_weight
        self.emb = emb_module
        self.cross_w = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, dim)) for _ in range(num_cross)])
        self.cross_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(num_cross)])
        dnn_layers, in_d = [], dim
        for out_d in dnn_dims:
            dnn_layers += [nn.Dropout(dropout), nn.Linear(in_d, out_d),
                           nn.BatchNorm1d(out_d), nn.ReLU()]
            in_d = out_d
        self.dnn    = nn.Sequential(*dnn_layers)
        self.output = nn.Linear(in_d, 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.cross_w: nn.init.xavier_normal_(p)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def cross_network(self, x0):
        x = x0
        for W, b in zip(self.cross_w, self.cross_b):
            x = x0 * (x @ W.T + b) + x
        return x

    def forward(self, x):
        e = self.emb(x)
        return self.output(self.dnn(self.cross_network(e))).squeeze(1)

    def reg_loss(self):
        return self.reg_weight * sum(W.norm(2) for W in self.cross_w)


def evaluate(model, loader) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds.append(torch.sigmoid(model(x)).cpu().numpy())
            targets.append(y.cpu().numpy())
    return roc_auc_score(np.concatenate(targets), np.concatenate(preds))


def train_model(model, train_loader, val_loader) -> tuple:
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    has_reg   = hasattr(model, "reg_loss")

    best_auc, best_state = 0.0, None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if has_reg:
                loss = loss + model.reg_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_auc = evaluate(model, val_loader)
        history.append(round(val_auc, 4))

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_auc, history


def _save_dataset_log(run_ts: str, dataset_name: str, cfg: dict, results: dict):
    LOG_DIR.mkdir(exist_ok=True)
    payload = {
        "dataset":     dataset_name,
        "timestamp":   run_ts,
        "config":      {**cfg, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "experiments": results,
    }
    (LOG_DIR / f"{run_ts}_{dataset_name}.json").write_text(json.dumps(payload, indent=2))


def _save_summary(run_ts: str, all_results: dict):
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / f"{run_ts}_summary.json").write_text(
        json.dumps({"timestamp": run_ts, "results": all_results}, indent=2))


def run_comparison(df: pl.DataFrame, labels: np.ndarray,
                   train_idx, val_idx, test_idx,
                   emb_dim: int, emb_levels: int) -> dict:
    cols        = df.columns
    emb_out_dim = len(cols) * emb_dim

    vocabs      = build_vocabs(df, cols)
    vocab_sizes = [len(vocabs[c]) for c in cols]

    hash_data = np.concatenate(
        [prehash(df[c].to_numpy(), (0,), emb_levels, feature_id=c) for c in cols], axis=1)
    cl_data   = preencode(df, cols, vocabs)
    nm_module = NonMultiplexedEmbedding(vocab_sizes, emb_levels, emb_dim)
    nm_data   = prehash_split(df, cols, nm_module.levels)

    def loaders(data):
        return (
            DataLoader(EmbDataset(data[train_idx], labels[train_idx]),
                       batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
            DataLoader(EmbDataset(data[val_idx],   labels[val_idx]),   batch_size=BATCH_SIZE),
            DataLoader(EmbDataset(data[test_idx],  labels[test_idx]),  batch_size=BATCH_SIZE),
        )

    hl, cl, nl = loaders(hash_data), loaders(cl_data), loaders(nm_data)

    experiments = [
        ("Non-multiplex + MLP",
         SimpleNN(NonMultiplexedEmbedding(vocab_sizes, emb_levels, emb_dim), emb_out_dim), nl),
        ("Non-multiplex + DCN",
         DCNV2(NonMultiplexedEmbedding(vocab_sizes, emb_levels, emb_dim), emb_out_dim),    nl),
        ("Multiplex + MLP",
         SimpleNN(UnifiedEmbedding(emb_levels, emb_dim), emb_out_dim),                     hl),
        ("Multiplex + DCN",
         DCNV2(UnifiedEmbedding(emb_levels, emb_dim), emb_out_dim),                        hl),
        ("Collisionless + MLP",
         SimpleNN(CollisionlessEmbedding(vocab_sizes, emb_dim), emb_out_dim),               cl),
        ("Collisionless + DCN",
         DCNV2(CollisionlessEmbedding(vocab_sizes, emb_dim), emb_out_dim),                  cl),
    ]

    results = {}
    for name, model, (tr, va, te) in experiments:
        t0 = time.time()
        best_val_auc, history = train_model(model, tr, va)
        test_auc = evaluate(model, te)
        results[name] = {
            "test_auc":        round(test_auc, 4),
            "best_val_auc":    round(best_val_auc, 4),
            "val_auc_history": history,
            "n_params":        sum(p.numel() for p in model.parameters()),
            "time_sec":        int(time.time() - t0),
        }

    return results


if __name__ == "__main__":
    run_ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    cfg = DATASET_CFG["movielens"]
    df, labels, tr, va, te = load_movielens(ML1M_PATH)
    all_results["movielens"] = run_comparison(df, labels, tr, va, te, cfg["emb_dim"], cfg["emb_levels"])
    _save_dataset_log(run_ts, "movielens", cfg, all_results["movielens"])

    cfg = DATASET_CFG["avazu"]
    df, labels, tr, va, te = load_avazu(AVAZU_PATH)
    all_results["avazu"] = run_comparison(df, labels, tr, va, te, cfg["emb_dim"], cfg["emb_levels"])
    _save_dataset_log(run_ts, "avazu", cfg, all_results["avazu"])

    cfg = DATASET_CFG["criteo"]
    df, labels, tr, va, te = load_criteo()
    all_results["criteo"] = run_comparison(df, labels, tr, va, te, cfg["emb_dim"], cfg["emb_levels"])
    _save_dataset_log(run_ts, "criteo", cfg, all_results["criteo"])

    _save_summary(run_ts, all_results)
