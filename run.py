import argparse
import datetime
import json
import pathlib
import torch

from data       import load_movielens, load_avazu, load_criteo
from benchmark import run_dataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ml1m",     required=True)
    p.add_argument("--avazu",    default=None)
    p.add_argument("--skip",     nargs="*", default=[], choices=["movielens", "avazu", "criteo"])
    p.add_argument("--epochs",   type=int,   default=30)
    p.add_argument("--patience", type=int,   default=5)
    p.add_argument("--batch",    type=int,   default=65536)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--rank",     type=int,   default=64)
    p.add_argument("--fast",     action="store_true")
    p.add_argument("--out",      default="experiment_logs")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = get_device()
    n_rows  = 1_000_000 if args.fast else None
    run_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_kw = dict(
        device     = device,
        batch_size = args.batch,
        lr         = args.lr,
        max_epochs = args.epochs,
        patience   = args.patience,
        rank       = args.rank,
    )

    cfg = {
        "fast": args.fast, "n_rows": n_rows,
        "epochs": args.epochs, "patience": args.patience,
        "batch": args.batch, "lr": args.lr, "rank": args.rank,
    }

    all_results = {}

    if "movielens" not in args.skip:
        df, labels, tr, va, te = load_movielens(args.ml1m)
        all_results["movielens"] = run_dataset("movielens", df, labels, tr, va, te, **run_kw)
        (out_dir / f"{run_ts}_movielens.json").write_text(
            json.dumps({"config": cfg, "results": all_results["movielens"]}, indent=2))

    if "avazu" not in args.skip:
        df, labels, tr, va, te = load_avazu(path=args.avazu, n_rows=n_rows)
        all_results["avazu"] = run_dataset("avazu", df, labels, tr, va, te, **run_kw)
        (out_dir / f"{run_ts}_avazu.json").write_text(
            json.dumps({"config": cfg, "results": all_results["avazu"]}, indent=2))

    if "criteo" not in args.skip:
        df, labels, tr, va, te = load_criteo(n_rows=n_rows)
        all_results["criteo"] = run_dataset("criteo", df, labels, tr, va, te, **run_kw)
        (out_dir / f"{run_ts}_criteo.json").write_text(
            json.dumps({"config": cfg, "results": all_results["criteo"]}, indent=2))

    (out_dir / f"{run_ts}_summary.json").write_text(
        json.dumps({"config": cfg, "results": all_results}, indent=2))


if __name__ == "__main__":
    main()
