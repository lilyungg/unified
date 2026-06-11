import argparse
import datetime
import json
import pathlib
import torch

from datasets  import load_movielens, load_avazu, load_criteo
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
    p.add_argument("--out",      default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    device  = get_device()
    n_rows  = 1_000_000 if args.fast else None
    run_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out     = pathlib.Path(args.out) if args.out else pathlib.Path(f"results_{run_ts}.json")

    run_kw = dict(
        device     = device,
        batch_size = args.batch,
        lr         = args.lr,
        max_epochs = args.epochs,
        patience   = args.patience,
        rank       = args.rank,
    )

    all_results = {}

    if "movielens" not in args.skip:
        df, labels, tr, va, te = load_movielens(args.ml1m)
        all_results["movielens"] = run_dataset("movielens", df, labels, tr, va, te, **run_kw)

    if "avazu" not in args.skip:
        df, labels, tr, va, te = load_avazu(path=args.avazu, n_rows=n_rows)
        all_results["avazu"] = run_dataset("avazu", df, labels, tr, va, te, **run_kw)

    if "criteo" not in args.skip:
        df, labels, tr, va, te = load_criteo(n_rows=n_rows)
        all_results["criteo"] = run_dataset("criteo", df, labels, tr, va, te, **run_kw)

    out.write_text(json.dumps({"config": vars(args), "results": all_results}, indent=2))


if __name__ == "__main__":
    main()
