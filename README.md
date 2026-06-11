# unified-embeddings

Benchmark comparing three categorical embedding strategies for CTR prediction:

- Non-multiplex: per-feature hash tables, budget split proportional to cardinality
- Multiplex: one shared table, feature-ID salting to separate features
- Collisionless: per-feature tables sized to exact vocabulary (upper bound)

Each is paired with two architectures: a plain MLP and a low-rank DCN-V2. Expected ordering by AUC: Non-multiplex < Multiplex < Collisionless.

Based on the Feature Multiplexing paper (https://arxiv.org/abs/2305.12102).


## Setup

```
pip install -r requirements.txt
```


## Data setup (one time)

MovieLens-1M — download and unzip:

```
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

Avazu and Criteo are pulled from HuggingFace automatically on first run and written to
datasets/avazu.parquet and datasets/criteo.parquet. All subsequent runs load from those files.
To pre-cache them without running a full experiment:

```
python -c "from datasets import load_movielens, load_avazu, load_criteo; load_avazu(); load_criteo()"
```

If you have a local Avazu gz file, pass it with --avazu to skip the download entirely.


## Running experiments

Results are saved to experiment_logs/ by default (one JSON per dataset + a summary).
Pass --out to change the directory.

### Sampled run (1M random rows from Avazu and Criteo, good for CPU / quick iteration)

```
python run.py --ml1m /path/to/ml-1m --fast
```

### Full run (complete datasets: MovieLens ~1M, Avazu ~36M, Criteo ~45M)

```
python run.py --ml1m /path/to/ml-1m
```

### Single dataset

```
python run.py --ml1m /path/to/ml-1m --skip avazu criteo    # MovieLens only
python run.py --ml1m /path/to/ml-1m --skip movielens criteo # Avazu only
python run.py --ml1m /path/to/ml-1m --skip movielens avazu  # Criteo only
```

### All options

```
--avazu     path to local avazu train.gz (optional, falls back to HuggingFace)
--skip      datasets to skip: movielens avazu criteo
--fast      1M random sample from Avazu and Criteo instead of full data
--out       output directory, default experiment_logs/
--epochs    max epochs, default 30
--patience  early stopping patience, default 5
--batch     batch size, default 65536
--lr        learning rate, default 1e-3
--rank      low-rank dimension in LR-DCN, default 64
```

### test.py

Standalone script with hardcoded paths (edit ML1M_PATH and AVAZU_PATH at the top).
Runs all 6 experiments and saves logs to experiment_logs/.

```
python test.py
```


## Files

```
ue.py          embedding modules and prehash helpers
models.py      SimpleMLP and DCNV2LowRank
datasets.py    data loaders for MovieLens, Avazu, Criteo
train.py       training loop with early stopping
benchmark.py   runs 6 experiments for a single dataset
run.py         CLI entry point
test.py        standalone script with hardcoded paths
```
