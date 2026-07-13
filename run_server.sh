#!/usr/bin/env bash
# Full Avazu/Criteo pipeline on a GPU server (DCN only, paper Table 1 columns).
# Run from the repo root. Edit BATCH/RUNS/WORKERS for your box.
# Smoke-test first:  bash run_server.sh smoke
set -euo pipefail

PY=.venv/bin/python
BATCH=4096          # big batch: fewer steps; DCN table is tiny, fits any GPU
RUNS=1              # bump to 5 for mean±std once numbers look right
WORKERS=8           # DataLoader workers; raise on many-core servers

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv --python 3.12 .venv
  uv pip install --python $PY -r requirements.txt
  $PY -c "import torch; print('cuda:', torch.cuda.is_available())"
}

download() {
  mkdir -p raw
  # Criteo Kaggle Display Advertising Challenge -> raw/train.txt
  [ -f raw/train.txt ] || {
    curl -L -o raw/criteo.tar.gz \
      https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
    tar xzf raw/criteo.tar.gz -C raw/
  }
  # Avazu needs Kaggle auth (~/.kaggle/kaggle.json + accepted rules):
  #   uv pip install --python .venv/bin/python kaggle
  #   .venv/bin/kaggle competitions download -c avazu-ctr-prediction -f train.gz -p raw/
  echo "Avazu: fetch raw/train.gz via Kaggle CLI manually (see comment)."
}

prepare() {
  [ -f datasets/criteo_prepared.parquet ] || $PY prepare_data.py criteo --raw raw/train.txt
  [ -f datasets/avazu_prepared.parquet  ] || $PY prepare_data.py avazu  --raw raw/train.gz
}

smoke() {
  $PY run.py --skip movielens --fast --budgets 1.0 --epochs 2 \
    --batch $BATCH --workers $WORKERS
}

criteo() {   # columns 25 / 12.5 / 2.5 MB
  $PY run.py --skip movielens avazu --budgets 2.0 1.0 0.2 \
    --batch $BATCH --runs $RUNS --workers $WORKERS
}

avazu() {    # columns 32.4 / 3.24 / 0.324 MB
  $PY run.py --skip movielens criteo --budgets 10.0 1.0 0.1 \
    --batch $BATCH --runs $RUNS --workers $WORKERS
}

case "${1:-all}" in
  setup) setup ;;
  download) download ;;
  prepare) prepare ;;
  smoke) smoke ;;
  criteo) criteo ;;
  avazu) avazu ;;
  all) prepare; criteo; avazu ;;
  *) echo "usage: bash run_server.sh {setup|download|prepare|smoke|criteo|avazu|all}"; exit 1 ;;
esac
