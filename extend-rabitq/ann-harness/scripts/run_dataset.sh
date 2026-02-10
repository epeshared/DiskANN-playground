#!/usr/bin/env bash
set -euo pipefail

python run_dataset.py \
  --hdf5 /mnt/nvme2n1p1/xtang/ann-data/dbpedia-openai-1000k-angular.hdf5  \
  --dataset dbpedia-openai-1000k-angular \
  --distance cosine \
  --search-n 100 \
  --search-l 100,200,400,800 \
  --rebuild-index false \
  --emon-enable false