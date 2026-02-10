#!/usr/bin/env bash
set -euo pipefail

HDF5_PATH="${HDF5_PATH:-/mnt/nvme2n1p1/xtang/ann-data/dbpedia-openai-1000k-angular.hdf5}"
DATASET="${DATASET:-dbpedia-openai-1000k-angular}"
DISTANCE="${DISTANCE:-cosine}"
SEARCH_N="${SEARCH_N:-100}"
SEARCH_L="${SEARCH_L:-100,200,400,800}"
REBUILD_INDEX="${REBUILD_INDEX:-false}"
EMON_ENABLE="${EMON_ENABLE:-false}"

python run_dataset.py \
  --hdf5 "$HDF5_PATH" \
  --dataset "$DATASET" \
  --distance "$DISTANCE" \
  --search-n "$SEARCH_N" \
  --search-l "$SEARCH_L" \
  --rebuild-index "$REBUILD_INDEX" \
  --emon-enable "$EMON_ENABLE"