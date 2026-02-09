# ANN 数据集通用转换 + Benchmark Harness（DiskANN-rs）

这个目录是一个**通用 harness**，用于：
- 把 ann-benchmarks 风格的 `.hdf5` 数据集转成 DiskANN-rs `diskann-benchmark` 可读取的 `train.fbin / test.fbin / neighbors.ibin`
- 生成 **PQ vs Spherical Quantization（Extended RaBitQ 路线）** 的 benchmark 配置
- 运行 benchmark，并汇总 recall/QPS

说明：DiskANN-rs 的 `diskann-benchmark` **不直接读取 HDF5**，所以必须先转成 `.fbin/.ibin`。

## 一条命令跑完（推荐）

`scripts/run_dataset.py` 会在 `runs/<dataset>/` 下生成：
- `data/`：`train.fbin/test.fbin/neighbors.ibin`
- `configs/`：`pq-vs-spherical.json`
- `outputs/`：`output.json` + `summary.tsv`

示例（以 DBPedia OpenAI 1000K angular 为例）：

```bash
/mnt/nvme2n1p1/xtang/diskann-workspace/.venv/bin/python \
  DiskANN-playground/extend-rabitq/ann-harness/scripts/run_dataset.py \
  --hdf5 /mnt/nvme2n1p1/xtang/ann-data/dbpedia-openai-1000k-angular.hdf5 \
  --dataset dbpedia-openai-1000k-angular \
  --distance cosine \
  --search-n 100 \
  --search-l 100,200,400,800
```

如果你的 HDF5 key 不是默认的 `train/test/neighbors`，可以指定：

```bash
.../run_dataset.py --hdf5 your.hdf5 --train-key train --test-key test --neighbors-key neighbors
```

## 分步运行（需要时）

### 1) HDF5 -> DiskANN bin

```bash
/mnt/nvme2n1p1/xtang/diskann-workspace/.venv/bin/python \
  DiskANN-playground/extend-rabitq/ann-harness/scripts/convert_hdf5_to_diskann_bin.py \
  --hdf5 /path/to/dataset.hdf5 \
  --out-dir /path/to/out/data
```

### 2) 生成配置（PQ vs Spherical）

```bash
/mnt/nvme2n1p1/xtang/diskann-workspace/.venv/bin/python \
  DiskANN-playground/extend-rabitq/ann-harness/scripts/make_pq_vs_extended_rabitq_config.py \
  --data-dir /path/to/out/data \
  --out /path/to/out/configs/pq-vs-spherical.json
```

### 3) 运行 benchmark

```bash
DiskANN-playground/extend-rabitq/ann-harness/scripts/run_benchmark.sh \
  /path/to/out/configs/pq-vs-spherical.json \
  /path/to/out/outputs/output.json
```

### 4) 汇总输出

```bash
/mnt/nvme2n1p1/xtang/diskann-workspace/.venv/bin/python \
  DiskANN-playground/extend-rabitq/ann-harness/scripts/summarize_output.py \
  --input /path/to/out/outputs/output.json
```

## 距离度量注意事项

- 数据集名字带 `angular` 时，常用 `cosine`。
- DiskANN-rs 的 Spherical Quantization **不支持** `cosine_normalized`。
  - `make_pq_vs_extended_rabitq_config.py` / `run_dataset.py` 会对 spherical job 自动降级到 `cosine`，避免报错。
