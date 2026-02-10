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
- `outputs/`：`output.json` + `summary.tsv` + `details.md`

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

## 结果文件定位与解读

假设运行时指定：`--dataset my-dataset`，则输出目录固定为：

`DiskANN-playground/extend-rabitq/ann-harness/runs/my-dataset/`

其中关键文件：

- `outputs/output.json`
  - diskann-benchmark 的**原始完整输出**（build + search 的结构化结果），后处理都从这里读取
- `outputs/summary.tsv`
  - 用于画 recall–QPS/latency 曲线的**汇总表**（建议优先看这个）
- `outputs/details.md`
  - 从 `output.json` 抽取的更“底层”的解释报告（build 时间、training_time、quantized_bytes、每个 layout 的 run 结构等）

### summary.tsv 字段说明

每行对应“一个 job（或 spherical 的某个 layout）在某个 `L` 下”的统计汇总。

- `job`
  - `async-index-build-pq`：PQ
  - `async-index-build-spherical-quantization`：Spherical Quantization（对标 Extended RaBitQ 路线）
- `detail`
  - PQ：`pq_chunks=...`
  - Spherical：`num_bits=...; layouts=[...]; layout=...`
- `tasks`
  - search 的线程数（来自 config 的 `search_phase.num_threads`）
- `N`
  - `search-n`：每个 query 返回的 top-N
- `L`
  - `search-l`：搜索候选预算（L 越大通常 recall 更高，但更慢）
- `recall(avg)`：平均 recall
- `QPS(mean)` / `QPS(max)`：每秒查询数，对多次 `reps` 做汇总

Latency（单位：微秒 us）：

- `lat_mean_us(mean/max)`
  - 单条 query 的平均耗时（mean latency），对多次 `reps` 取 mean/max
- `lat_p99_us(mean/max)`
  - 单条 query 的 p99 尾延迟，对多次 `reps` 取 mean/max

注意：这里的 `lat_*` 是**单条 query 的延迟分布**，不是“跑完整个 `test.fbin` 的总时间”。

### spherical 的 layout 是什么

当 job 是 `async-index-build-spherical-quantization` 时，你会在 `summary.tsv` 里看到同一个 `L` 对应多行，并在 `detail` 里出现：`layout=...`。

这些 `layout` 表示 **query（查询向量）在搜索阶段使用的表示形式/内存布局**（同一个索引下的不同查询表示），用于做速度/精度的折中：

- `layout=full_precision`
  - query 使用原始 `float32` 全精度参与计算；通常作为“精度上限/基线”对照
- `layout=same_as_data`
  - query 使用与数据向量相同的量化/布局路径参与计算
- `layout=scalar_quantized`
  - query 使用标量量化（scalar-quantized）的表示参与计算；通常吞吐更高（QPS 更大）

一般做法：优先看 `scalar_quantized` 的曲线；若需要全精度对照或排查数值问题，再看 `full_precision`。

### details.md 看什么

`details.md` 会把 `output.json` 里关键字段展开成更好读的格式：

- build：`vectors_inserted`、`total_time_us`（并换算成秒）、`insert_latency_us`（mean/median/p90/p99）
- PQ：`pq_quant_training_time_us`
- Spherical：`training_time_us`、`quantized_bytes_per_vector`、以及按 `layout` 展开的 `runs` 结果

### output.json（需要做深分析时）

`outputs/output.json` 里包含更底层的原始数组（每次 rep 的观测值），例如：

- `qps`：长度 = `reps`
- `mean_latencies` / `p99_latencies`：长度 = `reps`，单位 us

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
