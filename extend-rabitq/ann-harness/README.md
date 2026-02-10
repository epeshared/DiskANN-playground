# ANN 数据集通用转换 + Benchmark Harness（DiskANN-rs）

这个目录是一个**通用 harness**，用于：
- 把 ann-benchmarks 风格的 `.hdf5` 数据集转成 DiskANN-rs `diskann-benchmark` 可读取的 `train.fbin / test.fbin / neighbors.ibin`
- 生成 **PQ vs Spherical Quantization（Extended RaBitQ 路线）** 的 benchmark 配置
- 运行 benchmark，并汇总 recall/QPS

说明：DiskANN-rs 的 `diskann-benchmark` **不直接读取 HDF5**，所以必须先转成 `.fbin/.ibin`。

## 一条命令跑完（推荐）

`scripts/run_dataset.py` 会在 `runs/<dataset>/<timestamp>/` 下生成（`timestamp` 为 UTC 时间戳）：
- `data/`：`train.fbin/test.fbin/neighbors.ibin`
- `configs/`：`pq-vs-spherical.json`（`--emon-enable` 时还会额外生成 `pq-vs-spherical.build.json` / `pq-vs-spherical.search.json`）
- `indices/`：（仅 `--emon-enable`）build 阶段产物，供后续 load+search 使用
- `outputs/`：
  - 默认：`output.json` + `summary.tsv` + `details.md`
  - `--emon-enable`：`output.build.json` / `output.search.json` + 对应的 `summary.*.tsv` / `details.*.md`（同时 `summary.tsv`/`details.md` 默认指向 search 结果）
- `server-info.txt`：本机 `lscpu`（以及 `uname -a`）输出，用于记录跑 benchmark 的机器信息
- （可选）`emon/<slice>/emon.dat`：如果传 `--emon-enable`，会把 **EMON 采样只包住 search 阶段**（build 不采样），并且会把搜索 sweep 拆开，做到更细粒度的采样文件：
  - PQ：按线程数 / N / L 拆分
  - Spherical：在上面基础上，再按 layout（`scalar_quantized` / `same_as_data` / `full_precision`）拆分
  因此最终会生成多份 `emon.dat`，每份只覆盖一个 slice 的一次 search invocation。
  1) 先跑 build-only（清空 search runs + 强制 save_path）
  2) `emon -collect-edp` 启动采样
  3) 再跑 load+search（从 `indices/` load）
  4) `emon -stop` 结束采样
  5) 对下一个 slice 重复 2)~4)

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

如果要采集 Intel EMON（需要你已经安装 SEP，并且 `emon` 在 PATH 里；常见做法是先 `source /opt/intel/sep/sep_vars.sh`）：

```bash
.../run_dataset.py --hdf5 your.hdf5 --dataset your-dataset --emon-enable
```
```

如果你的 HDF5 key 不是默认的 `train/test/neighbors`，可以指定：

```bash
.../run_dataset.py --hdf5 your.hdf5 --train-key train --test-key test --neighbors-key neighbors
```

## 本地快速测试（run_dataset.sh）

如果你只想快速验证“本机环境 + DiskANN-rs benchmark + harness 脚本”是否能通，可以直接跑 [`scripts/run_dataset.sh`](scripts/run_dataset.sh)。

它本质上是对 [`scripts/run_dataset.py`](scripts/run_dataset.py) 的一层 wrapper，参数通过环境变量传入（带默认值）：

- `HDF5_PATH`：输入 `.hdf5` 路径
- `DATASET`：输出目录名（写到 `runs/<DATASET>/...`）
- `DISTANCE`：距离度量（例如 `cosine`）
- `SEARCH_N` / `SEARCH_L`：search sweep 参数
- `REBUILD_INDEX`：`true/false`，是否强制重建缓存 index
- `EMON_ENABLE`：`true/false`，是否启用 EMON（本地 sanity 建议先 `false`）

### 用仓库自带的小数据跑通（推荐）

workspace 根目录自带了一个很小的 HDF5：`tmp_sanity_small.hdf5`，适合做“能不能跑通”的测试。

```bash
# 1) 确保你当前 shell 的 python 有 numpy/h5py
#    （例如：source /mnt/nvme2n1p1/xtang/diskann-workspace/.venv/bin/activate）

cd DiskANN-playground/extend-rabitq/ann-harness/scripts

# 2) 跑一个很小的 sweep，尽量快
HDF5_PATH=/mnt/nvme2n1p1/xtang/diskann-workspace/tmp_sanity_small.hdf5 \
DATASET=tmp-sanity-small \
DISTANCE=cosine \
SEARCH_N=10 \
SEARCH_L=10,20,40 \
REBUILD_INDEX=true \
EMON_ENABLE=false \
bash run_dataset.sh
```

预期输出目录：`DiskANN-playground/extend-rabitq/ann-harness/runs/tmp-sanity-small/<timestamp>/`。

如果你想跑你自己的数据，把 `HDF5_PATH` 换成你的 `.hdf5` 即可；如果远端/本地数据 key 不同，请直接改用 `run_dataset.py`（支持 `--train-key/--test-key/--neighbors-key`）。

### 常见问题

- `run_benchmark.sh` 会调用 `cargo run ... --package diskann-benchmark`，因此需要本机已安装 Rust toolchain，并且能编译 [DiskANN-rs](../../../DiskANN-rs/)。
- 如果 `python` 找不到 `numpy/h5py`，优先检查你是否激活了正确的虚拟环境。

## 远程跑（remote-test）

如果你希望把“上传代码 + 远端编译 + 远端跑 benchmark + 拉回 runs 结果”串起来，可以用 `extend-rabitq/remote-test`。

它会在远端执行：

1) 上传 `DiskANN-playground/`（以及可选的 `DiskANN-rs/`）
2) 确保远端存在 C++ DiskANN repo（缺失则 clone）
3) 运行 `DiskANN-playground/diskann-rs/build_all_targets.sh`（编译 DiskANN-rs）
4) 在远端跑 `DiskANN-playground/extend-rabitq/ann-harness/scripts/run_dataset.sh`
5) 把最新一次 `runs/<dataset>/<timestamp>/` 下载回本地 `DiskANN-playground/extend-rabitq/ann-harness/runs/`

详细说明见：

- [../remote-test/README.md](../remote-test/README.md)

### 最小用法

1) 从示例配置复制一份：

```bash
cp DiskANN-playground/extend-rabitq/remote-test/config.example.json \
   DiskANN-playground/extend-rabitq/remote-test/config.json
```

安装依赖（本机）：

```bash
python3 -m pip install -r DiskANN-playground/extend-rabitq/remote-test/requirements.txt
```

2) 编辑 `config.json`（至少填远端 host/username，确认远端路径与数据集路径）。

3) 先 dry-run 看看会做什么：

```bash
python3 DiskANN-playground/extend-rabitq/remote-test/remote_test.py \
  --config DiskANN-playground/extend-rabitq/remote-test/config.json \
  --dry-run
```

4) 正式跑：

```bash
python3 DiskANN-playground/extend-rabitq/remote-test/remote_test.py \
  --config DiskANN-playground/extend-rabitq/remote-test/config.json
```

### 关键配置项提示

- `paths.remote_hdf5_path`：远端数据集路径。remote-test 会通过设置 `HDF5_PATH=...` 环境变量传给 `run_dataset.sh`。
- `data_copy`：如果远端没有数据集，可以开启把本地 `.hdf5` 自动拷到远端。
- `paths.remote_results_dir`：把 `ann-harness/runs` 变成 symlink 指向独立目录，便于清理远端工作区但保留结果。
- `conda`：可选的远端 conda 集成（自动激活/按需创建 env）。
- `cpu_bind`：可选绑核（`taskset` / `numactl`），默认只对 test 阶段生效。

## 结果文件定位与解读

假设运行时指定：`--dataset my-dataset`，则输出目录为：

`DiskANN-playground/extend-rabitq/ann-harness/runs/my-dataset/<timestamp>/`

其中 `<timestamp>` 是每次运行自动生成的 UTC 时间戳目录；因此同一个 dataset 可以保存多次运行的结果而不互相覆盖。

其中关键文件：

- `outputs/output.json`
  - diskann-benchmark 的**原始完整输出**（build + search 的结构化结果），后处理都从这里读取
- `outputs/summary.tsv`
  - 用于画 recall–QPS/latency 曲线的**汇总表**（建议优先看这个）
- `outputs/details.md`
  - 从 `output.json` 抽取的更“底层”的解释报告（build 时间、training_time、quantized_bytes、每个 layout 的 run 结构等）
- `server-info.txt`
  - 记录本次运行所在服务器的 CPU/系统信息（来自 `lscpu`），方便横向对比不同机器上的结果

如果使用 `--emon-enable`，则会额外看到：

- `configs/pq-vs-spherical.build.json`
  - build-only 配置（清空 `search_phase.runs`，并强制每个 job 的 `save_path` 写入 `indices/`）
- `configs/pq-vs-spherical.search.json`
  - search-only 配置（把每个 job 的 `index_operation.source` 切到 `Load`，从 `indices/` 读取）
- `outputs/output.build.json` / `outputs/output.search.json`
  - 分别对应 build-only 与 load+search 的 benchmark 输出
- `outputs/summary.build.tsv` / `outputs/summary.search.tsv`
- `outputs/details.build.md` / `outputs/details.search.md`
- `emon/emon.dat`
  - `emon -collect-edp` 的输出（每个 job 一份；每份采样只覆盖该 job 的 search-only invocation）

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
