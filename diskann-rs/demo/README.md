# DiskANN-rs Benchmark Demo

这是一个“端到端能跑起来”的最小 demo：

1. 生成随机 base/query 向量（`.fbin`）
2. 计算 groundtruth（当前版本写出为无扩展名的 `gt` 文件）
3. 用 `diskann-benchmark` 跑一个 `async-index-build` job（Build index + Search）

## 运行

```bash
cd /mnt/nvme2n1p1/xtang/diskann-workspace/DiskANN-playground/diskann-rs/demo
./run_benchmark_demo.sh
```

脚本会自动：

- `conda activate diskann-rs`
- 进入 `DiskANN-rs/`
- 编译 `diskann-tools` 和 `diskann-benchmark`（release）
- 生成数据到 `data/`
- 写 runner 输入到 `output/benchmark_run.json`
- 运行 benchmark 并写结果到 `output/benchmark_output.json`

## 常见问题

- 第一次运行会比较慢：`rust-toolchain.toml` pin 了 Rust `1.92`，rustup 会自动下载。
- 若你在全新 shell 里手动跑命令，记得先执行：
  - `source "$(conda info --base)/etc/profile.d/conda.sh"`
  - `conda activate diskann-rs`
