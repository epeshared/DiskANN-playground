# DiskANN-rs：代码速览 + 编译/运行

本文档面向这个 workspace 结构：

- `DiskANN-rs/`：Rust workspace（真正的代码在这）
- `DiskANN-playground/`：本目录，用来放可复现的编译/运行步骤与 demo

> 重要：如果你开了新的会话窗口/新 terminal，需要先 `conda activate diskann-rs`。

## 1) 代码速览（审查笔记）

DiskANN-rs 是一个 Rust workspace（见 `DiskANN-rs/Cargo.toml` 的 `[workspace]`）：

- `diskann/`：核心算法库（图索引、search/insert、provider 抽象、错误类型等）
- `diskann-disk/`：disk index 构建/读写相关
- `diskann-providers/`：数据/存储/模型相关的 provider 实现与工具（读写 `.bin/.fbin` 等）
- `diskann-tools/`：一组 CLI 工具（生成随机数据、计算 groundtruth、生成 PQ 等）
- `diskann-benchmark/` + `diskann-benchmark-core/` + `diskann-benchmark-runner/`：基准测试 runner（JSON 输入、job 分发、输出汇总）

核心 crate `diskann` 顶层导出比较克制（`diskann/src/lib.rs`）：

- 图索引实现集中在 `diskann::graph`，其中 `DiskANNIndex<DP: DataProvider>` 在 `diskann/src/graph/index.rs`
- 构建参数通过 `diskann::graph::config::Builder` 组装并校验（`diskann/src/graph/config/mod.rs`）
- `provider` 里定义了 `DataProvider / Accessor / NeighborAccessor ...` 这类 trait，把“算法”与“数据来源”解耦
- 错误统一到 `ANNError/ANNResult`（`diskann/src/error/*`）

测试方面：`diskann/README.md` 里提到 baseline 缓存机制（`DISKANN_TEST=overwrite` 可重生成）。

## 2) Rust 工具链与编译约束

`DiskANN-rs/rust-toolchain.toml` 把工具链 pin 在 Rust `1.92`。

- 第一次在 `DiskANN-rs/` 下跑 `cargo build` 时，`rustup` 会自动下载/切换到 `1.92`（如果本机有 rustup）。
- 你的 conda 环境里可能自带 `rustc/cargo`，但以 `rust-toolchain.toml` 为准，最终会由 rustup 拉齐到 `1.92`。

## 3) 新开会话时：先激活 conda 环境

在新的 terminal 里（bash 为例）：

```bash
# 让 conda activate 生效（路径以你的 conda 安装为准）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diskann-rs
```

## 4) 编译（常用命令）

从 workspace 根目录进入 `DiskANN-rs/`：

```bash
cd /mnt/nvme2n1p1/xtang/diskann-workspace/DiskANN-rs

# 只编译核心库（推荐先跑这个确认环境）
cargo build --release -p diskann

# 编译整个 workspace（更慢，但最完整）
cargo build --release --workspace

# 编译 + 所有 targets（包含 tests/benches/examples）
cargo build --release --workspace --all-targets

# 测试（示例）
cargo test -p diskann
```

## 5) 可复现的最小可跑 Demo（脚本 + 需要的“代码”）

我在本目录提供了一个端到端脚本：

- 生成随机 base/query 向量（`diskann-tools`）
- 计算 ground truth（`diskann-tools`，当前版本会按你给的路径原样写出）
- 用 `diskann-benchmark` 跑一个 `async-index-build` job（会 build index + search）

入口：

- `diskann-rs/demo/run_benchmark_demo.sh`

使用：

```bash
cd /mnt/nvme2n1p1/xtang/diskann-workspace/DiskANN-playground/diskann-rs/demo
./run_benchmark_demo.sh
```

输出：

- 生成的数据：`diskann-rs/demo/data/`（包含 `base.fbin` / `query.fbin` / `gt`）
- benchmark 输出：`diskann-rs/demo/output/benchmark_output.json`
- 使用的 runner 输入文件：`diskann-rs/demo/output/benchmark_run.json`

如果你想调参（R/L、search_l、点数维度等），直接改脚本顶部的变量即可。

## 6) Cargo 命令速查表

进入工程目录：

```bash
cd /mnt/nvme2n1p1/xtang/diskann-workspace/DiskANN-rs
```

常用命令：

```bash
# 只编译核心库（最快确认环境）
cargo build -p diskann

# release 编译整个 workspace
cargo build --release --workspace

# release 编译 workspace + all-targets（tests/benches/examples 也会编）
cargo build --release --workspace --all-targets

# 只跑某个 package 的测试
cargo test -p diskann

# 跑整个 workspace 测试
cargo test --workspace

# 格式化检查
cargo fmt --check

# clippy（把 warning 当 error）
cargo clippy --workspace --all-targets -- -D warnings
```

性能/体积相关：

```bash
# 指定 target 目录（把编译产物放到独立目录，方便缓存复用）
export CARGO_TARGET_DIR=/mnt/nvme2n1p1/xtang/diskann-workspace/.cargo-target
```

## 7) 一键 build workspace（含 all-targets）

如果你希望“新开 shell → conda activate → 一键 build”，用这个脚本：

- [build_all_targets.sh](build_all_targets.sh)

示例：

```bash
cd /mnt/nvme2n1p1/xtang/diskann-workspace/DiskANN-playground/diskann-rs

# 默认：release + build --workspace --all-targets
./build_all_targets.sh

# 加上 fmt/clippy
./build_all_targets.sh --fmt --clippy

# debug + test
./build_all_targets.sh --debug --test
```
