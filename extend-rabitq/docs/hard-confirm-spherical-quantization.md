# “硬核确认”：DiskANN-rs `spherical-quantization` 是否等价于论文 Extended RaBitQ

目的：给出一套**可重复、可点击定位**的检查点，用来确认 DiskANN-rs 的 `spherical-quantization` 实现确实包含 Extended RaBitQ 的关键算法组件（而不是仅“名字像”）。

> 结论先行（基于代码证据）：DiskANN-rs 的 `spherical-quantization` **明确实现了** extended 部分最核心的 *Algorithm 1（maximize cosine similarity / critical values 枚举求最优缩放）*，并在距离估计端实现了 *offset + correction* 的 compensated 估计式（代码注释也直接引用 RabitQ）。
> 
> 同时：论文里常见的 *MSB split + FastScan 的两阶段候选增量精化* 并不是 DiskANN-rs 目前的同构实现方向（更多是 query layout/SIMD 友好布局）。

---

## A. “必须存在”的 Index Phase 核心：Algorithm 1

Extended RaBitQ 相比原始 RaBitQ 的一个硬核特征是：对多 bit/维度的量化码，使用 **Algorithm 1**（critical values 枚举）来选择最优缩放，从而最大化余弦相似度。

**检查点 A1：Algorithm 1 的明确引用与函数入口**
- 入口注释直接写着“Refer to algorithm 1 …”
- 位置：
  - `maximize_cosine_similarity` 的注释与函数定义：
    - [DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs](../../DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs#L971)
    - [DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs](../../DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs#L982)

**检查点 A2：critical values + heap 更新结构（同构特征）**
- 你应该能在 `maximize_cosine_similarity` 内看到：
  - “Compute the critical values and store them on a heap.”（critical values）
  - heap root 更新逻辑（`update_root`）
- 位置：
  - [DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs](../../DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs#L999)
  - [DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs](../../DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs#L1020)
  - [DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs](../../DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs#L1029)

**检查点 A3：多 bit 压缩走 maximum cosine 路线**
- 你应该能看到压缩函数 `compress_via_maximum_cosine`，并在其内部调用 `maximize_cosine_similarity`。
- 位置：
  - `compress_via_maximum_cosine` 定义：[DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs](../../DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs#L862)
  - 内部调用 `maximize_cosine_similarity`：[DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs](../../DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs#L877)

---

## B. “必须存在”的 Query/Distance 核心：offset + correction 的 compensated 估计

Extended RaBitQ 的另一个硬核特征是：距离/内积不是简单 bit inner product，而是要把**offset（无符号码 → 以 0 为中心的格点）**与**校正项 correction**组合起来做 compensated/unbiased 估计。

**检查点 B1：代码注释直接引用 RabitQ 估计式**
- 位置：
  - [DiskANN-rs/diskann-quantization/src/spherical/vectors.rs](../../DiskANN-rs/diskann-quantization/src/spherical/vectors.rs#L33)
  - [DiskANN-rs/diskann-quantization/src/spherical/vectors.rs](../../DiskANN-rs/diskann-quantization/src/spherical/vectors.rs#L41)

**检查点 B2：数据侧 meta 包含 `inner_product_correction` 与 `bit_sum`**
- `inner_product_correction`（校正）+ `bit_sum`（展开 offset 修正时必需的求和项）是非常“extended-rabitq-like”的结构。
- 位置：
  - [DiskANN-rs/diskann-quantization/src/spherical/vectors.rs](../../DiskANN-rs/diskann-quantization/src/spherical/vectors.rs#L232-L246)

**检查点 B3：offset term 的定义（对应 $(2^B-1)/2$）**
- 位置：[DiskANN-rs/diskann-quantization/src/spherical/vectors.rs](../../DiskANN-rs/diskann-quantization/src/spherical/vectors.rs#L312)

**检查点 B4：CompensatedIP / CompensatedSquaredL2 的存在与组合方式**
- 你应该能看到 `CompensatedIP` 与 `CompensatedSquaredL2`，并且其计算里出现：
  - `inner_product_correction` 的乘法
  - `offset` 与 `bit_sum` 的修正项（形如 `ip - offset*(...) + offset^2*dim` 或类似展开）
- 位置：
  - offset/bit_sum 修正项示例：[DiskANN-rs/diskann-quantization/src/spherical/vectors.rs](../../DiskANN-rs/diskann-quantization/src/spherical/vectors.rs#L484-L527)
  - `CompensatedSquaredL2`：[DiskANN-rs/diskann-quantization/src/spherical/vectors.rs](../../DiskANN-rs/diskann-quantization/src/spherical/vectors.rs#L537)
  - `CompensatedIP`：[DiskANN-rs/diskann-quantization/src/spherical/vectors.rs](../../DiskANN-rs/diskann-quantization/src/spherical/vectors.rs#L688)

---

## C. “必须存在”的随机正交/近似正交变换（论文 $P$）

Extended RaBitQ workflow 里通常有一个随机正交（或近似正交）变换 $P$（或 $P^{-1}$）。DiskANN-rs 抽象为 transform。

**检查点 C1：TransformKind 提供 RandomRotation / Hadamard 类变换**
- 位置：DiskANN-rs/diskann-quantization/src/algorithms/transforms/mod.rs
- 你可以重点看这些定义/枚举：
  - `TransformKind::PaddingHadamard`：[DiskANN-rs/diskann-quantization/src/algorithms/transforms/mod.rs](../../DiskANN-rs/diskann-quantization/src/algorithms/transforms/mod.rs#L78)
  - `TransformKind::DoubleHadamard`：[DiskANN-rs/diskann-quantization/src/algorithms/transforms/mod.rs](../../DiskANN-rs/diskann-quantization/src/algorithms/transforms/mod.rs#L92)
  - `TransformKind::RandomRotation`：[DiskANN-rs/diskann-quantization/src/algorithms/transforms/mod.rs](../../DiskANN-rs/diskann-quantization/src/algorithms/transforms/mod.rs#L103)

---

## D. “实现差异确认”：论文 FastScan / MSB split

如果你要确认 DiskANN-rs **是不是完整复刻**论文里 *MSB split + FastScan 两阶段增量精化*，可以用反证法：

- 在 `DiskANN-rs/diskann-quantization/src/spherical/**` 下搜索 `FastScan` / `MSB` / `split` 等关键字（目前没有直接命中）。
- DiskANN-rs 更偏向提供不同的 query 表示方式（layout），比如 `FourBitTransposed`，这更像“布局/SIMD 优化”，而不是论文 4.2 那种“数据码分层 + 候选集两阶段计算”。

---

## E. Query Layout（你 benchmark 里看到 layout 的根源）

你在 benchmark 输出里看到 `layout=full_precision/same_as_data/scalar_quantized`，这来自 `QueryLayout` 的定义。

- `QueryLayout` enum：DiskANN-rs/diskann-quantization/src/spherical/iface.rs#L494-L508
- `FourBitTransposed`：DiskANN-rs/diskann-quantization/src/spherical/iface.rs#L501
- `ScalarQuantized`：DiskANN-rs/diskann-quantization/src/spherical/iface.rs#L505
- `FullPrecision`：DiskANN-rs/diskann-quantization/src/spherical/iface.rs#L508

- `QueryLayout` enum：[DiskANN-rs/diskann-quantization/src/spherical/iface.rs](../../DiskANN-rs/diskann-quantization/src/spherical/iface.rs#L494-L508)
- `FourBitTransposed`：[DiskANN-rs/diskann-quantization/src/spherical/iface.rs](../../DiskANN-rs/diskann-quantization/src/spherical/iface.rs#L501)
- `ScalarQuantized`：[DiskANN-rs/diskann-quantization/src/spherical/iface.rs](../../DiskANN-rs/diskann-quantization/src/spherical/iface.rs#L505)
- `FullPrecision`：[DiskANN-rs/diskann-quantization/src/spherical/iface.rs](../../DiskANN-rs/diskann-quantization/src/spherical/iface.rs#L508)

---

## F. 一键复现（你本机可跑的“确认命令”）

在 repo 根目录执行：

```bash
grep -n "Refer to algorithm 1" DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs
grep -n "fn maximize_cosine_similarity" DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs
grep -n "critical values" DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs

grep -n "CompensatedIP" DiskANN-rs/diskann-quantization/src/spherical/vectors.rs
grep -n "CompensatedSquaredL2" DiskANN-rs/diskann-quantization/src/spherical/vectors.rs
grep -n "offset_term" DiskANN-rs/diskann-quantization/src/spherical/vectors.rs
grep -n "inner_product_correction" DiskANN-rs/diskann-quantization/src/spherical/vectors.rs

grep -n "pub enum QueryLayout" DiskANN-rs/diskann-quantization/src/spherical/iface.rs
grep -n "TransformKind" DiskANN-rs/diskann-quantization/src/algorithms/transforms/mod.rs
```

如果你本机装了 ripgrep（`rg`），也可以用 `rg -n` 做同样的事（输出更舒服）。

如果这些检查点都在，并且 `maximize_cosine_similarity` 的实现具备 critical values + heap 更新结构，同时 `vectors.rs` 的 compensated 公式包含 offset/bit_sum/correction，那么就可以非常强地说：DiskANN-rs 的 `spherical-quantization` 在“extended RaBitQ 的核心算法机制”上是对齐的。
