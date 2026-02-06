# Extended RaBitQ 论文 vs DiskANN-rs（Rust）逐条对照

> 目标：把论文（`extend rabitq.pdf` / `extend_rabitq.txt`）里“Extended RaBitQ”的关键步骤（索引阶段/查询阶段/估计式）逐条映射到 DiskANN-rs 的具体 **函数 / 模块 / 数据结构**，并说明哪些地方是“同构实现”、哪些地方是“等价但实现不同”、哪些地方目前 **未实现**。

相关材料：
- 论文文本抽取：[`../docs/extend_rabitq.txt`](../docks/extend_rabitq.txt)
- 论文 PDF：[`../docs/extend rabitq.pdf`](../docks/extend%20rabitq.pdf)

代码范围（DiskANN-rs）：主要集中在 `diskann-quantization` crate 的 `spherical` 实现：
- 量化器：`DiskANN-rs/diskann-quantization/src/spherical/quantizer.rs`
- 距离估计（含补偿项）：`DiskANN-rs/diskann-quantization/src/spherical/vectors.rs`
- 动态接口 / query layout：`DiskANN-rs/diskann-quantization/src/spherical/iface.rs`
- 随机正交变换：`DiskANN-rs/diskann-quantization/src/algorithms/transforms/*`

---

## 0. 论文工作流（对照锚点）

论文在 `extend_rabitq.txt` **约 3.4 小节**位置明确总结了 workflow（可在 txt 中搜索 `We summarize the workflow of the extended RaBitQ`，大约在 390 行附近）。该 workflow 的核心是：

- **Index phase**：采样随机正交矩阵 $P$；对数据向量做 $P^{-1}$ 变换并归一化得到 $o'$；用 Algorithm 1 得到每个向量的量化码。
- **Query phase**：对 query 做同样的 $P^{-1}$ 与归一化得到 $q'$；用论文给出的估计式（文本里对应 Eq. (11)(12) 等）估计内积/距离。

下面逐条映射到 DiskANN-rs。

---

## 1. 论文符号 → DiskANN-rs 对应物（总表）

| 论文概念 / 符号 | 论文含义 | DiskANN-rs 对应实现 |
|---|---|---|
| $P$（随机正交矩阵） | 随机旋转（正交变换） | `TransformKind::{RandomRotation, PaddingHadamard, DoubleHadamard}` + `Transform::transform_into` |
| $P^{-1}$ 应用于 $o, q$ | 将向量旋转到量化空间 | DiskANN-rs 直接“应用 transform”得到 `transformed`（正交时 $P^{-1}=P^T$，方向不影响“正交等距”性质） |
| $C$ / centroid / mean | 居中用的中心点 | `SphericalQuantizer.shift: Poly<[f32]>`（训练得到的均值/中心） |
| 居中：$X = X' - C$ | 消除均值偏置 | `SphericalQuantizer::preprocess` 里构造 `shifted = mul * x - shift[i]` |
| 归一化：$x = X/|X|$ | 映射到单位球 | `preprocessed.shifted_norm` + 在压缩前 `shifted[i] /= shifted_norm` |
| 网格 $G$ 与偏移 $(2^B-1)/2$ | 将 $B$ bit 无符号码对应到以 0 为中心的格点 | `DataMeta::offset_term::<NBITS>()` 以及 `compress_via_maximum_cosine` 的 `offset = (2^B-1)/2` |
| Algorithm 1（critical values 枚举、找最优缩放 $t$） | 找使余弦相似度最大的格点 | `maximize_cosine_similarity`（注释直接引用 arxiv Algorithm 1） |
| 存储量化码 $\bar y_u$ | $B$ bit/维度无符号编码 | `DataMut<'_, NBITS>` 的 `vector_mut().set_unchecked(i, v as u8)` |
| 估计式（内积/距离 + unbiased correction） | 用 $\langle \bar o, q\rangle/\langle \bar o, o\rangle$ 等 | `DataMeta.inner_product_correction` + `CompensatedIP` / `CompensatedSquaredL2` 里的组合公式 |
| MSB split / FastScan（论文 4.2） | 用高位先筛，再用低位增量精化 | DiskANN-rs **没有按论文方式 split data code**；但有 `QueryLayout::FourBitTransposed`（query 的 bit-transpose 布局，用 SIMD 快算 bit inner product），属于“实现方向相近但不是论文同一套机制” |

---

## 2. Index phase（建索引/压缩数据向量）逐条对照

### 2.1 采样随机正交矩阵 $P$

论文：构造 codebook 需要一个随机正交矩阵 $P$。

DiskANN-rs：把“随机正交变换”抽象为 `TransformKind`，在训练量化器时创建并保存在 `SphericalQuantizer.transform` 中。

- 入口：`SphericalQuantizer::train(...)` → `SphericalQuantizer::generate(...)`
- 关键点：`Transform::new(transform_kind, dim, Some(rng), allocator)`
- 具体 transform：
  - `TransformKind::RandomRotation { .. }`：材料化一个矩阵并做矩阵乘（见 `algorithms/transforms/random_rotation.rs`）。
  - `TransformKind::PaddingHadamard { .. }`：实现 $H D x / \sqrt{n}$，其中对角 $D$ 的随机符号由 RNG 采样（见 `padding_hadamard.rs`）。Hadamard 是正交（再配合缩放），属于“随机正交变换”的一种高效实现。
  - `TransformKind::DoubleHadamard { .. }`：类似思路但支持非 2 次幂维度（更偏工程优化）。

对应关系：论文的 $P$ 在 DiskANN-rs 中等价为 `transform`（可能不是“显式随机旋转矩阵”，但仍是 distance-preserving/（近似）正交的随机变换）。

### 2.2 居中（shift / centroid）

论文描述了对数据向量做 shift/normalize/rotate（见 `extend_rabitq.txt` 3.2.1 附近以及 3.4 workflow）。

DiskANN-rs：居中是 `SphericalQuantizer.shift`（训练数据均值）。

- 训练时：
  - 若 metric 为 L2/IP：`compute_means_and_average_norm(data)`
  - 若 metric 为 Cosine：`compute_normalized_means(data)`
- 压缩时：`SphericalQuantizer::preprocess` 构造
  - `shifted[i] = mul * x[i] - shift[i]`

注意：论文的“网格 shift（把无符号整数 shift 到以 0 为中心的格点）”是另一个层次的 shift；DiskANN-rs 将其体现在编码/解码的 `offset_term` 中（见 2.4）。

### 2.3 归一化得到 $o'$（单位向量）

论文：对 $P^{-1}$ 后的向量做 normalize 得到 $o'$。

DiskANN-rs：对居中向量（`shifted`）做归一化，并把其作为 transform 的输入：

- `compress_into_with`（Data / Query / FullQuery 都类似）：
  - `preprocessed.shifted_norm = FastL2Norm.evaluate(shifted)`
  - `shifted[i] /= shifted_norm`
  - `transform.transform_into(dst, shifted, scratch)`

对应关系：
- 论文中的归一化在 DiskANN-rs 是显式实现。
- 论文中的 $P^{-1}$ 在 DiskANN-rs 是“应用 transform”；正交情形下方向（$P$ vs $P^{-1}$）不会破坏等距性。

### 2.4 用 Algorithm 1 计算量化码（critical values 枚举 + 最优缩放）

论文：Algorithm 1 的核心是枚举 critical values（让某一维 rounding 发生变化的缩放阈值），维护当前 rounding 的格点向量 $y_{cur}$ 的内积与范数，找到使余弦相似度最大的缩放 $t_{max}$。

DiskANN-rs：对应实现非常直接，且源码注释明确指向论文 arxiv 的 Algorithm 1。

- 关键函数：
  - `compress_via_maximum_cosine::<NBITS>(...)`
  - `maximize_cosine_similarity(v, num_bits, allocator) -> scale`

实现要点对照：

1) **格点/offset**（论文 Eq. (7) 的“把无符号码 shift 到以 0 为中心的格点”）
- DiskANN-rs 在写入码时用：
  - `offset = (2^B - 1)/2`
  - `v = round(scale * t + offset)`（再 clamp 到 `[0, 2^B-1]`）
  - `dv = v - offset`（这一步就是把无符号码映射回中心化格点）

2) **critical values 的枚举**（论文 Algorithm 1 line 3-5）
- DiskANN-rs 用 `Pair { value, position }` 表示“下一次让某一维 rounding 增 1 的最小缩放值”。
- 用 `SliceHeap` 维护 min-heap，每次取最小 critical value 只改变 `rounded[position]` 一维，然后 $O(1)$ 更新：
  - `current_ip += abs(vp)`
  - `current_square_norm += 2 * old_r`（对应论文中 $\|y_{cur}\|$ 的增量更新）

3) **最大余弦相似度的选择**（论文 Algorithm 1 line 6-7）
- DiskANN-rs 计算 `similarity = current_ip / sqrt(current_square_norm)` 并更新 `optimal_scale`。

结论：DiskANN-rs 的 `maximize_cosine_similarity` 是论文“extended”部分最关键的 Algorithm 1 的同构实现。

### 2.5 写入数据向量元数据（用于 unbiased / compensated 距离估计）

论文：距离估计依赖一些“只与 data 有关的量”（例如 $\|\bar y\|$、$\langle \bar o, o\rangle$ 等），可在索引阶段预计算。

DiskANN-rs：把这些补偿项存进 `DataMeta`（`vectors.rs` 定义）：

- `DataMeta.inner_product_correction: f16`
- `DataMeta.metric_specific: f16`
- `DataMeta.bit_sum: u16`

对应写入位置：
- 1-bit data（原始 RaBitQ-like）：`FinishCompressing for DataMut<1>`
- 多 bit data（extended）：`compress_via_maximum_cosine`
  - 计算 `self_inner_product = \langle dv, transformed \rangle`
  - 计算 `inner_product_correction = (transformed_norm * shifted_norm) / self_inner_product`
  - `bit_sum` 为无符号码的和（对应论文里会用到的 $\sum y_u$ / 偏移修正项）

`metric_specific` 的含义：
- L2：存 $\|X\|^2$（shifted 的平方范数）
- IP/Cosine：存 $\langle X, C\rangle$（居中后与 centroid 的内积）

这些来自 `Preprocessed::metric_specific()`。

---

## 3. Query phase（查询向量处理 + 距离估计）逐条对照

### 3.1 query 的同构预处理：居中 + 归一化 + transform

论文：对 query 做 $P^{-1}$ 与归一化得到 $q'$。

DiskANN-rs：query 压缩同样调用 `preprocess` 和 `transform.transform_into`。

在接口层（dyn 版本）：`spherical/iface.rs` 暴露 `QueryLayout` 来选择 query 的表示方式。

- `QueryLayout::FullPrecision`：`SphericalQuantizer` 压缩到 `FullQuery`（保留 float transformed 向量）
- `QueryLayout::ScalarQuantized`：把 query 在 transformed 空间做 **标量量化**（每个 query 取 min/max 做线性量化）
- `QueryLayout::FourBitTransposed`：把 query 压缩成 4-bit 并做 bitwise transpose（为了 SIMD 友好）

对应论文的“标准 extended RaBitQ query”（即对 $q'$ 做 Eq. (12)）：DiskANN-rs 最接近的是 `FullPrecision`（因为直接保留 transformed 的 float），以及“与 data 同码本的 Data-as-query”。

### 3.2 论文 Eq. (11)(12) 的计算落点：CompensatedIP / CompensatedSquaredL2

论文的关键形式（以文本里 Eq. (11)(12) 为代表）：

- 用 $y_u$（无符号码）和 $q'$（浮点 query）做内积；
- 再减去 offset（即 $(2^B-1)/2$）乘以 $\sum q'$ 的项；
- 再乘上与 $\|y\|$、$\langle \bar o,o\rangle$ 相关的校正，得到 unbiased 的距离/内积估计。

DiskANN-rs：这些逻辑分散在两层：

1) **补偿项的定义与存储**：`DataMeta` / `QueryMeta` / `FullQueryMeta`
2) **距离函数把这些项组合起来**：`vectors.rs` 的 `CompensatedIP` / `CompensatedSquaredL2`

#### 3.2.1 FullPrecision query × quantized data

对应实现（示例：IP）：
- `vectors.rs` 中 `impl Target2<..., FullQueryRef, DataRef> for CompensatedIP`

核心结构：
- 先算 bit-level inner product：`s = <float_query, bitslice_data>`（通过 `distances::InnerProduct` kernel）
- 再做 offset 修正：`ip = s - sum(query) * offset`
- 再乘上 data 的 `inner_product_correction` 与 query 的 `shifted_norm` 等项（见实现里的 `xc.shifted_norm * yc.inner_product_correction * (...)`）

这一步就是论文 Eq. (12) 的工程化版本：
- `offset` 来自 `DataMeta::offset_term::<NBITS>()`（对应 $(2^B-1)/2$）
- `sum(query)` 对应 $\sum q'$
- `inner_product_correction` 承载了“除以 $\|y\|$、再除以 self-dot/correction”的组合项

#### 3.2.2 Scalar-quantized query × quantized data

对应实现（IP/L2 都有）：
- `CompensatedIP`/`CompensatedSquaredL2` 对 `QueryRef<Q, Perm> × DataRef<D>` 的实现

其形式是更一般的：

- query 侧有 `QueryMeta { inner_product_correction, bit_sum, offset, metric_specific }`
- data 侧有 `DataMeta { inner_product_correction, bit_sum, metric_specific }`
- 组合式中出现：
  - `ip - y_offset * xc.bit_sum + xc.offset * yc.bit_sum - y_offset * xc.offset * dim`

这就是把论文里“无符号码 + offset 的展开”推广到“query 和 data 都带 offset/scale”的通用表达。

### 3.3 由内积估计到距离（论文 Eq. (2) 的落点）

论文：用估计内积 $\langle o, q\rangle$（或 $\langle X,Y\rangle$）来得到平方 L2：
$$\|X-Y\|^2 = \|X\|^2 + \|Y\|^2 - 2\langle X,Y\rangle$$

DiskANN-rs：`CompensatedSquaredL2` 正是把该式与补偿内积拼起来：
- 对称量化（Data×Data）：`xc.metric_specific + yc.metric_specific - 2*kernel(...)`
- 混合（Query×Data）：`yc.metric_specific + xc.metric_specific - 2*corrected_ip`
- FullPrecision×Data：同样先 offset 修正再组合

`metric_specific` 在 L2 下就是 $\|X\|^2$，所以直接落到公式里。

---

## 4. 论文 4.2（MSB split + FastScan 增量精化）与 DiskANN-rs 的差异

论文 4.2（在 `extend_rabitq.txt` 大约 440 行之后）提出：
- 把 $B$ bit/维度的码 $\bar y_u$ 拆成“最高位拼接形成 $\bar y_0$”和剩余位 $\bar y_{last}$；
- 先用 $\bar y_0$（等价于原始 RaBitQ 的 1-bit code）配合 FastScan 快速估计并剪枝；
- 对未剪枝候选再访问 $\bar y_{last}$ 做更精确估计（增量式）。

DiskANN-rs（截至当前代码）：

- **没有**看到对 **data code** 做 MSB split 并在候选集上“先算高位、再按需算低位”的增量流程。
- 但是它在 query 侧提供了一个“为了 SIMD 友好而改变布局”的选项：
  - `QueryLayout::FourBitTransposed`（见 `spherical/iface.rs`）
  - 本质是把 query 的 4-bit 编码做 bitwise transpose，方便后续在 `distances::InnerProduct` 内核里做更高吞吐的 bit inner product。

因此：DiskANN-rs 与论文在“想把 bit inner product 做到极快（FastScan 类方向）”上目标相近，但实现点不同：
- 论文：**数据码分层 + 候选增量精化**
- DiskANN-rs：**query layout/压缩策略选择 + 统一一次性 distance kernel**

如果你希望在 DiskANN-rs 路线里复现论文的 4.2，需要的新增点大概率是：
- 为 data 侧引入“分层存储”（MSB 与剩余 bits 分离）；
- 在候选评估中实现两阶段/多阶段 distance 估计与剪枝策略；
- 并把它接入 IVF/图检索的 candidate loop。

---

## 5. 你可以从哪些入口开始追代码（建议阅读顺序）

1) `SphericalQuantizer::train` / `generate`（量化器训练与 transform 初始化）
2) `SphericalQuantizer::compress_into_with`（Data / Query / FullQuery 三类压缩路径）
3) `compress_via_maximum_cosine` + `maximize_cosine_similarity`（Algorithm 1 同构实现）
4) `DataMeta` / `QueryMeta` / `FullQueryMeta`（补偿项含义）
5) `CompensatedIP` / `CompensatedSquaredL2`（最终距离公式如何拼出来）
6) `spherical/iface.rs` 的 `QueryLayout`（工程化的 query 表示选择）

---

## 6. 小结（结论性映射）

- 论文“Extended RaBitQ”的 **核心 extended 量化算法（Algorithm 1：critical values + 最优缩放）** 在 DiskANN-rs 中对应 `maximize_cosine_similarity`，并且实现方式与论文描述高度一致。
- 论文“随机正交矩阵 $P$”在 DiskANN-rs 中对应 `TransformKind`（`RandomRotation` 或 Hadamard 系列变换）。
- 论文的估计式（offset 展开 + unbiased correction）在 DiskANN-rs 中以 `DataMeta/QueryMeta/FullQueryMeta` + `CompensatedIP/L2` 的方式工程化落地。
- 论文 4.2 的 **MSB split + FastScan 增量剪枝** 在 DiskANN-rs 中未见同构实现；DiskANN-rs 提供的是“query layout/压缩策略选择”来优化吞吐。
