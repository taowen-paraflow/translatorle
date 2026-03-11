# Gated Delta Networks 混合架构在移动 NPU 上的部署挑战分析

> 基于 Qwen3.5-0.8B 在 Intel NPU 上的实测经验，以及 HY-MT 1.8B 在高通 NPU 上的部署经验，分析 Intel NPU 的 FP16 限制在高通 NPU 上是否同样存在，以及高通 NPU 的精度模型对 GDN 部署意味着什么。

## 1. GDN 是什么

**Gated Delta Networks**（GDN / Gated DeltaNet）是 NVIDIA 提出的线性时间序列建模架构（arXiv:2412.06464, ICLR 2025），结合了 delta rule（联想记忆）和 gating 机制（类似 Mamba）。

核心递推公式：

```
S_t = exp(g_t) × S_{t-1} + beta_t × outer(k_t, delta_t)

其中:
  delta_t = v_t - exp(g_t) × S_{t-1}^T @ k_t     ← "delta rule"：只存新信息
  g_t = 衰减门控（learned, input-dependent）
  S_t ∈ R^{d×d} = 固定大小的隐状态矩阵
```

与 Transformer 的关键差异：

| 维度 | Transformer | GDN |
|------|------------|-----|
| 复杂度 | O(N²) — attention 对所有历史 token | O(N) — 线性递推，逐步更新 |
| 状态管理 | KV cache **持续增长**（每步追加一行） | 隐状态 S **固定大小**（d×d 矩阵） |
| 推理模式 | 并行 attention + 自回归 decode | 类 RNN——逐 token 串行更新状态 |
| 长序列 | KV cache 内存线性增长，可能 OOM | 内存恒定，天然支持超长序列 |

### 混合架构

实际部署的模型不会全用 GDN。Qwen3.5 和 Kimi-Linear 采用 **GDN + 全注意力混合**，典型比例 3:1：

```
Qwen3.5-0.8B: 24 层
  ├── 18 层 GDN (线性注意力，delta rule 递推)
  └──  6 层 Full Attention (标准 softmax，每 4 层一个)

状态变量:
  ├── 18 conv state + 18 recurrent state (GDN 层)
  └──  6 key cache + 6 value cache (注意力层)
```

### 使用 GDN 的代表模型

| 模型 | 规模 | 架构 | 状态 |
|------|------|------|------|
| Qwen3.5-0.8B | 0.8B | 18 GDN + 6 Attn | 已发布 |
| Qwen3.5-35B-A3B | 35B (MoE) | GDN + Attn 混合 | 已发布 |
| Kimi-Linear-48B-A3B | 48B (MoE) | "Kimi Delta Attention" | 已发布 |
| Gated-Deltanet-1.3B | 1.3B | 纯 GDN | 研究用 |

## 2. Intel NPU 上的实测：FP16 精度导致输出乱码

我们在 `translatorle/qwen35` 项目中对 Qwen3.5-0.8B 做了完整的 Intel NPU 部署实验。结论很残酷：

| 模式 | 速度 | 输出质量 | 说明 |
|------|------|---------|------|
| CPU | 10-14 tok/s | **正确** | FP32 计算，精度充足 |
| NPU (NPUW_LLM) | 18.9 tok/s | **乱码** | FP16 精度不足 |
| NPU (旧静态) | 13.4 tok/s | **乱码** | FP16 精度不足 |
| NPU+CPU 混合 | 10.7 tok/s | **乱码** | Shadow FP32 state 仍不够 |

**同一份 IR，CPU 上输出正确，NPU 上输出乱码**——问题已精确定位到 NPU 硬件的 FP16 计算精度。

### 根因分析：两个独立的精度问题

Intel NPU 的计算精度固定为 FP16——**计算和存储全部 FP16，没有 FP32 累加器**。即使 IR 声明 FP32，NPU 插件自动转 FP16 执行和存储。这导致了两个独立的精度问题：

**问题 1：跨步 state 累积（乘性误差）**

```
S_t = exp(g_t) × S_{t-1} + ...
          ↑          ↑
     FP16 门控   FP16 状态 ← 上一步的 FP16 舍入误差在这里被乘法传播

第 1 步:  S_1 有 ~0.1% 误差
第 10 步: S_10 的误差已经累积到 ~1%
第 50 步: 误差 3-5%，logits 开始分叉
```

KV cache 不存在这个问题——写一次读多次（append-only），第 10 步写入的值在第 100 步读取时完全不变。

**问题 2：逐层 hidden_state 传播（加性误差）**

```
即使 state 来自 CPU FP32（Shadow FP32 方案）:
  Layer 0: hidden_state 经过 FP16 matmul → 引入 ~0.1% 误差
  Layer 1: 在上一层误差基础上再 FP16 matmul → 误差叠加
  ...
  Layer 23: 24 层 FP16 传播后，累积误差足以让 logits 分叉
```

这就是 Shadow FP32 State 方案失败的原因——它解决了问题 1，但没解决问题 2。Intel NPU 的矩阵乘法用 FP16 做加法累积（没有 FP32 累加器），大量乘累加操作的舍入误差在每层放大。

## 3. 高通 NPU 的精度模型：关键差异

在 HY-MT 项目中，我们在高通 SM8650 上成功部署了 HY-MT 1.8B Transformer。仔细审视这个管线的精度特性，发现和 Intel NPU 有**结构性差异**：

### 3.1 高通 NPU 有 FP32 累加器

我们的量化配方是 `dynamic_wi4_afp32`：
- **wi4**：权重 INT4 量化
- **afp32**：FP32 累加（accumulated FP32）

这意味着高通 HTP 在做矩阵乘法时：

```
Intel NPU:   a[FP16] × b[FP16] → 累加到 FP16 寄存器 → 输出 FP16
                                        ↑
                              每次加法都有 FP16 舍入
                              数千次乘累加后误差显著

高通 HTP:    a[FP16] × w[INT4] → 累加到 FP32 寄存器 → 输出转回 FP16
                                        ↑
                              加法在 FP32 精度下执行
                              数千次乘累加的舍入误差大幅减少
```

**FP32 累加器是一个关键差异。** 矩阵乘法的精度主要取决于累加精度（因为一次 matmul 涉及成千上万次乘累加），而不是单次乘法精度。FP32 累加意味着每层 matmul 的输出误差远小于纯 FP16 的 Intel NPU。

### 3.2 CPU 侧 buffer 是真 FP32

在 HY-MT 的 split 架构中：

```
高通 split 管线的数据精度流:

  Embedder (CPU):   FP32 查表 → FP32 输出到共享内存
  AUX mask (CPU):   FP32 计算 → FP32 输出到共享内存
  AUX rope (CPU):   FP32 计算 → FP32 输出到共享内存
  Main (NPU):       读取 FP32 输入 → 内部 FP16 计算(FP32累加) → FP32 输出
  AUX cache (CPU):  读取 FP32 kv_slice → FP32 scatter-write 到 FP32 kv_cache
```

KV cache 存储在 CPU 管理的 FP32 buffer 中（通过 AHWB/ION 共享内存）。NPU 读取时可能内部转 FP16，但写回时转回 FP32。关键是**状态的持久化存储是 FP32**。

对比 Intel NPU：

| 维度 | Intel NPU | 高通 NPU (split 管线) |
|------|-----------|---------------------|
| **matmul 累加** | FP16 累加 | **FP32 累加** |
| **层间 hidden_state** | FP16 | FP16 计算，但 FP32 累加减少误差 |
| **状态存储** | ReadValue/Assign = FP16 | CPU 侧 buffer = **FP32** |
| **辅助计算** | 全在 NPU = FP16 | AUX 在 CPU = **FP32** |

### 3.3 实测验证：Transformer 在高通 NPU 上精度无问题

HY-MT 1.8B 在高通 NPU 上的翻译输出与 CPU 完全一致——32 层 Transformer 在 FP16 计算 + FP32 累加下没有精度退化。Intel NPU 上的传统 Transformer（如 Qwen3-ASR、HY-MT）同样 FP16 够用。说明：

- **对 Transformer：** 两家 NPU 都没有精度问题（KV cache append-only，无乘性累积）
- **对 GDN：** Intel NPU 失败了（FP16 一切）。高通 NPU 有 FP32 累加 + FP32 状态存储，**情况可能不同**

## 4. 核心问题：高通 NPU 的 FP32 能力能否拯救 GDN？

回到 Intel NPU 的两个精度问题，逐一分析高通 NPU 是否能避免：

### 4.1 问题 1（跨步 state 累积）：高通可以解决

在高通的 split 管线中，state update 可以完全在 CPU 上用 FP32 执行——和 HY-MT 的 KV cache update 走 CPU AUX 模型完全一样：

```
HY-MT Transformer (已验证可行):
  NPU: 32 层 matmul → 输出 kv_slice (FP32 buffer)
  CPU AUX: kv_cache[pos] = kv_slice   ← FP32 scatter-write

假设的 GDN split (同样思路):
  NPU: 投影 matmul → 输出 q, k, v, g, beta (FP32 buffer)
  CPU AUX: S_new = exp(g) × S_old + beta × outer(k, delta)   ← FP32 递推
  → state 始终在 CPU 侧 FP32 buffer 中，永不降到 FP16
  → 跨步累积误差问题彻底消除
```

Intel NPU 上 Shadow FP32 方案能解决问题 1，高通也能。两者在这一点上等价。

### 4.2 问题 2（逐层 hidden_state 传播）：高通有优势

这是关键差异所在。Intel NPU 上 Shadow FP32 仍然失败，因为 24 层 FP16 matmul 的累积误差太大。但高通有 **FP32 累加器**：

```
Intel NPU 逐层传播:
  Layer N 输入 x (FP16)
  → q = x @ W_q   ← FP16 × FP16, FP16 累加 → 输出 FP16 (误差 ~0.1%)
  → ... 7 次 matmul，每次 FP16 累加 ...
  → Layer N 输出 (误差 ~0.5-1.0%)
  → 24 层后: 误差 ~5-10% → 乱码

高通 HTP 逐层传播:
  Layer N 输入 x (FP16)
  → q = x @ W_q   ← FP16 × INT4, FP32 累加 → 输出转回 FP16 (误差 ~0.01%)
  → ... 7 次 matmul，每次 FP32 累加 ...
  → Layer N 输出 (误差 ~0.05-0.1%)
  → 24 层后: 误差 ~1-2% → 可能仍然可用？
```

FP32 累加让每次 matmul 的输出精度提高约一个数量级。24 层传播后，累积误差可能控制在 logits 不分叉的范围内。

**但这需要实测验证**——Transformer 没有逐层精度问题（因为它不依赖递推 state），所以 HY-MT 的成功不能直接证明 GDN 也行。GDN 的 readout `out = q @ S` 依赖 state 精度，而 state 即使是 FP32 的，query 在 NPU 上仍然是 FP16 计算出的。

### 4.3 关键的架构约束：GDN readout 的层内依赖

即使精度问题解决了，GDN 在 NPU 上还有一个 Transformer 没有的架构约束：

```
Transformer 每层的数据流（可以整体放 NPU）:
  x → QKV投影 → Attention(Q, K, V, KV_cache) → O投影 → FFN → x_next
  ↑ 全部是 matmul + softmax + 拼接，NPU 一次搞定

GDN 每层的数据流（readout 依赖更新后的 state）:
  x → QKV投影(NPU可做) → state_update(需FP32) → readout(依赖state) → O投影+FFN(NPU可做)
                                    ↑                      ↑
                              必须在 CPU FP32         依赖 CPU 的结果
  → 每层需要 NPU→CPU→NPU 切换
```

如果把 24 层全放 NPU（像 HY-MT 那样一次调 NPU 跑完所有层），state update 也会在 NPU 上用 FP16 执行，退回 Intel NPU 的老路。如果逐层拆分 NPU 和 CPU，则每层需要 NPU↔CPU 切换，开销很大。

**可能的折中：按 block 拆分**

```
方案: 把模型拆成若干 block，每 block 包含 3 层 GDN + 1 层 Attention

  Block 0 (Layer 0-3):
    NPU: Layer 0-2 GDN 的投影+FFN + Layer 3 Attention 全部
         GDN state_update+readout 也在 NPU 上做 (FP16+FP32累加)
    CPU: 读出 NPU 输出的 state，FP32 校正后写回

  Block 1 (Layer 4-7): 同上
  ... × 6 blocks
```

每 block 只有 3 层 GDN 在 NPU 上用 FP16 做 state 操作，误差累积有限。block 之间由 CPU 用 FP32 校正 state。这比逐层切换（72 次）少很多（6 次），但比 HY-MT 的一进一出（1 次）还是多。

### 4.4 定量估算

以 Qwen3.5-0.8B 为例（1024 维，24 层）：

```
每次 NPU↔CPU 切换的数据搬运:
  hidden_state: 1024 × 4 bytes = 4KB
  GDN state (18 层): 18 × (state_dim × state_dim) × 4 bytes
    state_dim = head_dim × num_heads = 128 × 8 = 1024
    = 18 × 1024 × 1024 × 4 = 72MB  ← 这才是大头

HY-MT Transformer 每步搬运:
  embeddings: 2048 × 4 = 8KB     (进)
  kv_slice: 64 × 512 bytes = 32KB (出)
  logits: 120818 × 4 = 472KB     (出)
  总计: ~512KB/step

GDN 逐层切换每步搬运 (最坏):
  18 层 × state × 2 (读+写) = ~144MB/step
  + hidden_state 切换 = ~几 KB (可忽略)
  → 远超 HY-MT
```

GDN state 太大是根本问题。state 尺寸 `d×d`（如 1024×1024 per head）乘以 18 层 = 72MB。即使用 AHWB 共享内存避免物理拷贝，NPU↔CPU 缓存一致性同步也有开销。

**但如果按 block 拆分（6 块），只在 block 之间搬运 state**，每步 ~6 次 × 状态校正，量级可控。

## 5. 对比总结

| 精度维度 | Intel NPU | 高通 NPU (split 管线) | 对 GDN 的影响 |
|----------|-----------|---------------------|--------------|
| matmul 累加 | **FP16** | **FP32** | 高通逐层误差更小，24 层传播后差距显著 |
| state 存储 | ReadValue/Assign = **FP16** | CPU buffer = **FP32** | 高通可以彻底消除跨步 state 累积 |
| 辅助计算 | 全在 NPU = **FP16** | AUX 在 CPU = **FP32** | state update 在 CPU FP32，精度有保障 |
| 层间传播 | 24 层纯 FP16 → 发散 | 24 层 FP16+FP32累加 → **可能可控** | 需要实测验证 |

```
Intel NPU 的 GDN 困境:
  ① 跨步 state 累积: FP16 存储 → 乘性误差爆炸 → Shadow FP32 可解
  ② 逐层传播: FP16 累加 → 24 层后发散 → 无解 ← 致命

高通 NPU 的潜在优势:
  ① 跨步 state 累积: CPU FP32 buffer → 彻底解决 ✅
  ② 逐层传播: FP32 累加 → 每层误差更小 → 24 层后可能不发散 → 需要实测
                                                                    ↑
                                              这是 Intel 和高通的核心差异
```

## 6. 结论

| 问题 | 回答 |
|------|------|
| Intel NPU 的 FP16 限制是什么？ | **计算和存储全部 FP16，没有 FP32 累加器**。导致 GDN state 跨步累积误差 + 逐层传播误差双重爆炸 |
| 高通 NPU 也有同样限制吗？ | **部分有，部分没有。** 计算仍是 FP16，但有 **FP32 累加器**且 state 可存在 **CPU 侧 FP32 buffer** |
| FP32 累加器有多重要？ | 让每次 matmul 的输出误差降低约一个数量级。Intel 逐层传播发散的根因正是 FP16 累加 |
| 高通能跑 GDN 吗？ | **有可能但需验证。** 跨步 state 累积可解（CPU FP32）；逐层传播误差因 FP32 累加而大幅减少，但 24 层是否够稳定需要实测 |
| 主要障碍是什么？ | 不再是精度，而是 **架构**——GDN 每层的 readout 依赖 state update，导致 NPU↔CPU 切换频繁。需要按 block 拆分来平衡精度和效率 |

**核心结论：Intel NPU 跑 GDN 失败的两个根因中，高通 NPU 彻底解决了一个（FP32 state 存储），大幅缓解了另一个（FP32 累加减少逐层误差）。高通 NPU 上 GDN 的可行性显著高于 Intel NPU，但最终能否成功取决于 FP32 累加能否让 24 层传播保持稳定——这需要实际移植和测试来验证。**
