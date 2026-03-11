# 当 GDN 遇上 Intel NPU：FP16-only 硬件的累加精度瓶颈

> 我们在 Intel Lunar Lake NPU 上部署 Qwen3.5-0.8B（Gated Delta Networks 混合架构）的完整历程。尝试了 6 种方案，全部因 NPU 的 FP16-only 累加器精度不足而失败。本文分析根因，并对比不同架构在 FP16-only NPU 上的适用性。

## 背景

Qwen3.5 是阿里最新发布的语言模型，采用了 **Gated Delta Networks (GDN)** 混合架构——24 层中 18 层是 GDN 线性注意力，6 层是标准 Full Attention（每 4 层一个）。GDN 用线性递归的状态矩阵替代了传统 Transformer 的 KV cache，理论上推理更高效。

GDN 架构在业界已获得广泛支持和验证：

- **vLLM** 已为 Qwen3.5 GDN 架构提供专门优化（GDN attention layout、projector fusion、decode 加速）
- **主流硬件全面兼容**：GDN 在 NVIDIA GPU、AMD GPU、Apple Silicon、高通 NPU、华为昇腾、百度昆仑芯等平台上均可正确运行
- 社区围绕 GDN 的讨论集中在"效率与精度的工程权衡"和"Delta Rule 带来的比 Mamba 更强的可编辑性"，而非数值稳定性问题
- 已知的数值不稳定报告仅限于极端实验配置或特定 FP16-only 硬件，并非 GDN 架构本身的缺陷

我们的目标：在 Intel Core Ultra 7 258V（Lunar Lake）笔记本的 NPU 上跑通 Qwen3.5-0.8B 推理。由于该 NPU 的 FP16-only 累加器限制，6 种方案均未成功。以下是完整分析。

## 三种架构的状态更新机制

要理解问题，先看三种主流架构如何管理"记忆"：

### Transformer：追加式 KV Cache

```
K_cache = concat(K_cache, k_new)    # 追加新 key
V_cache = concat(V_cache, v_new)    # 追加新 value
output  = softmax(q @ K_cache^T) @ V_cache
```

**关键特性：写一次，读多次。** 每个 token 的 KV 写入 cache 后就不再修改。历史值永远保持原始精度。即使存储为 FP16，误差不会跨步累积。

### Mamba (S6)：对角衰减 + 线性累加

```
dA = exp(A * dt)                    # 离散化衰减矩阵（对角）
dB = B * dt                         # 离散化输入矩阵
S_t = dA ⊙ S_{t-1} + dB ⊙ x_t     # 状态更新（逐元素）
y_t = sum(S_t ⊙ C_t)               # 输出
```

**状态张量**：`[batch, heads, dim, dstate]`，其中 `dstate` 通常为 16~64。

**关键特性：乘法衰减。** 每步把上一帧的状态乘以 `dA`（接近 1.0 的衰减因子），再加上新输入。这个乘法让 FP16 的舍入误差**逐步累积**。

### GDN (Gated Delta Networks)：门控衰减 + Delta Rule

```
S_t = exp(g_t) × S_{t-1}                      # 门控衰减
    + k_t ⊗ (β_t × (v_t − S_{t-1} @ k_t))    # Delta 规则更新
o_t = q_t^T @ S_t                              # 输出
```

**状态张量**：`[batch, heads, key_dim, value_dim]`，Qwen3.5-0.8B 中为 `[1, 16, 128, 128]`。

**关键特性：不仅有乘法衰减（同 Mamba），还有一个 Delta 反馈环。** `v_t − S_{t-1} @ k_t` 这一项从当前状态中"读回"信息来计算残差——状态中的误差直接影响下一步的更新方向，形成**误差正反馈**。

### 对比总结

| 特性 | Transformer | Mamba | GDN |
|------|:-----------:|:-----:|:---:|
| 状态更新方式 | 追加（append） | 乘法衰减 + 加法 | 乘法衰减 + Delta 反馈 |
| 状态是否被修改 | 否（只追加） | 是（每步覆写） | 是（每步覆写） |
| FP16 误差累积 | **无** | 线性累积 | 线性累积 + 反馈放大 |
| 状态大小 (0.8B) | KV: `[H, L, D]` 随序列增长 | `[H, D, 16]` 固定 | `[H, 128, 128]` 固定 |
| FP16-only NPU 风险 | **安全** | **较高** | **很高** |

## Intel NPU 的硬件限制

Intel Lunar Lake NPU (NPU 4) 的计算精度：

| 运算类型 | 输入精度 | 累加器精度 |
|---------|---------|-----------|
| FP16 MAC | FP16 | **FP16** |
| INT8 MAC | INT8 | **INT32** |
| INT4 MAC | INT4 | **INT32** |

**关键：FP16 乘加运算的累加器只有 FP16 精度。** NPU 支持的精度仅限 FP16、INT8、INT4，**不支持 BF16 和 FP32**。

作为对比，同一颗 Lunar Lake SoC 上的其他计算单元：
- **CPU**（Lion Cove）：通过 AVX-512 支持 BF16，完整 FP32 计算能力
- **GPU**（Xe2 / BMG）：Xe2 XMX 支持 BF16，FP32 累加器

也就是说，BF16 和 FP32 并非 Lunar Lake 平台不具备的能力，只是 NPU 单元没有实现。

对比高通 SM8650 (Hexagon NPU)：

| 运算类型 | 输入精度 | 累加器精度 |
|---------|---------|-----------|
| FP16 MAC | FP16 | **FP32** |
| INT8 MAC | INT8 | **INT32** |

高通的 FP32 累加器意味着：即使输入是 FP16，中间计算和状态更新用 FP32——足够支撑 GDN 的线性递归。实际验证：**高通 NPU 上 Qwen3.5 GDN 输出与 CPU 完全一致**（参见 [qwen35-impl.md](https://github.com/AQL-org/HY-MT/blob/main/qwen35-impl.md)）。

Intel NPU 没有 FP32 累加器。即使 IR 模型声明 FP32，NPU 插件会自动将所有计算降为 FP16。这不是软件 bug，是硬件设计。

**未来展望**：Intel Panther Lake (NPU 5, 预计 2025-2026) 可能增加 FP32 中间处理能力和 BF16 支持，届时线性递归架构的兼容性有望改善。

## 我们的 6 次尝试

我们在 Qwen3.5-0.8B 上系统性地尝试了所有可能的方案：

### 方案 1：NPUW_LLM 直接推理

最直接的方案——用 OpenVINO 的 NPUW_LLM 管线。

```
结果：18.9 tok/s，输出乱码
原因：24 层 GDN state 全程 FP16，误差从第 1 层累积到第 24 层
```

### 方案 2：Python 管理 State（旧静态模式）

不用 NPUW_LLM，自己在 Python 端管理所有 48 个 state 张量。

```
结果：13.4 tok/s，输出乱码
原因：同上，state 在 NPU 上更新后读回，仍然是 FP16 精度
```

### 方案 3：Shadow FP32 State（NPU+CPU 混合）

**核心思路**：NPU 执行前向传播，同时输出 GDN 中间量（g_t, k_t, v_t, β_t）。CPU 用这些中间量做 FP32 精度的 state 更新，下一步再喂回 NPU。

```python
# 每步推理：
# 1. FP32 state → 喂入 NPU（自动截断 FP16）
# 2. NPU 前向：FP16 计算 + 输出 72 个中间量
# 3. CPU FP32 更新 shadow state
S_fp32 = S_fp32 * g_t + outer(k_t, beta_t * (v_t - S_fp32 @ k_t))
# 4. 下一步用 FP32 state，不用 NPU 的 FP16 state
```

```
结果：10.7 tok/s，仍然输出乱码
原因：跨步的 state 累积误差解决了，但每步内部 24 层 hidden_state
     在 NPU 上 FP16 传播，层间误差仍然太大
```

**这个方案证明了：问题不仅在 state 更新，hidden_state 在 24 层 FP16 传播本身就会发散。**

### 方案 4-5：LL2 Pipeline / Fork OpenVINO

尝试用 LowLatency2 自动展开 Loop 节点、fork OpenVINO 教 NPUW_LLM 识别 GDN state。架构层面的改进都是正确的，但本质问题不变——NPU FP16 精度不够。

### 方案 6：多子图 NPU v2（最终尝试）

受高通 NPU 实现的启发，做了最激进的精度保护方案：

**24 层拆成 6 个子图**（每组 4 层 = 3 GDN + 1 Full Attention），子图之间 hidden_state 回 CPU 转 FP32。

额外改进：
- **Host 预计算 Rotary Embedding**：cos/sin 在 CPU 上用 FP32 计算，作为子图的显式输入（不在 NPU 上算）
- **FP32 shadow state**：每个子图的 GDN 中间量输出到 CPU 做 FP32 state 更新
- **FP32 inter-subgraph hidden_state**：子图间传递 FP32 张量，只有 2KB/边界

```python
for si in range(6):  # 链式执行 6 个子图
    outputs = subgraph[si].infer(hidden_fp32, cos, sin, mask, *states)
    hidden_fp32 = outputs["hidden_states"].astype(np.float32)  # 回 CPU FP32
    cpu_fp32_state_update(si, outputs)  # FP32 shadow state
logits = outputs["logits"]
```

理论上，FP16 误差从 24 层累积降到了 4 层。

```
CPU 验证（同样的子图 IR，在 CPU 上跑）：
  "Hello, my name is John. I am a 20-year-old male." ✅ 正确

NPU 实测：
  "Hello, my name is a very important part of the part of the part of..." ❌ 乱码
```

**即使仅 4 层 FP16，GDN 的 Delta Rule 反馈环仍然导致输出发散。**

### 完整结果表

| 方案 | 速度 | 输出 | FP16 层数 |
|------|------|------|----------|
| NPUW_LLM | 18.9 tok/s | 乱码 | 24 层 |
| Python 管 State | 13.4 tok/s | 乱码 | 24 层 |
| Shadow FP32 State | 10.7 tok/s | 乱码 | 24 层（state 已修复） |
| **多子图 v2** | **8.1 tok/s** | **乱码** | **4 层** |
| CPU 多子图 | 4.2 tok/s | 正确 | 0 层 |
| **CPU 标准** | **7.8 tok/s** | **正确** | **0 层** |

**NPU 速度（8.1 tok/s）甚至没有超过 CPU（7.8 tok/s）。** 0.8B 模型太小，host↔device 数据拷贝抵消了 NPU 的并行计算优势。

## GDN 对累加精度的要求比 Mamba 更高

Mamba 和 GDN 都包含线性递归，但 GDN 的 Delta Rule 对累加器精度提出了更高要求。这不是架构设计缺陷——Delta Rule 的反馈机制正是 GDN 具备比 Mamba 更强"可编辑性"的关键。但它意味着硬件必须提供足够的中间计算精度：

### Mamba 的状态更新

```
S_t = dA ⊙ S_{t-1} + dB ⊙ x_t
```

误差路径是**单向的**：`S_{t-1}` 的误差通过 `dA` 衰减传递给 `S_t`，但不影响输入项 `dB ⊙ x_t`。

### GDN 的状态更新

```
residual = v_t − S_{t-1} @ k_t        # ← S 的误差直接污染 residual
S_t = exp(g_t) × S_{t-1} + k_t ⊗ (β_t × residual)
```

误差路径是**双向的**：`S_{t-1}` 的误差不仅通过衰减传递，还通过 `S_{t-1} @ k_t` **反馈到下一步的更新量**。这意味着：

1. `S` 有误差 → `S @ k` 偏离真实值
2. `v - S @ k` 计算出错误的残差
3. 错误的残差写入 `S` → 误差放大
4. 循环 1-3

在具备 FP32 累加器的硬件上（GPU、高通 NPU 等），这个反馈环正常工作，是 GDN 优于 Mamba 的核心优势。但在 FP16-only 硬件上，反馈环会放大舍入误差——这就是为什么即使只有 4 层 FP16，GDN 仍然发散，而 Mamba 在类似配置下 **可能** 表现更好（状态维度也小得多：16~64 vs 128x128）。

### 误差累积模型

设 FP16 单步相对误差为 ε ≈ 5×10⁻⁴（半精度尾数 10 位）：

| 架构 | N 步后状态误差 | 机制 |
|------|--------------|------|
| Transformer | 0 | 无状态修改 |
| Mamba | ~N × ε | 线性累积（衰减抑制） |
| GDN | ~N × ε × (1 + feedback) | 线性累积 + Delta 反馈放大 |

对于 Qwen3.5-0.8B 的 256 token 生成：
- Mamba 理论误差：~12.8% （可能仍可用）
- GDN 理论误差：远大于 12.8%（Delta 反馈使有效 ε 增大）

## 对其他线性递归架构的影响

这个问题**不是 GDN 独有的**。任何包含 `S_t = f(S_{t-1})` 递归状态更新的架构，在 FP16-only 硬件上都面临同样的风险：

| 架构 | 递归公式 | 反馈环 | FP16-only NPU 风险 |
|------|---------|--------|-------------------|
| **Transformer** | 无（append-only KV） | 无 | **安全** |
| **Mamba / S4 / S6** | `S = dA * S + dB * x` | 无 | **中等**（dstate 小可能撑住） |
| **RWKV** | `wkv = e^w * wkv + e^k * v` | 无 | **中等**（类似 Mamba） |
| **GDN** | `S = e^g * S + k ⊗ (v - S@k)` | **有** | **很高** |
| **Linear Attention** | `S = S + φ(k) ⊗ v` | 无 | **低**（无衰减，但可能溢出） |
| **RetNet** | `S = γ * S + k ⊗ v` | 无 | **中等** |

注意：以上风险评估**仅针对 FP16-only NPU**（如 Intel Lunar Lake NPU）。在具备 FP32 累加器的硬件上（GPU、高通 NPU 等），这些架构均可正常工作。

**建议：在 FP16-only NPU 上部署线性递归模型前，务必先在 CPU 上验证输出一致性。**

## INT8 量化行不行？

不行。原因如下：

Intel NPU 的 INT8 MAC 确实有 INT32 累加器，精度比 FP16 高。但 GDN 的状态更新包含：

- `exp(g_t)`：指数函数，值域 0.9~1.0，无法用 INT8 精确表示
- 逐元素乘法 `S × exp(g)`：不是 MatMul，INT8 MAC 单元做不了
- `S @ k`：可以做 INT8 MatMul，但 S 的动态范围每步都在变化

INT8 量化适用于**线性层（MatMul/Conv）**的权重和激活，这些在 GDN 的投影部分没问题。但递归 state 更新中的非线性操作（exp、逐元素乘、动态范围变化的矩阵）不是 INT8 能处理的。

## GDN 在主流平台的表现

需要强调的是：本文记录的问题是 Intel NPU FP16-only 累加器的特定限制，而非 GDN 架构的普遍缺陷。GDN 在主流计算平台上均可正常工作：

| 平台 | 累加器精度 | GDN 状态 |
|------|-----------|---------|
| NVIDIA GPU (CUDA) | FP32 | 正常 |
| AMD GPU (ROCm) | FP32 | 正常 |
| Apple Silicon (MPS/ANE) | FP32 | 正常 |
| 高通 NPU (Hexagon) | FP32 | 正常（已验证） |
| 华为昇腾 (Ascend) | FP32 | 正常 |
| 百度昆仑芯 | FP32 | 正常 |
| x86/ARM CPU | FP32 | 正常（已验证） |
| **Intel NPU (Lunar Lake)** | **FP16** | **输出错误** |

大多数开发者不会遇到本文描述的问题，除非专门针对 Intel NPU 部署。

## 结论与建议

### 对模型开发者

1. **GDN 在主流硬件上没有问题**：在 GPU、高端 NPU（具备 FP32 累加器）及 CPU 上，GDN 工作正常。不需要因为 Intel NPU 的特定限制而回避 GDN 架构
2. **如果你的模型需要在 Intel NPU 上部署**：目前建议使用标准 Transformer。Intel NPU 对 KV cache（append-only）支持完善
3. **部署前确认目标硬件的累加器精度**：GDN 的 Delta Rule 反馈机制要求至少 FP32 累加器。高通 NPU 可以，Intel NPU 不行

### 对硬件厂商

1. **FP32 累加器是支持新一代模型架构的关键**：随着 Mamba、GDN、RWKV 等线性递归架构的普及，FP16-only 的 NPU 将无法运行越来越多的新模型
2. **核心需求是 FP32 累加器，而非 BF16**：BF16 虽然有更大的指数范围（8 位 vs FP16 的 5 位），有助于避免溢出/下溢，但其尾数精度（7 位）实际上低于 FP16（10 位）。对于线性递归中的 state 累积，尾数精度同样重要。真正的解决方案是 FP32 累加器——即使输入为 FP16/BF16，中间计算和状态存储使用 FP32

### 对部署工程师

1. **先 CPU 验证**：在 NPU 部署前，用同一份 IR 在 CPU 上跑一遍，确认输出一致
2. **小心"速度正常但输出乱码"**：NPU 推理速度可能很好看（18.9 tok/s），但输出完全错误。这是 FP16-only 累加器的精度问题，不是模型或软件 bug
3. **传统 Transformer 在 Intel NPU 上很好**：我们的 ASR（Qwen3-ASR 1.7B）和 MT（HY-MT 1.8B）都是标准 Transformer，在 Intel NPU 上运行正确且高效
4. **这是一个边界案例**：除非你的部署目标包含 Intel NPU，否则 GDN 模型可以放心使用

## 实验环境

- **硬件**：Intel Core Ultra 7 258V (Lunar Lake), 32GB RAM
- **NPU**：Intel NPU (4th Gen), FP16 compute
- **软件**：OpenVINO 2025.1, Python 3.12
- **模型**：Qwen3.5-0.8B (24 layers: 18 GDN + 6 Full Attention)
- **代码**：[translatorle/qwen35](https://github.com/anthropics/translatorle/tree/main/qwen35)（含全部 6 种导出和推理方案）

---

*2026 年 3 月 | 在 Intel Lunar Lake NPU 上的实测记录*
