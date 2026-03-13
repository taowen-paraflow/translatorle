# Chunkwise Parallel GDN Prefill 优化方案

**目标**: 消除 Qwen3.5 GDN prefill 阶段的串行瓶颈，用 WY 表示法将逐 token Loop 转为 chunk 级并行 MatMul。

**硬件**: Intel Core Ultra 7 258V (Lunar Lake) -- iGPU Xe2 8-core + NPU 4th gen
**框架**: OpenVINO 2026.0
**模型**: Qwen3.5-0.8B (24 层: 18 GDN + 6 full attention)

## 1. 问题背景

当前 GDN 层在 OpenVINO IR 中使用 **Loop 节点**实现递归，每个 token 串行执行一次 Loop body。Decode 阶段（S=1）没有问题，但 **prefill 阶段 GDN 是绝对瓶颈**。

### 实测数据（35 token prompt, GPU_ONLY）

```
Prefill 时间拆解:

  GDN blocks x 6 (GPU):     1295ms  ████████████████████████████████████████  80.3%
  Attn blocks x 6 (GPU):     262ms  ██████                                   16.2%
  Head (GPU):                  56ms  █                                         3.5%
  总计:                      1613ms
```

### 为什么 GDN prefill 这么慢

模型有 6 个 GDN block，每个 block 内含 3 个 GDN 层，总共 **18 个 GDN 层**。每个 GDN 层的 Loop 节点对输入序列中的每个 token 串行迭代一次。

对于 44 token 的 prompt:

```
串行迭代总数 = 18 层 x 44 token = 792 次 Loop 迭代
```

每次 Loop 迭代包含:
- Gate 计算 (`alpha`, `beta`)
- 递推状态更新 `S_t = alpha_t * S_{t-1} * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T`
- 输出计算 `o_t = S_t * q_t`

这 792 次 iteration 全部串行，GPU 无法并行化。相比之下，attention 层的 MatMul 天然支持 batch 并行（S=44 一次计算完成）。

### Layer-major prefill 没有帮助

当前 layer-major prefill 已经优化了调度（GDN 层用 full-batch S=prompt_len 一次 `infer()`），但 **Loop 节点内部仍然是逐 token 串行执行**。`infer()` 调用次数减少了，Loop 迭代次数没有变。

## 2. 核心算法: WY 表示法 + Chunkwise Parallel

### 来源

- **DeltaNet**: Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", arXiv:2406.06484, NeurIPS 2024
- **Gated DeltaNet**: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", arXiv:2412.06464, ICLR 2025 (NVIDIA)

### GDN 递推公式

标准逐 token 递推:

```
S_t = alpha_t * S_{t-1} * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T
o_t = S_t * q_t
```

其中:
- `S_t`: 递推状态矩阵 [d_v, d_k]（Qwen3.5: d_v = d_k = 128）
- `alpha_t`: 遗忘门标量（gating）
- `beta_t`: 输入门标量
- `k_t, q_t`: key/query 向量 [d_k]
- `v_t`: value 向量 [d_v]

**关键洞察**: 转移矩阵 `(I - beta_t * k_t * k_t^T)` 是 rank-1 更新的 Householder 类似物（Householder-like），其乘积可以用 **WY 表示法** 紧凑表达。

### WY 表示法

对于 chunk 内的 C 个 token，转移矩阵的乘积:

```
prod_{i=s}^{t} (I - beta_i * k_i * k_i^T) = I - W_{s:t} * K_{s:t}^T
```

其中 `W` 可通过三角求解得到（不需要串行递推！）:

```
A = tril(-diag(beta) * Gamma * (K * K^T), -1)    # 下三角，对角线以下
T = (I - A)^{-1} * diag(beta)                      # 三角矩阵求逆（前代法）
W = T * K                                           # [C, d_k]
```

其中 `Gamma` 是由 `alpha` 门控值构成的衰减矩阵: `Gamma[i,j] = prod_{m=j+1}^{i} alpha_m`。

**核心突破**: 从 O(C) 步串行递推 变为 一次矩阵运算 + 三角求解。

## 3. 算法步骤（chunk 大小 C）

### Intra-chunk 计算（可并行）

给定一个 chunk 的输入 Q, K, V, alpha, beta（各 [C, d] 或 [C]），以及进入 chunk 时的状态 S:

**Step 1: 构建衰减掩码矩阵**

```
Gamma[i,j] = prod_{m=j+1}^{i} alpha_m      # [C, C] 衰减矩阵
A = tril(-diag(beta) * Gamma * (K @ K^T), -1)   # [C, C] 下三角
```

- `K @ K^T` 是标准 MatMul [C, C]
- `diag(beta)` 是逐行缩放
- `tril(..., -1)` 是下三角掩码（对角线置零）

**Step 2: 三角求解**

```
T = solve_triangular(I - A, diag(beta))    # (I - A) * T = diag(beta)
```

- `(I - A)` 是单位下三角矩阵，求逆即前代法 (forward substitution)
- T 的形状 [C, C]
- 这一步只涉及 C x C 矩阵运算，C 通常很小（16~64）

**Step 3: 构建 W 和 U**

```
W = T @ K    # [C, d_k]
U = T @ V    # [C, d_v]
```

- 两个标准 MatMul

**Step 4: 计算输出**

```
# 状态贡献（跨 chunk）
state_contrib = Q @ S^T                         # [C, d_v]

# chunk 内贡献（因果掩码）
causal_mask = tril(ones(C, C))                   # 下三角
intra_chunk = (Q @ K^T * causal_mask) @ (U - W @ S^T)  # [C, d_v]

# 最终输出
O = state_contrib + intra_chunk                  # [C, d_v]
```

### Inter-chunk 状态传播（串行，但只有 L/C 步）

```
Delta = (U - W @ S^T)^T @ K    # [d_v, d_k]
S_next = S + Delta              # 新状态
```

每个 chunk 结束后更新一次状态，传递给下一个 chunk。

### 计算复杂度对比

| 方法 | 串行步数 | 总计算量 | 适合 GPU |
|------|---------|---------|---------|
| **Sequential Loop** | O(L) | O(L * d^2) | 否（无法并行） |
| **Chunkwise Parallel** | O(L/C) | O(L * C * d + L * d^2) | **是（chunk 内全并行）** |

对于 L=44, C=64:
- Sequential: **44 步**串行 Loop
- Chunkwise: **1 个 chunk**（44 < 64），~6 个并行 MatMul

对于 L=256, C=64:
- Sequential: **256 步**串行 Loop
- Chunkwise: **4 步**串行（4 个 chunk），每步 ~6 个并行 MatMul

## 4. 对我们的适用性

### OpenVINO 操作映射

Chunkwise 算法中的所有操作都是 OpenVINO 标准 opset，**不需要 Loop 节点**:

| 算法操作 | OpenVINO Op | 备注 |
|---------|-------------|------|
| `K @ K^T`, `Q @ S^T` | MatMul | 标准，GPU/NPU 均支持 |
| `tril(mask)` | 常量掩码 + Multiply | 预计算三角掩码 |
| `diag(beta)` | Unsqueeze + Multiply | 广播乘法 |
| `Gamma` 衰减矩阵 | CumProd / ReduceProd | 或离线计算 |
| `(I-A)^{-1}` 三角求解 | MatMul (小矩阵 C*C) | C 很小，直接展开 |
| `S_next = S + Delta` | Add | 状态更新 |

### 对典型 prompt 长度的效果

| Prompt 长度 | Chunk 大小 C | Chunk 数 | 串行步数 | 加速比 (vs Loop) |
|------------|-------------|---------|---------|----------------|
| 35 | 64 | 1 | 1 | ~35x |
| 44 | 64 | 1 | 1 | ~44x |
| 128 | 64 | 2 | 2 | ~64x |
| 256 | 64 | 4 | 4 | ~64x |

对于 44 token prompt，C=64 只需要 **1 个 chunk**。44 次串行 Loop 迭代变为 ~6 个可并行的 MatMul。

### Decode 不受影响

Decode 阶段 S=1，chunkwise 算法退化为原始递推公式（chunk 大小 1 = 单步更新）。可以保留当前 Loop 实现，或者用同一份 chunkwise 图（C=1 时等价）。

### NPU 可能性

当前 NPU 不能跑 GDN 的两个障碍:
1. FP16 精度不足 -- chunkwise 不解决
2. **NPU 不支持 Loop 节点** -- **chunkwise 完全消除 Loop**

消除 Loop 后，GDN 变为纯 MatMul + element-wise 操作图，NPU 编译器可以正常处理。这为 **NPU 跑 GDN** 打开了可能性（前提是精度问题另行解决，比如用 INT8 累加或未来 FP32 累加器）。

## 5. 参考实现

### fla-org/flash-linear-attention

```
fla/ops/gated_delta_rule/chunk.py
```

- Triton kernel 实现，vLLM v0.17+ 使用此库
- `chunk_gated_delta_rule` 函数: 输入 Q, K, V, alpha, beta, chunk_size -> 输出 O, 最终 state
- 核心函数: `chunk_gated_delta_rule_fwd` (前向), `chunk_gated_delta_rule_bwd` (反向)
- 支持 head-level 并行 (num_heads 维度)

关键代码路径:
```
fla/ops/gated_delta_rule/
  chunk.py          # 主入口
  wy_fast.py        # WY 表示法的快速 Triton 实现
  chunk_A_fwd.py    # Step 1: 构建 A 矩阵
  chunk_o_fwd.py    # Step 4: 输出计算
```

### NVlabs/GatedDeltaNet

```
gated_delta_net/
  kernel/
    chunk.py        # NVIDIA 官方 chunkwise 实现
```

- ICLR 2025 论文官方代码
- 与 flash-linear-attention 共享底层 Triton kernel
- 包含完整的训练 + 推理管线

### vLLM 集成

```
vLLM v0.17+:
  --chunked-prefill-size 2048    # 使用 chunked prefill
```

- vLLM 通过 flash-linear-attention 库支持 Gated DeltaNet
- 默认 chunk size 2048（面向 GPU HBM，我们用更小的 chunk）

## 6. 实现计划

### Phase 1: Python standalone 验证

1. **实现 chunkwise 算法**
   - 从 `fla/ops/gated_delta_rule/chunk.py` 提取纯 PyTorch/NumPy 逻辑
   - 去掉 Triton kernel 依赖，用标准 MatMul 替代
   - 输入: Q, K, V, alpha, beta, initial_state, chunk_size
   - 输出: O, final_state

2. **数值验证**
   - 加载当前 Loop 模型的权重
   - 对比 Loop 逐 token 递推 vs chunkwise 的输出
   - 验证 final_state 数值等价（容忍 FP32 精度内误差）

### Phase 2: OpenVINO IR 导出

3. **修改 `export_hybrid.py` 的 GDN 导出**
   - 在 `ov_ops.py` 的 `convert_recurrent_attention_cell` 中:
     - 当 `seq_len > 1` 时使用 chunkwise MatMul 图（prefill 路径）
     - 当 `seq_len == 1` 时使用原始递推（decode 路径）
   - 或者: 导出两套 GDN 子图（prefill 版 + decode 版），推理时按阶段选择

4. **IR 验证**
   - 导出新的 GDN block IR
   - 对比新旧 IR 在相同输入下的输出
   - 检查 state 传播正确性

### Phase 3: GPU Benchmark

5. **Prefill 性能对比**
   - 在 GPU 上 benchmark: Loop IR vs Chunkwise IR
   - 测量不同 prompt 长度 (16, 32, 64, 128, 256) 的加速比
   - Profiling 确认 MatMul 是否真正并行执行

6. **端到端集成**
   - 修改 `inference_hybrid.py` 使用新的 chunkwise GDN 子图
   - 测量完整推理（prefill + decode）的性能提升

### Phase 4: NPU 探索（可选）

7. **NPU 上跑 chunkwise GDN**
   - 无 Loop 的 chunkwise IR 应该能通过 NPU 编译
   - 测试 NPU 上 chunkwise GDN 的精度和性能
   - 如果可行，实现 GDN 也在 NPU 上跑的全 NPU 方案

## 7. 预期收益

### Prefill 阶段

| 指标 | 当前 (Loop) | 预期 (Chunkwise) | 改善 |
|------|------------|-----------------|------|
| GDN 6 blocks (35 tok) | 1295ms | 200-400ms | **3-6x** |
| 总 prefill (35 tok) | 1613ms | 518-718ms | **2-3x** |
| GDN 占比 | 80.3% | ~40-55% | 不再是绝对瓶颈 |

预期来源:
- 44 次串行 Loop -> 1 个 chunk 的 ~6 个并行 MatMul
- MatMul 是 GPU 最擅长的操作，可充分利用 Xe2 8-core 并行度
- 三角求解在 C=64 的小矩阵上开销极低

### Decode 阶段

```
Decode: 不变（S=1 时 chunkwise 等价于原始递推）
```

### NPU 兼容性

```
当前: NPU 无法编译 Loop 节点 -> GDN 只能跑 GPU
未来: Chunkwise IR 无 Loop -> NPU 可能可以编译运行 GDN
```

如果 NPU 能跑 GDN，则:
- 全部 24 层可以跑 NPU（GDN + Attention），GPU 只跑 Head
- GPU 完全空闲，可同时跑其他模型（ASR、MT）
- Decode 也可能移到 NPU（取决于 NPU MatMul 性能 vs 内存带宽）

## 8. 与 AMD 对比

### 当前差距

AMD OGA Hybrid 用 **phase-level split**（NPU prefill + GPU decode），Intel 我们用 **layer-level split**（GDN->GPU, Attn->NPU）。Layer-level 的核心问题是 GDN 串行瓶颈导致两个设备互相等待。

### Chunkwise 后的格局变化

| 维度 | AMD OGA Hybrid | Intel Chunkwise (预期) |
|------|---------------|----------------------|
| Prefill 策略 | NPU prefill (全层) | **Chunkwise GDN** (GPU 并行 MatMul) |
| Prefill GDN 瓶颈 | 无（NPU 编译时展开 Loop） | **消除**（无 Loop，纯 MatMul） |
| Decode 策略 | GPU decode | GPU decode（不变） |
| NPU GDN 可能性 | 已支持 | **新开可能**（无 Loop，待验证精度） |

### 关键意义

1. **Prefill 加速直接可见**: 不需要等 Intel NPU 支持 Loop，直接在 GPU 上用 chunkwise 加速
2. **解除 HYBRID 模式的 GDN 瓶颈**: GDN 从 80% 降到 ~40-55%，Attn 在 NPU 上的节省开始有意义
3. **HYBRID 有望超过 GPU_ONLY**: 当 GDN 不再是绝对瓶颈时，NPU 跑 Attn 节省的 5ms/token 不再被 GPU 空转抵消
4. **为全 NPU 方案铺路**: 如果 chunkwise GDN 能在 NPU 编译通过，加上精度方案（INT8 累加 / 未来 FP32），Qwen3.5 有可能全部跑 NPU

## 9. 风险和注意事项

1. **精度**: WY 表示法涉及三角求解和矩阵乘法累积，FP16 下可能引入额外误差。建议:
   - Phase 1 全程用 FP32 验证
   - Phase 3 在 GPU 上用 FP32 计算中间结果
   - `(I-A)^{-1}` 三角求解对小矩阵 (C<=64) 精度问题不大

2. **Chunk 大小选择**: C 太大 -> 中间矩阵占内存多; C 太小 -> 串行步数多。对 Lunar Lake iGPU (4GB shared):
   - C=64 是合理起点（每 head 中间矩阵 64x128 = 8KB，微乎其微）
   - 短 prompt (<64) 一个 chunk 搞定，长 prompt 多几步串行

3. **IR 复杂度**: chunkwise 图比 Loop 图更复杂（更多 MatMul + reshape + mask 节点），但每个操作都是 OpenVINO 原生支持的标准 op，GPU 编译器优化成熟

4. **Decode 路径**: 必须保留 S=1 的高效路径。两种方案:
   - 导出两套 IR（prefill 版 chunkwise + decode 版 single-step）
   - 一套 IR 内用 If 节点分支（但增加图复杂度）
   - 推荐: **两套 IR**，推理时按阶段选择（与当前 NPU attention 多 shape 编译方案一致）
