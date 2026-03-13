# Qwen3.5-0.8B Hybrid GPU+NPU 性能分析

**硬件**: Intel Core Ultra 7 258V (Lunar Lake)
**内存**: LPDDR5X-8533, 峰值带宽 68.3 GB/s（双通道）
**iGPU**: Xe2 8-core, ~67 TOPS INT8 / ~34 TFLOPS FP16
**NPU**: 4th gen, ~48 TOPS INT8
**框架**: OpenVINO 2026.0

## 1. 当前性能

| 模式 | Decode 速度 | 每 token 耗时 | 备注 |
|------|------------|-------------|------|
| **GPU_ONLY** (FP16, stateful) | **22.6 tok/s** | 44ms | 所有子图在 GPU，KV cache 常驻 GPU 显存 |
| HYBRID GPU+NPU (FP16) | 15-17 tok/s | 59-67ms | GDN→GPU, Attn→NPU (explicit I/O) |
| HYBRID GPU+NPU (量化) | 14.5 tok/s | 69ms | Attn INT4 + GDN INT8 + Head INT4 |

**问题：GPU+NPU 混合反而比 GPU 单独慢 25-35%。**

## 2. 时间都花在哪

Profiling 实测（含 profiling overhead ~30%）：

```
HYBRID 每 token 拆解 (74ms with profiling, ~59ms without):

  GDN block × 6 (GPU):    51ms  ████████████████████████████████████  69%
  Attn block × 6 (NPU):   16ms  ███████████                          22%
  Head (GPU):              6.5ms ████                                  9%
  Embed + 胶水:            0.5ms                                       1%

GPU_ONLY 每 token 拆解 (77ms with profiling, ~44ms without):

  GDN block × 6 (GPU):    50ms  ████████████████████████████████      65%
  Attn block × 6 (GPU):   21ms  ██████████████                        27%
  Head (GPU):              6.3ms ████                                   8%
```

NPU 的优势：Attn 21ms(GPU) → 16ms(NPU)，每 token 省 5ms。

## 3. 为什么 HYBRID 反而慢

### 3.1 NPU KV Cache 传输开销

GPU stateful 模式下，KV cache 常驻 GPU 显存，`infer()` 只传 hidden (4KB)。

NPU 必须用 explicit I/O（NPU 不支持 stateful + ScatterUpdate），每次 `infer()` 需要传入传出完整 KV cache：

| past_seq 长度 | 单个 Attn block KV 传输 | 6 个 block 总计 | 方向 |
|--------------|----------------------|---------------|------|
| 64 | 256 KB | 1.5 MB | 单向 |
| 128 | 512 KB | 3.0 MB | 单向 |
| 256 | 1024 KB | 6.0 MB | 单向 |

**每 token 来回传输 = 6-12 MB**（输入 + 输出），随序列增长线性增加。

在 Lunar Lake 统一内存上 memcpy 几乎零成本（cache coherent，非 PCIe），但 NPU 插件在 `set_input_tensor` / `get_output_tensor` 时有内部开销（格式转换、DMA 对齐等）。

### 3.2 串行执行，两个设备互相等待

```
时间线：
  GPU: ████ gdn_0 ░░░░░ idle ████ gdn_1 ░░░░░ idle ████ gdn_2 ...
  NPU: ░░░░ idle  ██ a_0 ░░░░ idle  ██ a_1 ░░░░ idle  ██ a_2 ...
                   ↑                  ↑
              GPU 等 NPU 完成     GPU 等 NPU 完成
```

GPU 利用率 ~70%，NPU 利用率 ~22%。两个设备大部分时间在等对方。

### 3.3 Async Pipeline 不可行

数据依赖是严格的：`gdn_i → attn_i → gdn_{i+1}`。每个 block 需要上一个 block 的 hidden 输出。无法 overlap：

- `attn_i` 必须等 `gdn_i` 完成才能开始
- `gdn_{i+1}` 必须等 `attn_i` 完成才能开始
- `start_async()` 最多省 ~0.5ms 胶水代码时间

## 4. 内存带宽分析

### 4.1 权重读取量

Decode 时每生成 1 token，需要读取的模型权重：

| 组件 | FP16 | 量化后 (INT8/INT4) |
|------|------|-------------------|
| GDN blocks × 6 | 776 MB | 388 MB (INT8) |
| Attn blocks × 6 | 220 MB | 55 MB (INT4) |
| Head | 509 MB | 127 MB (INT4) |
| **Total per token** | **1505 MB** | **570 MB** |

### 4.2 理论上限 vs 实际

| | 每 token 读取 | 理论上限 (68.3 GB/s) | 50% 利用率 | **实测** |
|---|---|---|---|---|
| FP16 (GPU_ONLY) | 1505 MB | 45 tok/s | 22.7 tok/s | **22.6 tok/s** ✓ |
| FP16 (HYBRID) | 1505 MB | 45 tok/s | 22.7 tok/s | **17 tok/s** ✗ |
| 量化 (HYBRID) | 570 MB | 120 tok/s | 60 tok/s | **14.5 tok/s** ✗ |

**GPU_ONLY FP16 已经达到 ~50% 带宽利用率，这是合理的上限。**

**HYBRID 量化后只用了 8.3 GB/s (12% 带宽)，严重浪费。** 量化减小了权重体积，但 NPU explicit I/O 开销 + 串行执行成为新的瓶颈。

## 5. 核心问题

### 问题 1：NPU 能跑 stateful 模式吗？

如果 NPU attention 能用 stateful（KV cache 常驻 NPU 内存），消除每 token 6-12MB 传输：
- 预计 HYBRID 可从 17 → 20+ tok/s
- 但 OpenVINO NPU 插件 stateful 模式有限制：
  - NPU stateful 实测比 explicit 慢 ~2x（ScatterUpdate: 13.5 vs 17.6 tok/s）
  - ReadValue + Assign 管理开销 > memcpy 开销
  - 长序列后偶有精度偏差

### 问题 2：GPU 和 NPU 能真正并行吗？

当前串行依赖链 `gdn_0 → attn_0 → gdn_1 → ...` 无法 overlap。要实现并行需要**架构层面改变**：

- **方案 A：Layer-level 并行** — 如果能把 GDN 和 Attn 拆成不依赖的子任务（但它们处理同一个 hidden state，天然串行）
- **方案 B：Token-level pipeline** — Token N 的 attn_i 和 Token N+1 的 gdn_j 并行（但 autoregressive decode 每个 token 依赖上一个 token 的输出）
- **方案 C：Speculative decode** — GPU 跑 draft model 猜多个 token，NPU 验证。但 0.8B 模型已经很小，没有更小的 draft model

### 问题 3：GDN 能在 NPU 上跑吗？

如果全部 24 层都在 NPU，GPU 完全空闲，这是最理想的场景。但目前两个障碍：

1. **FP16 精度不足** — GDN delta rule 线性递归 `state = state * g + k^T * δ`，每步乘法放大舍入误差，18 层串联后发散
2. **NPU 不支持 Loop 节点** — GDN 递归在 IR 中表示为 Loop op，NPU 编译器报 `to_shape was called on a dynamic shape`

对比：高通 SM8650 NPU 有 FP32 累加器 + TFLite 编译时展开 Loop，GDN 可正常推理。

### 问题 4：能否绕过 OpenVINO 的开销？

每次 `infer()` 调用有固定开销（~0.5-1ms）：plugin dispatch、tensor validation、kernel launch。13 个子图 × 13 次 `infer()` = 6.5-13ms 纯开销。

- 能否把多个子图 fuse 成一个？（OpenVINO 不支持跨 Loop 的子图合并）
- 能否直接调用底层 kernel？（Level Zero API？oneAPI？）
- C++ 实现已经消除了 Python 开销，但 OV runtime 开销仍在

## 6. 想请教的问题

1. **NPU stateful + ScatterUpdate** 在 OpenVINO 2026.0 上的支持情况？我们测试发现 NPU stateful 比 explicit 慢 2x，这是已知问题还是用法不对？

2. **NPU 上 Loop 节点的支持计划？** 如果 NPU 能编译时展开 Loop（类似 TFLite），GDN 就能全跑 NPU，彻底消除 GPU↔NPU 串行问题

3. **NPU FP32 累加器**？Lunar Lake NPU 是否有 FP32 累加模式？GDN 递归核心 `state * gate` 在 FP16 精度下会累积误差。如果 NPU 能用 FP32 做乘累加（哪怕输入输出是 FP16），精度问题就解决了

4. **GPU+NPU 同时访问内存的带宽表现？** Lunar Lake 统一内存架构下，GPU 和 NPU 同时读内存，实际带宽是各占一半(34+34)还是会互相干扰降到更低？

5. **有没有 NPU 的 low-level profiling 工具？** 想知道 NPU `infer()` 的 16ms 里，实际计算 vs DMA 传输 vs kernel launch 各占多少

6. **INT4 在 0.8B 模型上对 NPU 性能的影响？** CLAUDE.md 里写"≤1B 模型 INT4 反而更慢（反量化开销 > 带宽节省）"，但我们实测 Attn INT4 没有明显减速也没有加速。这是预期行为吗？INT4 反量化是在 NPU 片上做的还是软件模拟？

## 7. 总结

| 指标 | GPU_ONLY | HYBRID | 差距原因 |
|------|----------|--------|---------|
| Decode | 22.6 tok/s | 14.5-17 tok/s | NPU I/O + 串行等待 |
| 带宽利用率 | ~50% (34 GB/s) | ~12% (8.3 GB/s) | GPU 和 NPU 互相等着 |
| 权重量化收益 | 直接→更快 | 被 I/O 开销淹没 | 量化减小权重但不减 KV 传输 |
| 瓶颈 | 内存带宽 | NPU I/O + 串行调度 | 不同瓶颈 |

**当前结论**：对于 Qwen3.5 GDN 混合架构，GPU_ONLY 是最快的。HYBRID 只在 GPU 不够用（比如同时跑其他模型占用 GPU）时才有意义。要让 HYBRID 超过 GPU_ONLY，需要解决 NPU stateful 性能或 Loop 节点支持。

## 8. AMD XDNA Hybrid 对比 — 为什么 AMD 能做到而 Intel 目前不行

AMD Ryzen AI (Strix Point) 用 **ONNX Runtime GenAI (OGA) Hybrid 模式**实现 NPU+iGPU 同时加速，社区实测 hybrid 比 pure iGPU 快 10-30%。而我们 Intel Lunar Lake 的 hybrid 反慢 25-35%。

### AMD 的核心设计差异

| 维度 | AMD OGA Hybrid | Intel OpenVINO Hybrid (我们) |
|------|---------------|---------------------------|
| **分区方式** | 框架自动分区（OGA runtime 根据拓扑+配置动态拆） | 手动拆 13 个子图，13 次 `infer()` 调用 |
| **并行策略** | **Phase-level**：NPU 跑 prefill，iGPU 跑 decode | **Layer-level**：GDN→GPU, Attn→NPU（同一 token 内串行） |
| **KV Cache** | OGA 内置 stateful 管理，常驻设备内存 | NPU 必须 explicit I/O，每 token 传 6-12MB |
| **量化** | Quark INT4 对齐 XDNA 硬件，片上反量化 | NNCF INT4/8，NPU 上 INT4 无加速也无减速 |
| **Loop/递归** | XDNA 支持（TFLite 编译时展开） | NPU 不支持 Loop 节点 |
| **Overhead** | 单 API 调用，内部 orchestration | 13 × infer() 固定开销 ~6-13ms |

### 关键启发

**1. Phase-level 比 Layer-level 更实际**

AMD 的思路：prefill 阶段（大 batch，compute-bound）→ NPU 擅长；decode 阶段（单 token，memory-bound）→ iGPU 擅长。两个阶段天然解耦，无串行依赖。

我们的思路：同一 token 内 GDN→GPU + Attn→NPU。但数据依赖链 `gdn_i → attn_i → gdn_{i+1}` 导致串行等待，两个设备互相空转。

**如果要在 Intel 上做 phase-level：**
- Prefill：全部 24 层跑 GPU（已验证 GPU_ONLY 能跑）
- Decode：全部 24 层跑 NPU（需要 NPU 支持 Loop + FP32 累加器）
- 或者：Prefill 重 NPU（Attn 部分），Decode 全 GPU
- 问题：GDN 不能在 NPU 上跑，所以 phase-level 也无法完全分离

**2. Stateful KV Cache 是关键**

AMD NPU 的 stateful 是真正可用的（KV 常驻，无 per-token 传输），而 Intel NPU stateful 实测慢 2x。这一项差异就解释了大部分性能差距。如果 Intel NPU stateful 能修好，我们的 HYBRID 可能从 17 → 20+ tok/s。

**3. 自动化 vs 手动分区**

AMD 的 OGA 把分区、KV 管理、设备调度全自动化了。我们用 OpenVINO 手动管理 13 个子图 + 显式 I/O + 手写调度，不仅开发成本高，而且每增加一层 overhead 都是固定 per-token 惩罚。

### 对 Intel 的建议

短期（OpenVINO 2026.x）：
- **修复 NPU stateful 性能**（当前 explicit > stateful 是反直觉的）
- **减少 NPU infer() 固定开销**（当前 ~0.5-1ms/call × 6 = 3-6ms 纯调度成本）
- **NPU INT4 反量化硬件加速确认**（当前 INT4 无加速，怀疑是软件模拟）

中期（需要硬件/编译器支持）：
- **NPU Loop 节点支持**（编译时展开，类似 TFLite）→ GDN 能上 NPU
- **NPU FP32 累加器**（或混合精度：输入 FP16，累加 FP32）→ GDN 递归精度问题解决
- **类 OGA 的高层 Hybrid API**（自动分区 + phase-level 调度 + stateful KV 管理）

如果 Loop + FP32 累加器都支持，理论上全部 24 层跑 NPU + GPU 只跑 Head，decode 可以完全 memory-bound 在 NPU 侧，GPU 空出来给其他任务。
