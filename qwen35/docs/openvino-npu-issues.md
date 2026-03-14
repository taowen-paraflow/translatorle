# OpenVINO NPU 问题清单 — Qwen3.5-0.8B GDN 混合架构

## 背景

硬件：Intel Core Ultra 7 258V (Lunar Lake)，UMA 统一内存，iGPU + NPU。
模型：Qwen3.5-0.8B，Gated Delta Networks 混合架构（18 层 GDN linear attention + 6 层 full attention）。
当前方案：GDN → GPU，Attention → NPU，串行执行。Decode ~24 tok/s。
对比：llama.cpp Vulkan（fused GDN kernel）在同硬件 69 tok/s。

## 问题 1：NPU 不支持 Loop 节点

GDN 的递归计算在 OpenVINO IR 中表示为 Loop 节点（动态控制流）。NPU 编译器报错：

```
to_shape was called on a dynamic shape
```

**已绕过**：导出 noloop 版本（`SingleStepRecurrentAttentionCell`），把 Loop 展开为扁平的 MatMul/Exp/Add ops。NPU 能编译和运行，但精度不行（见问题 2）。

**期望**：NPU 编译器能处理 Loop 节点，或自动将已知 trip-count 的 Loop 展开。

## 问题 2：NPU FP16 精度不足，无 FP32 累加选项

GDN delta rule 递推：`state = state * exp(decay) + outer(key, value)`，每步乘法累积舍入误差。

NPU 全 FP16 计算，18 层 × 多 token 后误差发散。实测 noloop GDN 在 NPU 上第 1-3 个 decode token 即偏离 GPU baseline，输出陷入重复循环。

GPU 可以通过 `INFERENCE_PRECISION_HINT=f32` 保证递推精度。NPU 没有类似选项。

**对比**：高通 SM8650 NPU 有 FP32 累加器，GDN 可在 NPU 正常推理。

**期望**：NPU 提供 `INFERENCE_PRECISION_HINT` 或 per-op 精度控制（至少对累加/递推类 op 支持 FP32）。

## 问题 3：每次 infer() 调度开销 ~1ms

当前 GDN noloop 每 block 有 ~24 个独立 op（MatMul/Exp/Add），6 blocks = 144 个 op，但每 block 是一个完整子图，每次 `infer()` 调用有 ~1ms 的 runtime 开销。

对比 llama.cpp：整个 GDN 递推是单个 fused Vulkan compute shader，一次 dispatch。

这个开销在 GPU 和 NPU 上都存在。对于 S=1 decode 这种微小计算量（单个 MatMul 128×128，~33K FLOPs），runtime 开销占比很大。

**问题**：有没有办法减少 `infer()` 的调度开销？比如：
- 多个子图合并执行（batch inference）
- 预编译执行计划，减少每次调用的验证/同步开销
- 类似 CUDA Graph 的机制，录制一次执行序列后重放

## 问题 4：无法写自定义 NPU kernel

GPU plugin 支持 `add_extension()` 注册自定义 OpenCL kernel。NPU 没有类似机制。

如果能为 NPU 写一个 fused GDN kernel（把整个递推 state update 合成一个 op），理论上可以：
- 消除 op 间的调度开销
- 在 NPU 内部用寄存器/SRAM 保存中间结果，避免反复读写 DDR
- 可能支持混合精度（FP32 累加 + FP16 存储）

**期望**：NPU plugin 提供自定义 kernel 扩展机制，或者至少提供 fused op 注册接口。

## 问题 5：NPU↔GPU 无法真正并行

Lunar Lake 是 UMA，物理内存共享，数据传输接近零成本。但 OpenVINO 的 `infer()` 是同步阻塞的。

当前执行：
```
GPU: [GDN_0]         [GDN_1]         [GDN_2]
NPU:         [Attn_0]        [Attn_1]        [Attn_2]
```

即使用 `start_async()`，也无法让 GPU 和 NPU 真正同时执行独立子图，因为：
- 数据依赖：Attn_i 需要 GDN_i 的输出（这是模型结构决定的，不是框架问题）
- 但 prefill 阶段可能有机会：如果能让 GPU 执行 GDN_i 的同时 NPU 执行 Attn_{i-1}（pipeline），需要框架支持异步 + 低开销同步

**问题**：`start_async()` + `wait()` 在 GPU 和 NPU 之间的实际并行度如何？有没有 pipeline 执行的最佳实践？

## 问题 6：NPU set_output_tensor 零拷贝导致 2x 减速

尝试对 NPU attention block 用 `set_output_tensor()` 实现零拷贝（input/output 指向同一 host buffer 的不同区域），结果 decode 从 ~17 tok/s 降到 ~9 tok/s。

原因推测：NPU plugin 检测到 input/output aliased memory，内部增加了额外拷贝来保证正确性。

GPU plugin 的 `set_output_tensor()` 没有这个问题（GPU 用独立显存做中间计算）。

**期望**：NPU plugin 在 UMA 架景下优化 aliased buffer 处理，或提供显式的零拷贝 API。

## 问题 7：NPU Stateful 比 Explicit I/O 慢 2x

NPU 上 ReadValue/Assign（stateful）比手动传入传出 tensor（explicit I/O）慢约 2x：
- Stateful: 13.5 tok/s
- Explicit: 17.6 tok/s

推测 ReadValue/Assign 在 NPU 上有额外的状态管理开销。

**期望**：NPU stateful 性能与 explicit I/O 持平，或者提供使用指南说明何时该用 stateful。

## 总结

| 问题 | 影响 | 严重程度 |
|------|------|---------|
| 不支持 Loop | GDN 无法直接在 NPU 编译 | 已绕过（noloop） |
| 无 FP32 精度控制 | GDN 递推在 NPU 上发散 | **致命** — 阻止 NPU 跑 GDN |
| infer() 调度开销高 | 小算子密集调用时成为瓶颈 | 高 |
| 无自定义 kernel | 无法写 fused op 优化 | 高 |
| 跨设备无真正并行 | GPU+NPU 只能串行 | 中 |
| set_output_tensor 减速 | 零拷贝不可用 | 低（有 workaround） |
| Stateful 慢 | 增加 ~2x 开销 | 低（用 explicit I/O） |

核心诉求：**NPU 能否提供 FP32 累加（解决精度）+ 降低 per-infer 开销（解决性能）？** 这两个解决了，GDN 就能完全跑在 NPU 上，释放 GPU 给其他任务。
