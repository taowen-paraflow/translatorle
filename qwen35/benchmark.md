# Qwen3.5-0.8B Benchmark Results

**Hardware**: Intel Core Ultra 7 258V (Lunar Lake) — iGPU Xe2 8-core + NPU 4th gen
**Memory**: LPDDR5X-8533, 峰值带宽 68.3 GB/s（双通道）
**Framework**: OpenVINO 2026.0
**Date**: 2026-03-14

---

## Decode 性能对比

**Prompt**: "The capital of France is" (5 tokens) → 80 tokens greedy decode
**每组 3 轮，轮间 30 秒冷却**

### 总览

| 配置 | 引擎 | 拆图 | 设备 | Run 1 | Run 2 | Run 3 | **平均** |
|------|------|------|------|-------|-------|-------|---------|
| **量化 HYBRID** | **C++** | **19 子图** | **GPU+NPU** | 7.0 | 16.1 | 16.7 | **13.3** |
| 量化 HYBRID | Python | 19 子图 | GPU+NPU | 6.8 | 11.4 | 7.3 | **8.5** |
| Single-IR INT4 | Python | 单 IR | GPU | 7.4 | 10.4 | 8.6 | **8.8** |
| Single-IR FP16 | Python | 单 IR | GPU | 5.9 | 9.7 | 6.7 | **7.4** |

**关键发现**：
1. **C++ HYBRID 冷态 16+ tok/s** — 比 Single-IR GPU 快 1.6-1.9x
2. **热节流影响巨大** — Run 1 普遍偏低（前一组测试余热），Run 2 冷却后恢复
3. **Single-IR INT4 vs FP16** — INT4 量化带来 ~19% 提升 (8.8 vs 7.4)
4. **C++ vs Python HYBRID** — C++ 快 1.5-1.9x（零分配热循环 + 无 Python GIL）

### 各配置详细数据

#### C++ HYBRID 量化（19 子图，GPU+NPU）

最优配置：`Qwen3.5-0.8B-hybrid-attn-int4sym-gdn-int8sym-head-int4sym` (3.4 GB)

| 指标 | Run 1 | Run 2 | Run 3 | **平均** |
|------|-------|-------|-------|---------|
| Decode tok/s | 7.0 | 16.1 | 16.7 | **13.3** |
| ms/token | 142.7 | 62.1 | 60.1 | **88.3** |
| Prefill ms | 1829 | 961 | 892 | **1227** |

Run 1 受前序 benchmark 余热影响严重（7 tok/s），Run 2/3 冷态稳定在 16+ tok/s。

#### C++ HYBRID 量化（--timing 模式，含分解）

| 组件 | 每 token 耗时 | 占比 | 设备 |
|------|-------------|------|------|
| **GDN blocks × 6** | **63.2 ms** | **79.6%** | GPU |
| Attn blocks × 6 | 12.6 ms | 15.8% | NPU |
| Head | 3.5 ms | 4.4% | GPU |
| **Total** | **79.3 ms** | **100%** | — |

GDN 各 block 耗时 (ms/token)：

| Block | 0 | 1 | 2 | 3 | 4 | 5 |
|-------|---|---|---|---|---|---|
| ms | 10.9 | 10.3 | 10.0 | 10.5 | 10.6 | 10.9 |

Attn 各 block 一致在 **2.0-2.1 ms/token**（NPU 非常稳定）。

#### Python HYBRID 量化（19 子图，GPU+NPU）

| 指标 | Run 1 | Run 2 | Run 3 | **平均** |
|------|-------|-------|-------|---------|
| Decode tok/s | 6.8 | 11.4 | 7.3 | **8.5** |
| ms/token | 146.4 | 87.4 | 137.2 | **123.7** |

#### Single-IR GPU INT4（不拆图，纯 GPU baseline）

模型：`Qwen3.5-0.8B-ov-int4`，单个 `openvino_model.xml`，stateful IR，Python `GenerationMixin`

| 指标 | Run 1 | Run 2 | Run 3 | **平均** |
|------|-------|-------|-------|---------|
| Decode tok/s | 7.4 | 10.4 | 8.6 | **8.8** |
| ms/token | 135.1 | 96.2 | 116.3 | **115.9** |

#### Single-IR GPU FP16（不拆图，纯 GPU baseline）

模型：`Qwen3.5-0.8B-ov`，单个 `openvino_model.xml`，stateful IR，Python `GenerationMixin`

| 指标 | Run 1 | Run 2 | Run 3 | **平均** |
|------|-------|-------|-------|---------|
| Decode tok/s | 5.9 | 9.7 | 6.7 | **7.4** |
| ms/token | 169.5 | 103.1 | 149.3 | **140.6** |

---

## 热节流分析

Lunar Lake iGPU 有严重热节流问题：

- **C++ HYBRID**: 冷态 16.7 vs 热态 7.0 tok/s（**2.4x 差异**）
- **Python HYBRID**: 冷态 11.4 vs 热态 6.8 tok/s（1.7x 差异）
- **Single-IR GPU**: 冷态 10.4 vs 热态 7.4 tok/s（1.4x 差异）
- **30 秒冷却有效** — Run 2 (冷却后) 一致优于 Run 1/3
- **模型编译产生余热** — 每次启动编译 19 子图需数秒 GPU 计算
- **前序 benchmark 余热** — Run 1 若紧接前一组测试，性能显著下降
- **建议**: 连续推理以 sustained 性能为准，不应以 peak 作为标称值

---

## Prefill 时间分解（C++ HYBRID avg）

| 组件 | 耗时 | 占比 |
|------|------|------|
| **GDN prefill** | ~750 ms | **~79%** |
| Attn prefill (NPU, chunked) | ~150 ms | ~16% |
| Head | ~20 ms | ~2% |

Prefill 也是 GDN 主导。Chunkwise parallel GDN prefill（无 Loop，MatMul-based）已是最优。

---

## 可用模型变体

| 目录 | 类型 | 量化 | 大小 | 状态 |
|------|------|------|------|------|
| `Qwen3.5-0.8B-ov` | 单 IR | FP16 | ~1.5 GB | GPU baseline |
| `Qwen3.5-0.8B-ov-int4` | 单 IR | INT4 | ~0.5 GB | GPU INT4 baseline |
| `Qwen3.5-0.8B-hybrid` | 19 子图 | FP16 | 5.5 GB | Hybrid baseline |
| `...-attn-int4sym-gdn-int8sym-head-int4sym` | 19 子图 | INT4+INT8+INT4 | 3.4 GB | **推荐** |

---

## 已探索的优化方向

### 有效优化（已集成）

| 优化 | 效果 | 说明 |
|------|------|------|
| GDN stateful (GPU VRAM 常驻) | +58% | 消除每 token 41MB+ 传输 |
| Attn → NPU | +15-25% | NPU 2.4ms/block vs GPU 3.5ms |
| ScatterUpdate-3 KV cache | +66% (GPU_ONLY) | ConversionExtension 最优 KV 更新 |
| Chunked prefill (S=16/8/4/2) | +38% | NPU 多 shape 编译 |
| Layer-major prefill | +30-40% | GDN full-batch + Head 只跑最后 token |
| Chunkwise parallel GDN prefill | +100% (prefill) | WY representation 消除 Loop |
| 权重量化 (INT4+INT8) | +7-20% | 减少 GPU 带宽占用 |
| C++ 零分配热循环 | +50-90% vs Python | 增量 mask、无 timing 开销、内联 stop-id |

### 无效/负面优化（已验证放弃）

| 优化 | 结果 | 原因 |
|------|------|------|
| GPU-side argmax | -28ms/tok | iGPU argmax kernel 对 248K vocab 极慢 |
| Embedding 上 GPU | 30x 慢 | GPU Gather 开销 >> CPU numpy |
| NPU set_output_tensor 零拷贝 | 2x 减速 | aliased buffer 触发额外拷贝 |
| GPU attention for decode | 2.2x 减速 | iGPU 资源争抢 |
| LATENCY hint (Python) | -30% | Python OV API 不兼容 |
| Prefill 后释放 GPU 内存 | 无效 | GPU memory pressure 假说不成立 |
| Loop-free S=1 GDN decode | -21% | GPU Loop body fusion > Loop 管理开销 |
| MTP speculative decoding | -17~30% | 56% 接受率远低于 87% 盈亏平衡点 |

### 瓶颈总结

GDN blocks 占 decode ~80%，是唯一有意义的优化目标。当前约束：
- 必须 FP32（递归精度）
- 必须 GPU（NPU 不支持 Loop + FP16 精度不足）
- 已经 INT8_SYM（进一步量化需验证）
- 每 block ~10.5ms × 6 = 63ms（冷态）

### 潜在方向

| 方向 | 预期 | 风险 |
|------|------|------|
| **GDN INT4 量化** | ~1.5-2x GDN 加速 | 递归精度可能受损 |
| Intel NPU Loop 支持 (未来) | GDN 可上 NPU | 等待硬件/编译器更新 |
| NPU FP32 累加器 (未来) | 解决精度问题 | 等待硬件更新 |
