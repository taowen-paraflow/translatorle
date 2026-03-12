# Qwen3-0.6B Benchmark — Intel Core Ultra 7 258V (Lunar Lake)

模型：Qwen3-0.6B（标准 Transformer，FP16），OpenVINO GenAI LLMPipeline

硬件：Intel Core Ultra 7 258V — NPU 4 / Intel Arc GPU / CPU (4P+4E)

## 256 tokens

| 指标 | NPU | GPU | CPU |
|------|-----|-----|-----|
| 加载时间 | 38.2s | 11.2s | 3.0s |
| 生成 tokens | 256 | 256 | 256 |
| 总耗时 | 7.7s | 7.0s | 16.3s |
| **吞吐量** | **~33 tok/s** | **~37 tok/s** | **~16 tok/s** |

NPU 首次加载含编译时间（~38s），缓存后约 6-7s。

## 4096 tokens (`--ignore-eos`)

| 指标 | NPU | GPU | CPU |
|------|-----|-----|-----|
| 加载时间 | 6.5s | 11.3s | 4.6s |
| 实际生成 tokens | 3744 | 3574 | 3672 |
| 总耗时 | 234.7s | 244.3s | 492.2s |
| TTFT | 301ms | 446ms | 3426ms |
| **吞吐量** | **16.0 tok/s** | **14.6 tok/s** | **7.5 tok/s** |
| 解码速度 | 16.0 tok/s | 14.7 tok/s | 7.5 tok/s |

## 观察

1. **短序列 GPU 略快，长序列 NPU 略快** — 256 tokens 时 GPU 37 > NPU 33 tok/s；4096 tokens 时 NPU 16 > GPU 14.7 tok/s
2. **长序列速度约为短序列的一半** — KV-cache 增长导致内存带宽瓶颈（0.6B FP16 模型本身不大，瓶颈在 KV-cache 读写）
3. **NPU TTFT 最低**（301ms），GPU 446ms，CPU 3.4s — NPU prefill 优势明显
4. **CPU 速度约为 NPU/GPU 的一半**（7.5 vs 15-16 tok/s）
5. 输出内容正确，三设备一致 — 标准 Transformer FP16 在 NPU 上无精度问题

## 运行方式

```bash
# 256 tokens (默认)
uv run python -m qwen3.scripts.benchmark --device NPU

# 4096 tokens, 强制生成到上限
uv run python -m qwen3.scripts.benchmark --device NPU --max-new-tokens 4096 --ignore-eos

# 可选设备: NPU, GPU, CPU
uv run python -m qwen3.scripts.benchmark --device GPU --max-new-tokens 4096 --ignore-eos
```
