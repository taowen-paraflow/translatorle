# Translatorle — 语音识别 + 机器翻译桌面应用

## 环境

- **OS**: Windows 11 + WSL2
- **Python**: 3.12, 用 `uv` 管理依赖
- **uv 路径**: `C:\Users\taowen\.local\bin\uv.exe`
- **推理硬件**: Intel NPU (OpenVINO + openvino_genai)

## 从 WSL2 调用 Windows 命令

在 WSL2 中运行 Windows 端的 Python/uv 时，需要用 PowerShell 启动：

```bash
powershell.exe -Command 'cd C:\Apps\translatorle; C:\Users\taowen\.local\bin\uv.exe run python -m app'
```

设置环境变量的写法：

```bash
powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; cd C:\Apps\translatorle; C:\Users\taowen\.local\bin\uv.exe run python ...'
```

## 项目结构

| 目录 | 说明 |
|------|------|
| `asr/` | 流式语音识别（Qwen3-ASR, NPU 推理） |
| `hymt/` | 机器翻译（HY-MT1.5-1.8B, NPU 推理） |
| `app/` | PySide6 桌面 GUI |
| `models/` | OpenVINO IR 模型文件（不入 git） |

各模块有独立的 CLAUDE.md，详见 `asr/CLAUDE.md`、`hymt/CLAUDE.md`、`app/CLAUDE.md`。

## NPU 量化经验

### 小模型（≤1B）不要过度量化

- ASR Decoder (0.6B params) 在 NPU 上 **FP16 比 INT4 快 1.6x**（40.7 vs 24.6 tok/s）
- 原因：小模型 FP16 已能充分利用 NPU 带宽，INT4 反量化开销 > 带宽节省
- **结论：≤1B 模型在 NPU 上保持 FP16 即可**

### 大模型（≥1.5B）必须量化到 INT4_SYM

- NPU 上 LLM 推荐：`INT4_SYM` + `group_size=128` + `ratio=1.0`
- **NPU 不支持 `INT4_ASYM`**，会回退到慢路径（1.2 tok/s vs 29 tok/s）
- HY-MT 1.8B: INT4_ASYM → INT4_SYM 后速度提升 **23.4x**（1.2 → 29.0 tok/s）
- `NF4` 仅 Lunar Lake (Core Ultra 200V) 及以上支持
- Optimum-Intel 导出时必须指定 `--symmetric`，默认 `load_in_4bit=True` 会用 ASYM

### nncf.compress_weights() 用法

可直接对已导出的 OpenVINO IR 模型做权重压缩，无需原始 PyTorch 模型：

```python
import openvino as ov
import nncf
model = ov.Core().read_model("model.xml")
compressed = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT4_SYM, group_size=128, ratio=1.0)
ov.save_model(compressed, "model_int4.xml")
```

### Optimum-Intel 导出时指定对称量化

```bash
optimum-cli export openvino --model <model> --task text-generation-with-past \
  --weight-format int4 --group-size 128 --symmetric ov_model/
```

或在代码中：
```python
from optimum.intel import OVWeightQuantizationConfig
quant_config = OVWeightQuantizationConfig(bits=4, sym=True, group_size=128, ratio=1.0)
```

### Dynamic Quantization（Core Ultra Series 2 / Lunar Lake+）

运行时激活值动态量化为 INT8，减少带宽压力：

- NPU 编译器参数：`NPU_COMPILER_DYNAMIC_QUANTIZATION=YES`
- 可与 QDQ 优化结合：`NPU_QDQ_OPTIMIZATION=YES`
- **实测 HY-MT 1.8B：Dynamic Quant 反而慢 31%**（38.5 → 26.5 tok/s）
- 小模型（≤2B）INT4_SYM 已饱和带宽，动态量化只增加开销
- 可能对更大模型（7B+）有效，待验证

### NPU 量化决策总结

| 模型规模 | 推荐精度 | 实测结果 |
|---------|---------|---------|
| ≤1B | FP16 | ASR 0.6B: 40.7 tok/s (INT4 反而慢) |
| 1-2B | INT4_SYM (group=128) | HY-MT 1.8B: 38.5 tok/s |
| 1-2B + DynQuant | 不推荐 | HY-MT 1.8B: 26.5 tok/s (慢 31%) |
| ≥7B (Lunar Lake+) | INT4_SYM + dynamic quant | 理论最优，待验证 |

## 多设备性能对比

### Benchmark Results (2026-03-08)

**ASR Encoder** (FP16, conv model, input [1,128,800]):
| Device | Avg Inference |
|--------|-------------|
| CPU | 405.8 ms |
| GPU | **11.4 ms** |
| NPU | 24.3 ms |

**ASR Decoder** (FP16, 0.6B LLM, prefill 128 tokens + decode 20 tokens):
| Device | Prefill | Decode tok/s |
|--------|---------|-------------|
| CPU | 1.23 s | 21.4 |
| GPU | 75 ms | 42.7 |
| NPU | 85 ms | 36.6 |

**HY-MT** (INT4_SYM, 1.8B LLM, 翻译一句中文→英文):
| Device | Avg Translation |
|--------|----------------|
| CPU | 899 ms |
| GPU | 661 ms |
| NPU | 1.27 s |

### Key Findings:
- GPU 全面最快，甚至 LLM decode 也比 NPU 快 (42.7 vs 36.6 tok/s)
- NPU 对 MT 1.8B 反而最慢 (1.27s vs GPU 661ms)，可能是 INT4_SYM 在 NPU 上的 prefill 效率低
- CPU 的 encoder 极慢 (406ms vs GPU 11ms)，不适合实时 ASR
- ASR encoder 是 conv 模型，GPU 优势最大 (35x faster than CPU)

### 设备分配策略（边 ASR 边翻译）
- **当前配置**: ASR(encoder+decoder) → NPU, HY-MT → GPU
- 原因: NPU 独占无法同时跑两个模型，GPU 可独立处理 MT
- ASR on NPU: encoder 24ms + decoder 36.6 tok/s，满足实时需求
- MT on GPU: 661ms/句，翻译速度最快

### MT KV Cache 复用（Session 模式）

使用 `openvino_genai.LLMPipeline` 的 `start_chat()` / `finish_chat()` API 在多句翻译间复用 KV cache，避免每句都重新 prefill 整个上下文。

**API 用法**：
```python
engine = MTEngine(device="GPU")
engine.start_session("English")        # start_chat(system_prompt)
result = engine.translate("你好世界。")  # 只 prefill 新句子
result = engine.translate("今天天气好。") # 复用之前的 KV cache
engine.finish_session()                 # finish_chat() 释放 KV cache
```

**自动 token 限制保护**：`needs_reset(max_tokens=400)` 在累积 token 接近 `MAX_PROMPT_LEN`(512) 时触发 `reset_session()`，一次性清空 KV cache 重建会话。

**Benchmark (GPU, 10 句中→英, `bench_session.py`)**：

|  | Stateless | Session | Speedup |
|--|-----------|---------|---------|
| 平均每句 | 335 ms | 285 ms | 1.18x |
| 总耗时 | 3.36 s | 2.85 s | 1.18x |
| 延迟趋势 (末/首) | 1.05x (变慢) | 0.84x (稳定) | — |

- 10 句级别加速约 18%，主要收益是消除了 prefill 重复
- **长会话收益更大**：上下文越长，stateless 的 prefill 开销越高，session 模式始终只 prefill 新句子
- 第 8 句时自动 reset 生效，防止超出 512 token 限制
