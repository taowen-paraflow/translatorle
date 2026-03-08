# ASR 模块 — Qwen3-ASR 流式语音识别

支持 Qwen3-ASR-0.6B 和 1.7B 两种模型，通过 `ASRModelConfig` 配置切换。

## 快速开始

### 准备模型

```powershell
$env:PYTHONIOENCODING = "utf-8"
# 0.6B（导出到 models/）
uv run python asr/scripts/prepare_asr_models.py --model-id Qwen/Qwen3-ASR-0.6B
# 1.7B（导出到 models/asr_1.7b/）
uv run python asr/scripts/prepare_asr_models.py --model-id Qwen/Qwen3-ASR-1.7B
```

### 代码调用

```python
from asr import ASREngine
from asr.config import ASR_MODELS

engine = ASREngine(model_config=ASR_MODELS["1.7B"])
state = engine.new_session()
engine.feed(pcm_chunk, state)   # 16kHz float32 PCM
engine.finish(state)
print(state.text)
```

设备由 `ASRModelConfig.encoder_device` / `decoder_device` 决定，也可通过参数覆盖。

## 多模型配置

`asr/config.py` 中定义 `ASRModelConfig` dataclass：

```python
@dataclass
class ASRModelConfig:
    encoder_xml: str
    decoder_xml: str
    embed_table_npy: str
    hf_model_dir: str
    npu_decoder_config: dict
    encoder_device: str = "NPU"
    decoder_device: str = "NPU"
```

预定义配置：
- `ASR_MODELS["0.6B"]` — encoder NPU, decoder NPU (FP16, folding 正常)
- `ASR_MODELS["1.7B"]` — encoder NPU, decoder NPU (INT4_SYM, NPUW_FOLD=NO 绕过 folding)
- `DEFAULT_ASR_MODEL = "1.7B"`

## 模型导出流水线

`asr/scripts/prepare_asr_models.py` 执行 8 个步骤：

```
Step 1: 加载 Qwen3ASRForConditionalGeneration
Step 2: 导出 Audio Encoder → encoder_fp16.xml
Step 3: 提取 Text Decoder 权重 (排除 audio_tower)
Step 4: 构建独立 Qwen3ForCausalLM checkpoint
Step 5: optimum-intel 导出 stateful decoder (KV-cache)
Step 6: IR 图手术 → 移除 input_ids, 添加 inputs_embeds
Step 7: 提取 embed_tokens.npy
Step 8: 拷贝 tokenizer + preprocessor_config
```

脚本自动检测模型维度，>1B 模型使用 INT4_SYM 量化，≤1B 保持 FP16。

### IR 图手术（Step 6）

NPUW_LLM 插件如果同时看到 `input_ids` 和 `inputs_embeds`，会优先选 `input_ids`（int64 零值），导致输出乱码。必须从 IR 中**移除 `input_ids` 参数**，让 NPUW 回退到 `inputs_embeds` 路径。

**NPU 兼容性**：移除 input_ids 后，0.6B folding 正常。1.7B 默认 folding 会 assert 失败（层结构不均匀 + inputs_embeds 3D 入口），通过 `NPUW_FOLD=NO` + `NPUW_LLM_PREFILL_HINT=STATIC` 绕过，每层独立编译（编译时间 ~460s，但推理正确且 RTF 0.30x）。

## 模型文件

```
models/
├── encoder_fp16.xml/.bin               # 0.6B encoder
├── decoder_stateful_embeds/            # 0.6B decoder (FP16) ★推理用
├── embed_tokens.npy                    # 0.6B embedding table
├── Qwen3-ASR-0.6B/                    # tokenizer
└── asr_1.7b/
    ├── encoder_fp16.xml/.bin           # 1.7B encoder
    ├── decoder_stateful_embeds/        # 1.7B decoder (INT4_SYM) ★推理用
    ├── embed_tokens.npy                # 1.7B embedding table
    └── Qwen3-ASR-1.7B/                # tokenizer
```

## 推理架构

```
PCM 16kHz → MelProcessor → [1,128,800] mel
                              ↓
                     OVEncoder → [1, 104, D] audio features
                              ↓
              构建 inputs_embeds (prompt tokens + audio features)
                              ↓
                     OVDecoder → greedy decode → text
```

### 流式策略：累积重编码 + 前缀回退

1. **累积音频**：新 chunk 追加到 buffer，每次重新编码全部累积音频
2. **前缀回退**：上一轮转写结果去掉末尾 5 个 token 作为 decoder 前缀，引导续写
3. **分段提交**：音频填满编码器窗口（~8s）时提交转写，开始新段

参数：`chunk_size_sec=2.0`, `unfixed_token_num=5`, `max_new_tokens=32`

### 模块说明

| 模块 | 类 | 职责 |
|------|----|------|
| `config.py` | `ASRModelConfig` | 模型配置、路径常量、NPU 配置 |
| `processor.py` | `MelProcessor` | PCM → log-mel 频谱 |
| `ov_encoder.py` | `OVEncoder` | OpenVINO audio encoder |
| `ov_decoder.py` | `OVDecoder` | OpenVINO stateful decoder (NPU/GPU/CPU) |
| `engine.py` | `ASREngine` | 流式引擎：累积重编码 + 前缀回退 |

### Prompt 格式 (ChatML)

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n<|audio_start|><|audio_pad|>×104<|audio_end|><|im_end|>\n
<|im_start|>assistant\n[prefix...]
```

`<|audio_pad|>` 位置在 `inputs_embeds` 中替换为 encoder 输出的音频特征向量。

## 性能参考

| 指标 | 0.6B (全 NPU) | 1.7B (全 NPU, FOLD=NO) | 1.7B (NPU enc + GPU dec) |
|------|-------------|----------------------|--------------------------|
| Encoder | 25.8 ms | 40.9 ms | 40.9 ms |
| Decoder tok/s | 28.5 | ~16.8 | 21.0 |
| 初始化时间 | ~8s | ~460s | ~24s |
| 实时性 (RTF) | 0.17-0.20x | 0.30-0.32x | 0.22-0.65x |

- 0.6B 可实时流式
- 1.7B NPU: RTF 稳定 0.30x，长音频比 GPU 更快（KV-cache 优化），但编译时间 ~8 分钟
- 1.7B GPU: 编译快（24s），短音频快，长音频 RTF 退化（decode 随 context 增长变慢）
