# ASR 模块 — Qwen3-ASR NPU 流式语音识别

基于 Qwen3-ASR-0.6B，使用 OpenVINO 在 Intel NPU 上运行流式语音识别。

## 快速开始

### 1. 准备模型

```powershell
# 从 HuggingFace 下载并导出（需要 ~44s，产出 ~6.6GB）
$env:PYTHONIOENCODING = "utf-8"
uv run python asr/scripts/prepare_asr_models.py --model-id Qwen/Qwen3-ASR-0.6B
```

如果已有本地模型权重：

```powershell
uv run python asr/scripts/prepare_asr_models.py --model-id C:\path\to\Qwen3-ASR-0.6B
```

脚本会在 `models/` 下生成所有推理所需文件（见下方"模型文件"）。

### 2. 运行测试

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python asr/scripts/test_asr_wav.py test_zh.wav
```

### 3. 代码调用

```python
from asr import ASREngine

engine = ASREngine(encoder_device="NPU", decoder_device="NPU")
state = engine.new_session(language="Chinese")

# 流式喂入 PCM 音频（16kHz float32）
text = engine.feed(pcm_chunk, state)   # 每次返回当前累积转写
text = engine.feed(pcm_chunk2, state)
final = engine.finish(state)           # 结束并返回最终文本
```

## 模型导出流水线

`asr/scripts/prepare_asr_models.py` 执行 8 个步骤：

```
Step 1: 加载 Qwen3ASRForConditionalGeneration (HuggingFace transformers)
Step 2: 导出 Audio Encoder → encoder_fp16.xml (ov.convert_model, FP16)
Step 3: 提取 Text Decoder 权重 (排除 audio_tower 参数)
Step 4: 构建独立 Qwen3ForCausalLM checkpoint → qwen3_decoder_standalone/
Step 5: optimum-intel 导出 stateful decoder → decoder_stateful_ov/ (带 KV-cache)
Step 6: IR 图手术 → decoder_stateful_embeds/ (移除 input_ids, 添加 inputs_embeds)
Step 7: 提取 embed_tokens.npy (embedding 查找表)
Step 8: 拷贝 tokenizer + preprocessor_config.json → Qwen3-ASR-0.6B/
```

### IR 图手术（Step 6）关键说明

NPUW_LLM 插件按名称优先选择输入：如果模型同时有 `input_ids` 和 `inputs_embeds`，
会选 `input_ids`（int64 零值），导致输出乱码。必须从 IR 中**移除 `input_ids` 参数**，
让 NPUW 回退到 `inputs_embeds` 路径。

手术操作：
1. 将 Gather（embedding lookup）的消费者重定向到新的 `inputs_embeds` Parameter
2. 断开 `input_ids` 的所有消费者（ShapeOf、Convert→Gather）
3. 创建新 Model 时排除 `input_ids`

## 模型文件

导出后 `models/` 目录结构：

```
models/
├── encoder_fp16.xml/.bin           # Audio encoder, 固定输入 [1, 128, 800], ~356MB
├── decoder_stateful_embeds/        # Text decoder (IR手术后), inputs_embeds + KV-cache, ~1.1GB ★推理用
├── decoder_stateful_ov/            # Text decoder (手术前), optimum-intel 原始导出, ~2.3GB (中间产物)
├── qwen3_decoder_standalone/       # Qwen3ForCausalLM HF checkpoint (中间产物)
├── embed_tokens.npy                # Embedding 表 [151936, 1024], ~594MB
└── Qwen3-ASR-0.6B/                # tokenizer + preprocessor_config.json
```

推理只需要：`encoder_fp16.*`、`decoder_stateful_embeds/`、`embed_tokens.npy`、`Qwen3-ASR-0.6B/`。
中间产物 `decoder_stateful_ov/` 和 `qwen3_decoder_standalone/` 可删除。

## 推理架构

### 流程

```
PCM 16kHz → MelProcessor → [1,128,T] mel
                              ↓
                     OVEncoder (NPU) → [1, N, 1024] audio features
                              ↓
              构建 inputs_embeds (prompt tokens + audio features)
                              ↓
                     OVDecoder (NPU, NPUW_LLM) → greedy decode → text
```

### 流式策略：累积重编码 + 前缀回退

不是独立处理每个音频块，而是：
1. **累积音频**：新 chunk 追加到 buffer，每次重新编码全部累积音频
2. **前缀回退**：上一轮转写结果去掉末尾 5 个 token 作为 decoder 前缀，引导续写
3. **KV-cache 重置**：每个 chunk 因为 encoder 输出变化，需要重置 KV-cache 重新 prefill

参数默认值：
- `chunk_size_sec=2.0` — 每 2 秒触发一次推理
- `unfixed_token_num=5` — 回退 5 个 token 允许修正
- `max_new_tokens=32` — 每次最多生成 32 个新 token

### 模块说明

| 模块 | 类 | 职责 |
|------|----|------|
| `config.py` | — | 路径常量、NPU 配置、特殊 token ID |
| `processor.py` | `MelProcessor` | PCM → 128-bin log-mel 频谱 (WhisperFeatureExtractor) |
| `ov_encoder.py` | `OVEncoder` | OpenVINO audio encoder，固定 800 帧输入 |
| `ov_decoder.py` | `OVDecoder` | OpenVINO stateful decoder (NPUW_LLM)，支持 prefill + decode_step |
| `engine.py` | `ASREngine` | 流式引擎：累积重编码 + 前缀回退 + prompt 构建 + greedy decode |

### Prompt 格式 (ChatML)

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n
<|im_start|>assistant\n[language Chinese<asr_text>已转写文本前缀...]
```

`<|audio_pad|>` 的数量等于 encoder 输出的时间步数（800帧输入 → 104 个 pad token）。
构建 `inputs_embeds` 时，pad 位置替换为 encoder 输出的音频特征向量。

### 特殊 Token ID

```python
IM_START   = 151644   # <|im_start|>
IM_END     = 151645   # <|im_end|>
AUDIO_START = 151669  # <|audio_start|>
AUDIO_END  = 151670   # <|audio_end|>
AUDIO_PAD  = 151676   # <|audio_pad|>
ASR_TEXT   = 151704   # <asr_text>
```

## NPU 编译配置

```python
NPU_DECODER_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 256,
    "NPUW_LLM_MIN_RESPONSE_LEN": 64,
}
```

NPU 编译首次约 8s（后续有缓存），推理 RTF ~0.17x（4.2s 中文音频仅需 ~820ms）。

## 性能参考 (Intel Core Ultra 7 258V, Lunar Lake)

| 测试 | 音频时长 | 推理耗时 | RTF |
|------|---------|---------|-----|
| 中文 test_zh.wav | 4.2s | 822ms | 0.199x |
| 英文 test_en.wav | 15.1s | 2591ms | 0.172x |

### Decoder 量化对比

| 精度 | 模型大小 | Prefill | Decode | 吞吐量 |
|------|---------|---------|--------|--------|
| **FP16** | 1137 MB | 64 ms | 22.6 ms/tok | **40.7 tok/s** |
| INT4_SYM | 366 MB | 113 ms | 37.0 ms/tok | 24.6 tok/s |

**结论：0.6B 小模型在 NPU 上 FP16 已是最优精度，INT4 反而更慢。**
量化脚本：`asr/scripts/quantize_decoder.py`，基准测试：`asr/scripts/bench_decoder.py`。

## 依赖

核心依赖（见 `pyproject.toml`）：
- `openvino >= 2026.0.0`
- `optimum[openvino]` — 模型导出用
- `transformers` — tokenizer + 模型加载
- `torch`, `torchaudio` — 模型导出用（Intel XPU 源）
- `librosa` — 音频加载
- `nncf` — 量化支持

运行环境：Windows 11, Python 3.12, `uv` 管理依赖。
