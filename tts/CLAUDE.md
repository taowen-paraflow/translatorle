# TTS Module — Qwen3-TTS + OpenVINO

Standalone streaming TTS module ported from `vllm-omni/qwen3_tts_app`.

## Architecture

3-stage pipeline:
1. **Talker** (28-layer Qwen3 LLM, NPU) — autoregressive, predicts layer-0 codec token per step
2. **Code Predictor** (5-layer transformer, CPU) — stateful KV cache, predicts residual codes 1-15
3. **Speech Decoder** (CNN, NPU) — converts 16-layer codes to PCM waveform (1920x upsample)

Streaming: frames buffer → every 50 frames, async decode in background thread with 25-frame overlap.

## Files

| File | Role |
|------|------|
| `config.py` | `TTSModelConfig` dataclass, constants, NPU configs |
| `engine.py` | `TTSEngine` — orchestrates 3 stages with streaming |
| `ov_talker.py` | `OVTalker` — NPU/CPU, FP16/INT4, pseudo-inverse for hidden recovery |
| `ov_code_predictor.py` | `OVCodePredictor` — CPU, stateful, identity lm_head trick |
| `ov_decoder.py` | `OVSpeechDecoder` — static shape [1,16,75] |
| `text_conditioner.py` | `TextConditioner` — numpy-only text→embedding (no PyTorch) |
| `scripts/prepare_tts_models.py` | All-in-one: export + surgery + quantize |
| `scripts/qwen_tts/` | Vendored HF model code (export-time only) |

## Model Preparation

```bash
cd C:\Apps\translatorle
uv run python -m tts.scripts.prepare_tts_models --hf-model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

Output goes to `models/tts/`. Steps:
1. Load safetensors, extract Talker weights → Qwen3ForCausalLM → export stateful
2. IR surgery: remove `input_ids`, add `inputs_embeds` (NPUW_LLM requirement)
3. Create NPU variant (single-output) + pseudo-inverse for hidden state recovery
4. INT4 quantization (nncf INT4_SYM) + NPU INT4 variant
5. Extract Code Predictor weights (identity lm_head trick) → export stateful + surgery
6. Export Speech Decoder (static [1,16,75]) using vendored HF model code
7. Extract numpy artifacts (embeddings, projections, lm_heads)

## NPU Configuration

| Config | Key Settings |
|--------|-------------|
| FP16 | `NPUW_LLM=YES`, `MAX_PROMPT_LEN=512` |
| INT4 | + `NPUW_FOLD=NO`, `NPUW_LLM_PREFILL_HINT=STATIC` (bypasses folding crash) |

## Benchmark Results

### translatorle (2026-03-09)

Test: "今天天气真不错，我们一起出去走走吧。" (speaker=serena)

| Config | RTF | Talker avg | CodePred avg | Load time | Steps |
|--------|-----|-----------|-------------|-----------|-------|
| CPU (all) | 3.50-3.62 | 43.2 ms | 144-148 ms | 5.5s | 39-42 |
| NPU FP16 | **2.13-2.17** | 30.3-30.6 ms | 126-130 ms | 124s | 45-47 |
| NPU INT4 | 2.32-2.39 | 39.7-47.6 ms | 125-138 ms | 397s | 48-51 |

Bottleneck: CodePredictor ~77-81% of total time.

### qwen3_tts_app reference (2026-03-09)

| Config | RTF | Talker avg | CodePred avg |
|--------|-----|-----------|-------------|
| NPU FP16 | 2.31 | 31.2 ms | 134.5 ms |
| NPU INT4 | 3.20 | 53.0 ms | 182.3 ms |

translatorle matches or slightly improves on the reference (same models, pure-numpy text conditioner).

## Usage

```python
from tts import TTSEngine, TTS_MODELS

engine = TTSEngine(TTS_MODELS["FP16"])
for chunk_idx, wav_samples in engine.generate("Hello world", speaker_id=3066):
    # wav_samples is float32 1-D array, 24kHz
    pass
```

## Key NPU Tricks

1. **`input_ids` removal**: NPUW_LLM prefers `input_ids` over `inputs_embeds`. Surgery removes `input_ids` Parameter and redirects the Gather consumers to a new `inputs_embeds` Parameter.
2. **Pseudo-inverse**: NPU single-output model emits only logits. Hidden states recovered via `hidden = logits @ lm_head_pinv`.
3. **Identity lm_head**: Code Predictor uses `vocab_size = hidden_size` and `lm_head = Identity`, so "logits" output = hidden states.
4. **`NPUW_FOLD=NO`**: Required for INT4 models to prevent NPU compiler crash during graph folding.
