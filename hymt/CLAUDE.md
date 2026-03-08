# HY-MT 模块 — HY-MT1.5-1.8B NPU 机器翻译

基于腾讯 HY-MT1.5-1.8B，使用 OpenVINO + openvino_genai 在 Intel NPU 上运行机器翻译。

## 快速开始

### 1. 准备模型

```powershell
# 从 HuggingFace 下载并导出（需要下载 ~3.8GB + 导出时间）
$env:PYTHONIOENCODING = "utf-8"
uv run python hymt/scripts/prepare_mt_models.py
```

如果已有本地模型权重：

```powershell
uv run python hymt/scripts/prepare_mt_models.py --model-id C:\path\to\HY-MT1.5-1.8B --skip-download
```

### 2. 运行测试

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python hymt/scripts/test_mt.py
uv run python hymt/scripts/test_mt.py --device CPU  # CPU 回退
```

### 3. 代码调用

```python
from hymt import MTEngine

engine = MTEngine(device="NPU")
result = engine.translate("Hello world", target_lang="Chinese")
print(result)  # "你好，世界。"
```

## 模型导出流水线

`hymt/scripts/prepare_mt_models.py` 执行 5 个步骤：

```
Step 1: 下载 tencent/HY-MT1.5-1.8B (HuggingFace)
Step 2: 加载 HunYuanDenseV1ForCausalLM
Step 3: 重映射权重 (query_layernorm -> q_norm, key_layernorm -> k_norm)
Step 4: 构建独立 Qwen3ForCausalLM checkpoint → hy_mt_qwen3_standalone/
Step 5: optimum-intel 导出 + 权重压缩 + OV tokenizer 转换 → hy_mt_ov/
```

### 架构映射关键说明

HunYuanDenseV1 与 Qwen3 架构几乎完全相同（都有 QK norm），唯一区别是命名：
- `query_layernorm` → `q_norm`
- `key_layernorm` → `k_norm`

通过重命名 state dict key 即可直接加载到 Qwen3ForCausalLM。

## 模型文件

导出后 `models/` 目录结构：

```
models/
├── hy_mt_ov/                    OpenVINO IR + OV tokenizer (★推理用, ~1.7GB)
│   ├── openvino_model.xml/bin   模型 IR
│   ├── openvino_tokenizer.*     OV tokenizer (LLMPipeline 需要)
│   └── openvino_detokenizer.*   OV detokenizer
├── hy_mt_qwen3_standalone/      Qwen3ForCausalLM HF checkpoint (中间产物)
└── HY-MT1.5-1.8B/              原始下载权重 (可删除)
```

推理只需要：`hy_mt_ov/`。
中间产物 `hy_mt_qwen3_standalone/` 和 `HY-MT1.5-1.8B/` 可删除。

## 推理架构

```
Source text → build_prompt() → LLMPipeline.generate() → Translation
```

使用 `openvino_genai.LLMPipeline`，它内部处理 tokenize → prefill → decode → detokenize。

### Prompt 模板

中文目标：
```
将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

其他语言目标：
```
Translate the following segment into {target_language}, without additional explanation.

{source_text}
```

## NPU 配置

```python
NPU_CONFIG = {
    "MAX_PROMPT_LEN": 512,
    "NPUW_LLM_PREFILL_CHUNK_SIZE": 512,
    "GENERATE_HINT": "BEST_PERF",
}
```

## 模块说明

| 模块 | 类 | 职责 |
|------|----|------|
| `config.py` | — | 路径常量、NPU 配置 |
| `engine.py` | `MTEngine` | 翻译引擎：封装 LLMPipeline |

## 依赖

在 `pyproject.toml` 中需要额外添加：
- `openvino-genai >= 2026.0.0.0` — LLMPipeline 推理

运行环境：Windows 11, Python 3.12, `uv` 管理依赖。

## 支持语言

HY-MT1.5-1.8B 支持 33 种语言之间的互译，包括中、英、日、韩、法、德、西、俄等主要语言，
以及粤语、藏语、维吾尔语等 5 种民族语言/方言。
