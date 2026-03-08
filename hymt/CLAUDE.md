# HY-MT 模块 — HY-MT1.5-1.8B 机器翻译

基于腾讯 HY-MT1.5-1.8B，使用 OpenVINO + openvino_genai 的 `LLMPipeline` 进行推理。

## 快速开始

### 准备模型

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python hymt/scripts/prepare_mt_models.py
```

### 代码调用

```python
from hymt import MTEngine

engine = MTEngine(device="GPU")

# 单句翻译
result = engine.translate("Hello world", target_lang="Chinese")

# 多句 KV cache 会话模式（节省 prefill）
engine.start_session("English")
result1 = engine.translate("你好世界。")
result2 = engine.translate("今天天气好。")  # 复用 KV cache
engine.finish_session()
```

## 模型导出流水线

`hymt/scripts/prepare_mt_models.py` 执行 5 个步骤：

```
Step 1: 下载 tencent/HY-MT1.5-1.8B
Step 2: 加载 HunYuanDenseV1ForCausalLM
Step 3: 重映射权重 (query_layernorm -> q_norm, key_layernorm -> k_norm)
Step 4: 构建独立 Qwen3ForCausalLM checkpoint
Step 5: optimum-intel 导出 + INT4_SYM 量化 + OV tokenizer 转换
```

### 架构映射

HunYuanDenseV1 与 Qwen3 架构相同（都有 QK norm），只需重命名 `query_layernorm → q_norm`, `key_layernorm → k_norm`。

## 模型文件

```
models/
├── hy_mt_int4sym/               # INT4_SYM 模型 ★推理用
│   ├── openvino_model.xml/bin
│   ├── openvino_tokenizer.*
│   └── openvino_detokenizer.*
└── hy_mt_cache_sym/             # NPU 编译缓存
```

## KV Cache 会话模式

使用 `LLMPipeline.start_chat()` / `finish_chat()` 在多句翻译间复用 KV cache，避免每句重新 prefill 整个上下文。

```python
engine.start_session("English")        # start_chat(system_prompt)
engine.translate("你好世界。")          # 只 prefill 新句子
engine.translate("今天天气好。")        # 复用 KV cache
engine.finish_session()                # finish_chat() 释放 cache
```

**自动 token 限制保护**：`needs_reset(max_tokens=400)` 在累积 token 接近 `MAX_PROMPT_LEN`(512) 时触发 `reset_session()`，重建会话。App 层在每句翻译前检查并自动 reset。

会话模式比无状态模式快约 18%（10 句测试），长会话收益更大。

## Prompt 模板

中文目标：
```
将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

其他语言：
```
Translate the following segment into {target_language}, without additional explanation.

{source_text}
```

## 模块说明

| 模块 | 类 | 职责 |
|------|----|------|
| `config.py` | — | 路径常量、NPU 配置 |
| `engine.py` | `MTEngine` | 翻译引擎：封装 LLMPipeline + 会话管理 |

## 支持语言

HY-MT1.5-1.8B 支持 33 种语言互译，包括中、英、日、韩、法、德、西、俄等主要语言，以及粤语、藏语、维吾尔语等方言。
