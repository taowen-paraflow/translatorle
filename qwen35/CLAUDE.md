# Qwen3.5 模块 — 通用文本生成（Gated Delta Networks 混合架构）

Qwen3.5-0.8B OpenVINO 推理，自定义导出 + 推理管线，不依赖 optimum-intel。

## 独立 venv

qwen35 模块有自己的 `pyproject.toml`（`transformers>=5.0`），与主项目（`transformers>=4.49,<5`）隔离。

- **导出**需要 `transformers>=5.0`（`export.py` 导入 `transformers.models.qwen3_5.modeling_qwen3_5`）
- **推理**只使用 `AutoConfig`、`AutoTokenizer`、`GenerationMixin` 等通用接口，兼容主项目 venv

## 快速开始

### 准备模型（使用 qwen35 独立 venv）

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_models --hf-model Qwen/Qwen3.5-0.8B
```

模型导出到 `models/qwen35/Qwen3.5-0.8B-ov/`。

### 运行推理（使用主项目 venv）

```powershell
uv run python -m qwen35.scripts.run_inference --prompt "The capital of France is" --device CPU
```

### 代码调用

```python
from qwen35.inference import Qwen35OVModel

model = Qwen35OVModel.from_pretrained("models/qwen35/Qwen3.5-0.8B-ov", device="CPU")
inputs = model.tokenizer("你好", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(model.tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 架构特点

Qwen3.5 使用 **Gated Delta Networks (GDN) 混合架构**，不是标准 Transformer：

- 24 层：18 层 linear attention (GDN) + 6 层 full attention（每 4 层一个 full）
- **48 个 stateful 变量**：18 conv + 18 recurrent（线性层）+ 6 key + 6 value（注意力层）
- 推理时 token-by-token prefill（模型按 seq_len=1 trace），decode 单步推进

## 设备兼容性

| 设备 | 状态 | FP16 速度 | INT4 速度 | 备注 |
|------|------|-----------|-----------|------|
| CPU | 正常 | 5.0 tok/s | — | 稳定 |
| GPU | **正常** | 8.2 tok/s | **12-13 tok/s** | ScatterUpdate bug 已修复 |
| NPU | **不可用** | — | — | GDN 递归 FP16 精度不足 |

**NPU 结论**：Intel NPU 硬件精度固定 FP16，GDN delta rule 线性递归 `S_t = S_{t-1} * exp(g_t) + outer(k_t, delta_t)` 每步乘法放大舍入误差，18 层串联后 logit 快速发散。已验证多种方案（NPUW_LLM、混合 Shadow FP32、多子图 4 层分组）均无法解决。CPU/GPU 是推荐设备。详见 `intel-npu-gdn.md`。

## 导出流水线

`export.py` 执行自定义导出（不依赖 optimum-intel）：

1. 加载 PyTorch 模型 (`AutoModelForCausalLM`)
2. Patch forward：展开 cache_params 为 conv/recurrent/key/value 张量列表，接受 `inputs_embeds`（不是 `input_ids`）
3. Patch GDN 层：替换 CUDA-only 算子为 `RecurrentAttentionCell`（通过 ModuleExtension 转为 OV Loop 节点）
4. `ov.convert_model` 转 IR（支持动态 seq_len，batch prefill）
5. `stateful.py` 将 cache 输入/输出转为 OpenVINO stateful 变量 + beam_idx reorder
6. 保存 FP16 压缩 IR + 提取 `embed_tokens.npy`（float16）+ 拷贝 tokenizer 文件

IR 输入签名：`inputs_embeds [B, seq_len, 1024]`, `attention_mask [B, seq_len]`, `position_ids [3, B, seq_len]`, `beam_idx [B]`。
Embedding 查表在 Python 侧完成（从 `embed_tokens.npy` 查），text-only 和 VL 共用同一份 decoder IR。

## 模块说明

| 模块 | 职责 |
|------|------|
| `config.py` | 架构常量（层数、head 维度、层类型 pattern）+ 设备配置 |
| `inference.py` | `Qwen35OVModel` — 继承 `GenerationMixin`，管理 stateful 推理 |
| `export.py` | PyTorch → OpenVINO IR 导出（model patching + stateful 转换） |
| `stateful.py` | IR 后处理：cache 输入/输出 → ReadValue/Assign 变量 |

## VL（视觉语言）模块

### 快速开始

**导出 VL 模型**（qwen35 独立 venv，transformers 5.x）：
```powershell
uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_vl_models
```

**运行 VL 推理**（主项目 venv）：
```powershell
uv run python -m qwen35.scripts.run_vl_inference --image photo.jpg --prompt "描述这张图片"
```

### VL 模块说明

| 模块 | 职责 |
|------|------|
| `ov_vision_encoder.py` | Vision encoder OV 推理封装 |
| `inference.py` (`Qwen35VLModel`) | VL 推理引擎（vision encoder + decoder 拼接） |
| `export_vl.py` | vision encoder 导出 + embed_tokens 提取 |
| `config.py` (VL 部分) | VL 模型路径、special token IDs |
| `scripts/prepare_qwen35_vl_models.py` | VL 导出编排脚本 |
| `scripts/run_vl_inference.py` | VL 推理测试脚本 |

### VL 已知问题

#### 已修复
- **IR surgery 消除**（2026-03-10）：`export.py` 直接 trace `inputs_embeds`，text-only 和 VL 共用同一份 decoder IR，无需图操作
- **Prefill 策略**：`Qwen35VLModel.prefill()` 支持 batch prefill（单次 infer 调用处理整个 prompt）
- **Vision encoder 输出**：`OVVisionEncoder` 选择 `pooler_output`（144 tokens, 1024-dim）
- **预处理分辨率**：固定 384×384（`VL_IMAGE_SIZE` 常量），patch 数与导出时位置编码一致
- **embed_tokens.npy**：float16 存储（~485MB），加载时 cast float32

#### 待改进
- **固定分辨率限制**：当前只支持 384×384，不支持动态分辨率（ViT 位置编码在导出时固化）
- **NPU 不可用**：同 text-only，GDN 递归 state 在 NPU FP16 下累积误差发散
- **仅 Greedy 解码**：不支持 temperature/top-p/beam search
- **停止条件不完善**：生成末尾偶现空代码块等无意义 token

### VL 推理流程

1. **图像预处理**：固定 resize 到 384×384，temporal 帧复制 → pixel_values `[576, 1536]` + grid_thw `[1, 24, 24]`
2. **图像编码**：`OVVisionEncoder` → `pooler_output` `[1, 144, 1024]`（2×2 merge: 576→144 tokens, 768→1024 投影）
3. **Prompt 构建**：ChatML 格式，`<|vision_start|>` + 144 个 `<|image_pad|>` + `<|vision_end|>`
4. **Embedding 拼接**：用 embed_tokens.npy 查表获得文本 embedding，image_pad 位置替换为 visual features
5. **mRoPE 位置**：构建 `[3, 1, seq_len]` 位置 IDs，图像 token 用空间坐标（temporal/height/width）
6. **Prefill**：batch prefill — 单次 infer() 送入整个 prompt，GDN Loop 节点处理序列维度
7. **Decode**：自回归单步生成，greedy argmax

### VL 性能参考（CPU）

- Prefill 169 tokens + Decode 99 tokens = **7.6 tok/s**

## 模型文件

```
models/qwen35/
├── Qwen3.5-0.8B-ov/
│   ├── openvino_model.xml/.bin   # stateful IR (FP16, inputs_embeds 入口)
│   ├── embed_tokens.npy          # 248320×1024 float16 (485 MB)
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── Qwen3.5-0.8B-vl/
    ├── vision_encoder.xml/.bin   # ViT (189 MB)
    ├── openvino_model.xml/.bin   # 同 text-only decoder（复制）
    ├── embed_tokens.npy          # 同 text-only（复制）
    ├── config.json
    ├── preprocessor_config.json
    ├── tokenizer.json
    └── tokenizer_config.json
```
