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

| 设备 | 状态 | Prefill | Decode | 备注 |
|------|------|---------|--------|------|
| CPU | 正常 | ~1.5 tok/s | ~3.6 tok/s | 稳定 |
| **GPU_ONLY** | **正常** | **9.1 tok/s** | **13.6 tok/s** | 全 stateful（GDN+Attn），编译快 7.5s |
| **HYBRID (GPU+NPU)** | **正常** | **10.7 tok/s** | **15.3 tok/s** | GDN→GPU(stateful), Attn→NPU(explicit) |
| NPU only | 不可用 | — | — | GDN 递归 FP16 精度不足 |

### NPU 单独跑 GDN 不行，但 attention 子图可以

Intel NPU 不能单独跑完整模型（两个障碍）：
1. **FP16 精度不足**：GDN delta rule 线性递归每步乘法放大舍入误差，18 层串联后发散
2. **NPU 编译器不支持 Loop 节点**：GDN 递归在 IR 中为 Loop，NPU 报 `to_shape was called on a dynamic shape`

但 attention 子图（标准 SDPA，无 Loop）**可以在 NPU 运行**，FP16 精度完全够用（max diff 0.002）。
因此采用 **Hybrid 方案**：GDN blocks → GPU，Attention blocks → NPU，两者并行利用。

对比：高通 SM8650 NPU 有 FP32 累加器 + TFLite 编译时展开 Loop → GDN 可正常推理。详见 `intel-npu-gdn.md`。

## Hybrid GPU+NPU 推理（13 子图方案）

### 架构

将 24 层拆为 13 个独立子图 IR：

| 子图 | 数量 | 设备 | State 管理 | 内容 |
|------|------|------|-----------|------|
| GDN block | 6 | GPU | **Stateful**（ReadValue/Assign，state 常驻 GPU 显存） | 每块 3 层 GDN（含 Loop 节点，需 FP32） |
| Attention block | 6 | GPU/NPU | **GPU: Stateful**（KV cache 常驻 GPU）/ NPU: Explicit I/O | 每块 1 层 full attention（标准 SDPA，FP16 ok） |
| Head block | 1 | GPU | 无状态 | RMSNorm + lm_head |

执行顺序：gdn_0 → attn_0 → gdn_1 → attn_1 → ... → gdn_5 → attn_5 → head

**Stateful 优化**：IR 导出为显式 I/O，推理加载时用 `apply_make_stateful_transformation` 转换为 ReadValue/Assign，state 常驻 GPU 显存。

- **GDN blocks**：Conv/recurrent state（每块 ~3.4MB × 2）→ stateful。每次 infer() 只传 hidden (4KB)
- **Attention blocks (GPU_ONLY)**：KV cache → stateful。每次只传 hidden + position_ids + mask。KV cache 初始化 past_seq=0（空），内部 Concat 自动增长，无 dummy entry 无 padding
- **Attention blocks (HYBRID/NPU)**：仍用显式 I/O（NPU 需 static shapes + KV padding）

GPU_ONLY 总加速：8.6 → 13.6 tok/s (+58%)，消除每 token 41MB+ host↔GPU 传输。

### 命令

**导出**（qwen35 venv）：
```powershell
uv run --project qwen35 python -m qwen35.export_hybrid
```

**推理**（根 venv）：
```powershell
uv run python -m qwen35.scripts.run_inference --prompt "你好" --device HYBRID
```

也支持 `GPU_ONLY`（全部子图在 GPU）和 `CPU_ONLY` 模式。

### NPU Attention 关键设计

1. **静态 shape**：NPU 要求 static shape，KV cache 固定到 `MAX_CACHE=256`，不足部分零填充
2. **attention_mask 必须有**：零填充的 KV 位置在 softmax 中得到 `exp(0)=1` 的非零权重，导致 attention 稀释。必须传入 4D mask `[B, 1, query_seq, key_seq]`，padded 位置用 `-65504.0`（fp16 最小值）屏蔽
3. **Position 0 始终 masked**：KV cache 初始化有 dummy entry（position 0），attention_mask 对 position 0 也填 `-65504.0`

### 踩坑记录

#### `past_key_value` vs `past_key_values`（复数！）
HF transformers v5 的 `DecoderLayer.forward` 参数名是**复数** `past_key_values`。如果传单数 `past_key_value=cache`，由于函数有 `**kwargs`，错误参数被静默吞掉，导致：
- KV cache `update()` 从未被调用 → 没有 `torch.cat` → IR 中无 Concat
- 输出 KV cache 等于输入（identity）→ I/O tensor 共享同一底层对象
- `set_names()` 互相覆盖 → 命名错乱
- hidden_states 输出从 3D 变 2D（batch+seq 维度合并）

**一个参数名拼写错误导致了三个看似无关的 bug。**

#### `set_partial_shape()` + `validate_nodes_and_infer_types()` 不会改变图结构
只更新 PartialShape 声明并重新推导形状/类型，**不会删除 Concat、不会添加/移除节点**。如果导出后 IR 缺少 Concat，原因是 trace 阶段就没有捕获到，不是后处理破坏。

#### NPU attention_mask 填充值
`-1e4` 不够（Q 值 mean-centered 时 `Q @ K^T ≈ 0`，softmax 仍给 masked 位置分配权重）。必须用 `-65504.0`（fp16 最小值 `-HALF_MAX`）才能保证 near-zero。

#### `layer_idx` 必须设为 0
子图导出时，`self_attn.layer_idx` 必须临时改为 0。否则 `past_key_values.update(..., self.layer_idx)` 访问 `self.key_cache[全局layer_idx]` 会越界，导致 trace 失败或 cat 被跳过。

#### HETERO 方案不可行
测试了 8 种 HETERO 配置（OpenVINO 自动子图分割），全部失败（undeclared parameters / Cannot sort subgraphs）。HETERO 在 Loop + Stateful + dynamic shapes 模型上有 bug，只能手动拆子图。

#### `apply_make_stateful_transformation` 用法
将显式 state I/O 转为 OpenVINO 内部 ReadValue/Assign 节点。用法：
```python
from openvino._offline_transformations import apply_make_stateful_transformation
state_map = {"in_conv0": "out_conv0", "in_rec0": "out_rec0", ...}  # input_name -> output_name
apply_make_stateful_transformation(ir, state_map)  # in-place 修改
```
转换后 IR 只剩非 state 的 input/output（如 hidden + mask），state 变为内部变量。
注意：动态 dim 的 state 初始 shape 为 0，必须手动 `s.state = ov.Tensor(np.zeros(shape))` 初始化。`s.reset()` 会恢复 shape=0，不能用。

#### Stateful 优化性能对比
| 优化阶段 | GPU_ONLY | HYBRID |
|---------|----------|--------|
| 原始 (全 explicit I/O) | 8.6 tok/s | 9.7 tok/s |
| + GDN stateful | 10.0 tok/s | 13.6 tok/s |
| + Attn stateful (GPU) | **13.6 tok/s** | — (NPU 仍 explicit) |
| HYBRID 最新 | — | **15.3 tok/s** |

#### KV cache 初始化：past_seq=0 vs past_seq=1
GPU stateful attention 用 `past_seq=0`（空 KV cache），IR 内 `Concat([B,H,0,D], [B,H,1,D])→[B,H,1,D]` 能正确处理 0-length。无 dummy entry，attention_mask 全零（attend everything）。
NPU explicit I/O 仍用 `past_seq=1`（dummy entry），需 mask position 0。

### 模块说明

| 模块 | 职责 |
|------|------|
| `export_hybrid.py` | 导出 13 子图 IR（GDN/Attention/Head blocks），显式 I/O |
| `inference_hybrid.py` | `Qwen35HybridModel` — 编排 13 子图推理，GDN stateful + Attn explicit I/O |
| `scripts/run_inference.py` | 统一入口，`--device HYBRID/GPU_ONLY/CPU_ONLY` |

## 标准导出流水线（单 IR）

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
| `inference.py` | `Qwen35OVModel` — 继承 `GenerationMixin`，管理 stateful 推理（单 IR） |
| `export.py` | PyTorch → OpenVINO 单 IR 导出（model patching + stateful 转换） |
| `stateful.py` | IR 后处理：cache 输入/输出 → ReadValue/Assign 变量 |
| `export_hybrid.py` | 导出 13 子图 IR（GDN/Attention/Head blocks），显式 I/O |
| `inference_hybrid.py` | `Qwen35HybridModel` — 编排 13 子图混合推理 |

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
├── Qwen3.5-0.8B-ov/              # 标准单 IR（CPU/GPU stateful 推理）
│   ├── openvino_model.xml/.bin   # stateful IR (FP16, inputs_embeds 入口)
│   ├── embed_tokens.npy          # 248320×1024 float16 (485 MB)
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── Qwen3.5-0.8B-hybrid/          # 13 子图（Hybrid GPU+NPU 推理）
│   ├── gdn_block_{0-5}.xml/.bin  # 6 个 GDN 子图 (各 3 层, GPU)
│   ├── attn_block_{0-5}.xml/.bin # 6 个 Attention 子图 (各 1 层, NPU)
│   ├── head_block.xml/.bin       # Head 子图 (norm+lm_head, GPU)
│   ├── embed_tokens.npy
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
