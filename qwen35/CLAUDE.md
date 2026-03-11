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

| 设备 | 状态 | 速度 | 备注 |
|------|------|------|------|
| **CPU** | **正常** | 10-14 tok/s | Loop IR, batch prefill, **推荐** |
| GPU | 正常 | ~11 tok/s | Loop IR, batch prefill |
| NPU (NPUW_LLM) | **输出乱码** | 18.9 tok/s | Loop-free IR, FP16 精度不足 |
| NPU (旧静态) | **输出乱码** | 13.4 tok/s | Loop-free IR, FP16 精度不足 |
| NPU+CPU 混合 | **输出乱码** | ~10.7 tok/s | Shadow FP32 state 防止跨步累积，但逐层 FP16 误差仍太大 |
| NPU 多子图 | **实验中** | 待测 | 6 子图×4 层，子图间 FP32 hidden_states，FP16 误差限于 4 层 |

**结论：Qwen3.5 GDN 架构不适合 NPU。CPU 是最佳选择。** 多子图方案正在验证。
- NPU FP16 硬件精度对 GDN 递归 state 不够用（根因已确认）
- 混合方案虽然解决了跨步 state 累积问题，但每步内 24 层 FP16 hidden_state 传播仍然发散
- **多子图方案**：将 24 层拆为 6 个子图（每组 4 层），子图间 hidden_states 回 CPU 转 FP32，FP16 误差从 24 层降至 4 层
- 关键：NPU 速度 10.7 tok/s vs CPU 10.5 tok/s — 0.8B 模型太小，host↔device 拷贝抵消了 NPU 加速
- 传统 Transformer（如 Qwen3-ASR、HY-MT）无 GDN 递归，FP16 够用，NPU 有明显加速

### NPU 支持

**目标**：在 NPU 上运行 Qwen3.5，KV cache 留在设备端（不每步 host↔device 全量拷贝）。

**唯一本质断点**：NPU 编译器不支持 `Loop`/`TensorIterator` 节点。
- 传统 Transformer 天然无 Loop（Attention = matmul + softmax），NPU 直接编译
- GDN 的 delta rule recurrence 必须逐 token 串行更新 state → `ov.convert_model` 生成 Loop 节点 → NPU 卡死
- KV cache vs GDN state 的**全部本质区别**就是计算图里有没有 Loop 节点，其他都是表象

### NPU 方案对比：旧方案（Fork OpenVINO） vs LL2 方案（推荐）

**核心区别：GDN state 对 NPUW_LLM 是否可见**

| | 旧方案（Fork） | LL2 方案（推荐） |
|--|---------------|----------------|
| Loop 消除方式 | 手写 `patched_recurrent_gated_delta_rule_single_step` 在 Python 层内联 | LowLatency2 pass 自动 unroll |
| GDN state 位置 | **显式 I/O**（模型的 Parameter/Result） | **内部 ReadValue/Assign**（模型内部有状态节点） |
| NPUW_LLM 看到什么 | 48 个 GDN state + 12 个 KV cache → 把 GDN state 误当 KV cache | **只有 12 个标准 KV cache** → 原生支持 |
| 需要 fork OpenVINO | **是**（6 处 C++ 修改，教 NPUW_LLM 跳过 GDN state） | **否**（GDN state 对外不可见，NPUW_LLM 无需修改） |
| 导出目录 | `Qwen3.5-0.8B-npuw/` | `Qwen3.5-0.8B-ll2/` |
| FP16 精度 | 同样受 GDN state FP16 累积误差影响 | 同样受影响（NPU 硬件限制） |
| 当前状态 | 已废弃 | 架构正确，但 NPU FP16 精度不足且速度无优势，放弃 NPU |

**为什么 LL2 方案不需要 fork**：
- LL2 把 GDN 的 recurrent state 变成 ReadValue/Assign，MakeStateful 把 conv state 也变成 ReadValue/Assign
- ReadValue/Assign 是**模型内部的有状态节点**，由 OpenVINO runtime 自动在 `infer()` 调用之间保持值
- NPUW_LLM 只管理模型的显式 I/O（Parameter 和 Result），**根本看不到内部的 ReadValue/Assign**
- 所以 NPUW_LLM 只看到标准的 12 个 KV cache I/O → 原生支持，无需任何 C++ 修改

**类比**：
- 旧方案 = GDN state 是函数参数（调用者 NPUW_LLM 必须管）→ NPUW_LLM 不认识 → 必须改 NPUW_LLM
- LL2 方案 = GDN state 是局部静态变量（函数内部自己管）→ NPUW_LLM 看不到 → 不用改

### LL2 方案详细流水线

从非 stateful 的 Loop IR 出发（`export_model_loop_ir`）：

```
convert_model(RecurrentAttentionCell) → 18 Loop + 显式 I/O (非 stateful)
    ↓ reshape(batch=1, seq=1, kv_cache=2048) — 全静态
    ↓ ConstantFolding (独立 Manager) — trip_count → Constant(1)
    ↓ fix timestep Parameters → Constant(0)
    ↓ LowLatency2 (独立 Manager) — 18 Loop → 0 Loop + 36 ReadValue/Assign (全静态!)
    ↓ MakeStateful (GDN conv only) — conv state 也转 ReadValue/Assign
    ↓ Post-fix: 添加 beam_idx + 重命名 KV 为 HF 标准命名
    ↓ ConstantFolding + DeadCodeElimination
    → Qwen3.5-0.8B-ll2/ (0 Loop, 全静态, NPUW_LLM 原生兼容)
```

**关键技术细节**：
- 每个 pass 用独立的 `ov_passes.Manager()`（ConstantFolding 和 LowLatency2 在同一 Manager 冲突）
- LL2 的 "undeclared parameters" 错误：Loop body 的 timestep Parameter 未处理 → 手动替换为 Constant(0)
- **必须从非 stateful IR 开始**：从已 stateful 模型做 LL2 → ReadValue 的 variable_shape 在创建时固化为动态 → Python API 无法修改 → NPU 编译失败

**命令**：
```bash
# 导出非 stateful Loop IR（transformers 5.x venv）
uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_models --hf-model Qwen/Qwen3.5-0.8B --ll2
# 运行 LL2 pipeline（主项目 venv）
uv run python -m qwen35.scripts._test_ll2_pipeline
```

### 旧方案 Fork 详情（存档，不再推荐）

**OpenVINO Fork**（`C:\Apps\openvino\`）改动 3 个文件，解决 NPUW_LLM 把 GDN state 误当 KV cache 的问题：

| 问题 | 位置 | 根因 | Fork 修复 |
|------|------|------|-----------|
| reshape dim2 | `reshape_to_static` else 分支 | static dim2 被当 seq_len 覆盖 | `is_dynamic()` 检查跳过 static dim |
| name mapping | `copy_kvcache` / `update_kvcache_for` | `"present"→"past_key_values"` 对 GDN 名字映射失败 | 加 `"present"→"past"` fallback |
| KV slice copy | `copy_kvcache` / `update_kvcache_for` | GDN state 被当 KV 做 slice copy | 检测 src==dst shape → full copy |
| zero-init | constructor CPU workaround | GDN state 不被初始化为零 | 同上 fallback 扩展到 zero-init |
| bind_past_kv | `bind_past_kv` | 检查 "past_key_values" 字符串 | GDN 不匹配→不 bind，可接受 |
| variant sharing | `create_generate_request_variants` | 同上 | GDN 不匹配→不 share，可接受 |

Fork 构建命令（仅在需要旧方案时使用）：
```powershell
$env:Path = "C:\Apps\translatorle\.venv\Lib\site-packages\cmake\data\bin;C:\Apps\translatorle\.venv\Scripts;" + $env:Path
cd C:\Apps\openvino\build
cmake --build . --config Release
Remove-Item C:\Apps\openvino\build\wheels\*.whl -Force
cmake --build . --config Release --target ie_wheel
cd C:\Apps\translatorle
uv pip install --reinstall --no-deps (Get-ChildItem C:\Apps\openvino\build\wheels\*.whl).FullName
```

### 准备 NPU 模型（旧版全静态，Python 管 KV）

```powershell
uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_models --hf-model Qwen/Qwen3.5-0.8B --npu
```

### 准备 NPUW_LLM 模型（推荐：设备端 KV cache）

```powershell
uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_models --hf-model Qwen/Qwen3.5-0.8B --npuw
```

模型导出到 `models/qwen35/Qwen3.5-0.8B-npuw/`。特点：
- 动态 KV shapes + 标准 HF 命名 → NPUW_LLM 自动管理 KV cache
- Loop-free GDN（单步内联） → NPU 兼容
- 2D attention mask → 标准 LLM 接口

### 运行 NPU 推理

```powershell
# NPUW_LLM 模式（推荐，设备端 KV cache）
uv run python -m qwen35.scripts.run_inference --device NPUW --prompt "Hello"

# 旧版 NPU 模式（Python 管 state）
uv run python -m qwen35.scripts.run_inference --device NPU --prompt "Hello"
```

### NPU 测试结果（2026-03-10）

| 模式 | 模型 | 速度 | 输出质量 |
|------|------|------|---------|
| NPUW_LLM (`--device NPUW`) | Qwen3.5-0.8B-npuw | 18.9 tok/s | **乱码**（重复 "I am a name."） |
| 旧静态 (`--device NPU`) | Qwen3.5-0.8B-npu | 13.4 tok/s | **乱码**（同上） |
| **NPUW IR on CPU** | Qwen3.5-0.8B-npuw | ~10 tok/s | **正确**（"Hello! My name is Qwen3.5"） |
| CPU (`--device CPU`) | Qwen3.5-0.8B-ov | 10-14 tok/s | 正常 |

**关键发现**：NPUW 模型在 CPU 上**输出完全正确** — 证明 IR 无误，问题在 NPU FP16 精度。

### 乱码根因（已确认：NPU FP16 精度不足）

**核心问题**：同一个 IR，CPU 正确、NPU 乱码 — 与 state 管理方案无关。

**已排除的假设**：
- ~~Loop-free 单步内联数值有 bug~~：CPU 上验证输出正确 "Hello! My name is Qwen3.5"
- ~~IR 本身有 NaN~~：CPU 上无 NaN，logits 范围正常 `[-15, 25]`
- ~~FP16 压缩损失~~：CPU 也用 FP16 压缩 IR，输出正确
- ~~NPUW_LLM GDN state 不拷贝~~：旧 NPU 模式（Python 管 state）也乱码
- ~~state 管理方案差异~~：Fork 方案和 LL2 方案都会有同样的精度问题

**确认的根因：GDN recurrent state 在 FP16 下累积误差发散**

GDN 的 delta rule 递推公式：`S_t = S_{t-1} × exp(g_t) + outer(k_t, delta_t)`
- 这是**线性递归**，每步乘法把上一帧的 FP16 舍入误差继续放大
- 18 层串联 + 多步推理，误差迅速到 3-5% 以上，导致 logit 分叉
- 对比：Transformer 的 KV cache 是 append-only（写一次读多次），不会累积误差

**Intel NPU 硬件限制**：
- NPU 计算精度固定 FP16（官方文档："Computation precision for the HW is FP16."）
- 即使 IR 声明 FP32，NPU 插件自动全部转 FP16 执行和存储
- ReadValue/Assign 在 NPU 上存储和更新也走 FP16，无法通过"把 state 切 FP32"解决
- CPU 虽然 IR 也是 FP16 压缩，但运行时用 FP32 做实际计算 → 精度足够

```python
# 验证 NPU 精度限制
import openvino as ov
core = ov.Core()
print(core.get_property("NPU", "OPTIMIZATION_CAPABILITIES"))  # FP16, 无原生 FP32
```

### 混合 NPU+CPU 方案（Shadow FP32 State）— 已验证，精度仍不足

**状态**: 已实现并测试。Shadow FP32 state 成功防止了跨步 state 累积误差，但 NPU 逐层 FP16 hidden_state 传播仍导致输出发散。且 0.8B 模型 NPU 速度（10.7 tok/s）与 CPU（10.5 tok/s）几乎无差异，不值得继续优化。

**核心思路**：NPU 跑完整前向（包括 FP16 state update + readout），同时输出 GDN 中间量。
CPU 用中间量做 FP32 state 更新。每步开始时，FP32 state → FP16 喂入 NPU。

**为什么"拆分模型"不可行**：
- GDN 的 readout `out = q @ S_updated` 依赖更新后的 state
- 如果 NPU 不做 state 更新，就无法计算 readout
- readout 喂入后续层，每层输入都依赖前一层的 GDN 输出
- 所以不能"NPU 只跑投影，CPU 算 readout 拼回去"——除非逐层调 NPU（太慢）

**正确方案：Shadow FP32 State + NPU 输出中间量**

```
每步推理循环:
  1. S_fp32 → 喂入 NPU（NPU 自动截断为 FP16）
  2. NPU 跑完整前向：
     - FP16 state update + readout → 喂入后续层（readout 有 1 步 FP16 截断误差，可接受）
     - 输出: logits + state_out (不用) + intermediates (g_t, k_t, v_t, beta_t per GDN layer)
  3. CPU FP32 更新 shadow state:
     S_fp32 = S_fp32 * g_t.fp32 + outer(k_t.fp32, delta_t.fp32)
  4. 下一步: goto 1（用 FP32 state，不用 NPU 的 FP16 state_out）
```

**为什么误差不累积**：
- NPU 的 FP16 state 只用一步就丢弃（不作为下一步输入）
- 下一步的 state 输入来自 CPU FP32（通过 FP16 截断喂入，截断误差 ~0.1%）
- 每步 readout 误差 ≤ 1 次 FP16 截断 ≈ 1e-3，不会跨步累积
- 对比纯 NPU：state 每步累积 FP16 误差 → N 步后 ~N × 1e-3 → 很快发散

**实现要点**：

1. **export.py**: `export_model_hybrid` 导出，修改 `patched_recurrent` 函数，
   通过全局列表捕获中间量 (g_t, k_t, v_t, beta_t)，作为 IR 的额外输出。
   每个 GDN 层 4 个输出 × 18 层 = 72 个额外输出（每个 ≤8KB，总计 ~144KB）
2. **config.py**: 添加 `HYBRID_MODEL_DIR`、`HYBRID_OV_CONFIG`
3. **inference.py**: 检测 hybrid 模式（IR 含 `gdn_intermediate` 输出），
   维护 `_fp32_recurrent_states` FP32 数组，每步用中间量做 CPU FP32 更新

**导出命令**:
```powershell
uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_models --hf-model Qwen/Qwen3.5-0.8B --hybrid
```

**推理命令**:
```powershell
uv run python -m qwen35.scripts.run_inference --device HYBRID --prompt "Hello"
```

**模型目录**: `models/qwen35/Qwen3.5-0.8B-hybrid/`

**性能预估**：
- NPU 矩阵投影+conv+attention：~80% 计算量，FP16 全速
- CPU FP32 state 更新：~5% 计算量，~14.4M flops/step（NumPy <1ms）
- Host↔device 拷贝：每步 ~18MB state + ~144KB 中间量
- 预期速度：12-16 tok/s（介于纯 CPU 和纯 NPU 之间）

### 多子图 NPU 方案（Multi-Subgraph）— 实验中

**状态**: 代码已实现，待实际测试。

**核心思路**：将 24 层拆成 6 个子图（每组 4 层 = [GDN, GDN, GDN, FullAttn]），子图之间 hidden_state 回 CPU 转 FP32 再送下一子图。FP16 误差从 24 层累积降至 4 层累积（~0.2% vs ~1.2%）。数据传输开销仅 hidden_state [1,1,1024] = 2KB × 5 个边界 = 10KB/token，可忽略。

**实现**：
- `export.py`: `export_model_multisubgraph()` — 逐子图 trace + convert，每子图包含自己的 GDN state + KV cache + 中间量输出
- `inference.py`: `Qwen35MultiSubgraphModel` — 加载 6 个子图 IR，链式执行，FP32 shadow state + FP32 inter-subgraph hidden_states
- `config.py`: `MULTISUB_MODEL_DIR`、`MULTISUB_OV_CONFIG`
- `scripts/prepare_qwen35_models.py`: `--multisub` flag
- `scripts/run_inference.py`: `--device MULTISUB`

**导出命令**:
```powershell
uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_models --hf-model Qwen/Qwen3.5-0.8B --multisub
```

**推理命令**:
```powershell
uv run python -m qwen35.scripts.run_inference --device MULTISUB --prompt "Hello"
```

**采样测试**（Phase 0 诊断）：
```powershell
uv run python -m qwen35.scripts.run_inference --device HYBRID --temperature 0.6 --top-p 0.9 --prompt "Hello"
```

**模型目录**: `models/qwen35/Qwen3.5-0.8B-multisub/` (含 subgraph_0.xml ~ subgraph_5.xml)

**预期性能**：
- 6 个小子图 NPU 编译：首次 ~480s（每子图 ~80s），后续有 CACHE_DIR 缓存
- 推理速度：~12-15 tok/s（每 token 6 次 NPU infer + 5 次 FP32 hidden_state 传输）
- 精度改善：FP16 误差限于 4 层（vs 混合方案的 24 层），可能足以保持 token 一致性

## 导出流水线

`export.py` 执行自定义导出（不依赖 optimum-intel）：

### CPU/GPU 导出 (`export_model`)

1. 加载 PyTorch 模型 (`AutoModelForCausalLM`)
2. Patch forward：展开 cache_params 为 conv/recurrent/key/value 张量列表，接受 `inputs_embeds`（不是 `input_ids`）
3. Patch GDN 层：替换 CUDA-only 算子为 `RecurrentAttentionCell`（通过 ModuleExtension 转为 OV Loop 节点）
4. `ov.convert_model` 转 IR（支持动态 seq_len，batch prefill）
5. `stateful.py` 将 cache 输入/输出转为 OpenVINO stateful 变量 + beam_idx reorder

### NPU 导出 (`export_model_npu`) — 旧版全静态

1-2 同上
3. Patch GDN 层：使用 `patched_recurrent_gated_delta_rule_single_step`（内联单步计算，无 RecurrentAttentionCell）
4. `ov.convert_model` 转 IR（seq_len=1，无 Loop 节点，全静态 shapes，4D attention mask）
5. 无 stateful 变换 — Python 管理所有 state
6. 保存 FP16 压缩 IR + 提取 `embed_tokens.npy`

### NPUW_LLM 导出 (`export_model_npuw`) — 推荐

1-2 同上
3. Patch GDN 层：同 NPU 导出（单步内联，无 Loop）
4. Patch 注意力层：输出完整 KV concat（不是只输出 new token）
5. `ov.convert_model` 转 IR：
   - 动态 KV shapes（NPUW_LLM 做 `reshape_to_static`）
   - 标准 HF 命名（`past_key_values.{i}.key` / `present.{i}.key`）
   - GDN state 静态 shapes（dim2=128 固定）
   - 2D attention mask（标准 LLM 接口）
6. 无 stateful 变换 — NPUW_LLM 内部管理
7. 保存 FP16 压缩 IR + 提取 `embed_tokens.npy`

**NPUW_LLM 运行配置**：`NPUW_LLM=YES`, `NPUW_LLM_MAX_PROMPT_LEN=1`, `NPUW_FOLD=NO`

IR 输入签名：`inputs_embeds [B, seq_len, 1024]`, `attention_mask [B, mask_len]`, `position_ids [3, B, seq_len]`。
Embedding 查表在 Python 侧完成（从 `embed_tokens.npy` 查），text-only 和 VL 共用同一份 decoder IR。

## 模块说明

| 模块 | 职责 |
|------|------|
| `config.py` | 架构常量（层数、head 维度、层类型 pattern）+ 设备配置 |
| `inference.py` | `Qwen35OVModel` + `Qwen35MultiSubgraphModel` — 继承 `GenerationMixin`，管理 stateful/多子图推理 |
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
- **NPU 实验性**：需要使用 NPU IR（Loop-free），VL decoder 支持 token-by-token prefill
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
├── Qwen3.5-0.8B-npu/
│   ├── openvino_model.xml/.bin   # Loop-free 全静态 IR (FP16, seq_len=1, 4D mask)
│   ├── embed_tokens.npy          # 同 text-only
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── Qwen3.5-0.8B-npuw/
│   ├── openvino_model.xml/.bin   # NPUW_LLM 兼容 IR (动态 KV, HF 命名, 2D mask)
│   ├── embed_tokens.npy          # 同 text-only
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── Qwen3.5-0.8B-loop/
│   ├── openvino_model.xml/.bin   # 非 stateful Loop IR (18 Loop, 显式 I/O)
│   ├── embed_tokens.npy
│   └── ...
├── Qwen3.5-0.8B-hybrid/
│   ├── openvino_model.xml/.bin   # 混合 NPU+CPU IR (全静态, 72 个额外 GDN 中间量输出)
│   ├── embed_tokens.npy
│   └── ...
├── Qwen3.5-0.8B-ll2/
│   ├── openvino_model.xml/.bin   # LL2 后 IR (0 Loop, ReadValue/Assign 全静态)
│   ├── embed_tokens.npy
│   └── ...
├── Qwen3.5-0.8B-multisub/
│   ├── subgraph_{0..5}.xml/.bin  # 6 个子图 IR (每个 4 层, 全静态, GDN 中间量输出)
│   ├── embed_tokens.npy
│   └── ...
└── Qwen3.5-0.8B-vl/
    ├── vision_encoder.xml/.bin   # ViT (189 MB)
    ├── openvino_model.xml/.bin   # 同 text-only decoder（复制）
    ├── embed_tokens.npy          # 同 text-only（复制）
    ├── config.json
    ├── preprocessor_config.json
    ├── tokenizer.json
    └── tokenizer_config.json
```
