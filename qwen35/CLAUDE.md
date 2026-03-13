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
- 推理时 chunked prefill（S=16 chunks + S=1 remainder），decode 单步推进

## 设备兼容性

| 设备 | 状态 | Prefill | Decode | 备注 |
|------|------|---------|--------|------|
| CPU | 正常 | ~1.5 tok/s | ~3.6 tok/s | 稳定 |
| **GPU_ONLY** | **正常** | **2.2 tok/s** | **4.9 tok/s** | 全 stateful，ScatterUpdate-3 KV cache（量化模型） |
| **Single-IR GPU FP16** | **正常** | — | **7.4 tok/s avg** | 不拆图 baseline，单 IR stateful |
| **Single-IR GPU INT4** | **正常** | — | **8.8 tok/s avg** | 不拆图 INT4 baseline |
| **HYBRID (GPU+NPU) Python** | **正常** | — | **8.5 tok/s avg** | GDN→GPU(stateful), Attn→NPU(explicit), layer-major prefill |
| **HYBRID (GPU+NPU) C++** | **正常** | — | **13.3 tok/s avg** | 量化(INT4/INT8) + 零分配热循环。冷态 16.7，热态 7.0（Lunar Lake 热节流） |
| NPU only | 不可用 | — | — | GDN 递归 FP16 精度不足 |

详细多轮 benchmark 数据见 `benchmark.md`。

### NPU 单独跑 GDN 不行，但 attention 子图可以

Intel NPU 不能单独跑完整模型（两个障碍）：
1. **FP16 精度不足**：GDN delta rule 线性递归每步乘法放大舍入误差，18 层串联后发散
2. **NPU 编译器不支持 Loop 节点**：GDN 递归在 IR 中为 Loop，NPU 报 `to_shape was called on a dynamic shape`

但 attention 子图（标准 SDPA，无 Loop）**可以在 NPU 运行**，FP16 精度完全够用（max diff 0.002）。
因此采用 **Hybrid 方案**：GDN blocks → GPU，Attention blocks → NPU，两者并行利用。

对比：高通 SM8650 NPU 有 FP32 累加器 + TFLite 编译时展开 Loop → GDN 可正常推理。详见 `intel-npu-gdn.md`。

## Hybrid GPU+NPU 推理（19 子图方案）

### 架构

将 24 层拆为 19 个独立子图 IR：

| 子图 | 数量 | 设备 | State 管理 | 内容 |
|------|------|------|-----------|------|
| GDN block (decode) | 6 | GPU | **Stateful**（ReadValue/Assign，state 常驻 GPU 显存） | 每块 3 层 GDN（含 Loop 节点，需 FP32） |
| GDN prefill block | 6 | GPU | **Explicit I/O**（state 显式传入传出） | 每块 3 层 GDN（chunkwise parallel，无 Loop，90 MatMul） |
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
uv run --project qwen35 python -m qwen35.export_hybrid --kv-update-method scatter_update_ext
```

**推理**（根 venv）：
```powershell
uv run python -m qwen35.inference_hybrid --prompt "你好" --device HYBRID --no-attn-stateful --prefill-chunk-size 16
```

也支持 `GPU_ONLY`（全部子图在 GPU）和 `CPU_ONLY` 模式。`--prefill-chunk-size 1` 回退到 token-by-token prefill。

### NPU Attention 关键设计

1. **静态 shape**：NPU 要求 static shape，KV cache 固定到 `MAX_CACHE=256`，不足部分零填充。编译多套模型：S=1（decode）+ S=16/8/4/2（prefill，descending powers of 2 消除 token-by-token remainder）
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

### ConversionExtension 经验（KV cache 更新方法）

#### KV cache 更新方法对比

| PyTorch 操作 | OV IR 操作 | GPU 正确性 | NPU 正确性 | GPU stateful 速度 | 备注 |
|-------------|-----------|-----------|-----------|-----------------|------|
| `torch.where` (select) | Select | ✓ | ✓ | 13.6 tok/s | 慢且不支持 S>1 prefill（expand_as 失败） |
| `torch.index_copy_` | ScatterElementsUpdate | ✓ | ✗ 乱码 | — | NPU 有 bug |
| `torch.scatter` | ScatterElementsUpdate | ✓ | ✗ 乱码 | — | NPU 有 bug |
| `torch.index_put` | — | — | — | — | OV 前端不支持 |
| **ConversionExtension** | **ScatterUpdate-3** | **✓** | **✓** | **22.6 tok/s** | **最优，GPU +66%** |

#### ConversionExtension 模式

当 PyTorch 操作无法直接映射到想要的 OV IR op 时，可以用 `ModuleExtension` + `ConversionExtension` 绕过：

1. 创建自定义 `nn.Module`（如 `KVCacheScatterUpdate`），`forward()` 只用于 tracing shape 推断
2. `ModuleExtension(MyModule, "MyModuleOp")` — 告诉 OV 前端拦截该模块
3. `ConversionExtension("MyModuleOp", convert_fn)` — 用 `openvino.opset14` 构造任意 IR 节点
4. 传入 `ov.convert_model(..., extension=[...])` 即可

关键区别：
- **ScatterUpdate-3**（`ops.scatter_update`）：axis-based slice 替换，高效
- **ScatterElementsUpdate**（`torch.scatter`）：element-wise 散射，NPU 有 bug
- **ScatterNDUpdate**（`ops.scatter_nd_update`）：N维索引替换，需 transpose 技巧

#### NPU Stateful 限制

NPU stateful（ReadValue/Assign）在子图模式下有固有开销：
- NPU stateful 比 explicit 慢 ~2x（ScatterUpdate: 13.5 vs 17.6 tok/s）
- 即使更新操作是 NOP，ReadValue + Assign 的管理成本 > host<->NPU 传输
- NPU stateful 输出在长序列后偶尔有精度偏差

结论：**NPU attention 子图用 explicit I/O + ScatterUpdate-3 是目前最优方案**。

### HYBRID NPU 未来优化方向

#### 已验证可行

1. **ScatterUpdate-3 KV cache** ✓ — GPU_ONLY 13.6→22.6 tok/s (+66%)，HYBRID 15.3→17.6 tok/s (+15%)
2. **NPU 模型缓存** ✓ — `core.set_property({"CACHE_DIR": ...})` 全局设置，首次编译后 <1s 启动
3. **PREFER_PLUGIN 编译器** ✓ — 已在 NPU attn block 编译时设置 `"NPU_COMPILER_TYPE": "PREFER_PLUGIN"`
4. **Chunked prefill (S=16,8,4,2)** ✓ — NPU 编译多套 attention 模型（descending powers of 2），prefill 10.9→15.0 tok/s (+38%)。44 token prompt: 16+16+8+4 = 4 chunks（vs 旧版 16+16+12×S=1 = 14 forward calls）。首次编译 +7s，后续 cached <1s。`--prefill-chunk-size 16`
5. **Layer-major prefill** ✓ — 将 chunk-major（每个 chunk 穿过所有 13 层）改为 layer-major（每层处理所有 token），GDN 用 full-batch（S=prompt_len，一次 GPU infer），Attn 仍分 chunk（NPU static shapes）。Head 只在最后一个 token 运行一次。**HYBRID 1.3-1.4x prefill 加速**（41 tok: 900→678ms），GPU_ONLY 同样受益。优化来源：6x 更少 GDN kernel dispatch + 跳过 N-1 次 Head + GDN Loop 单次启动摊销
6. **Chunkwise parallel GDN prefill** ✓ — 用 WY representation 并行化 GDN prefill，消除 Loop 节点。`ChunkwiseRecurrentAttentionCell`（`chunkwise_gdn.py`）用 MatMul+tril+cumsum+exp 替代逐 token Loop，OV 直接 trace（无需 ModuleExtension）。导出为 `gdn_prefill_block_{0-5}.xml`（888 ops, 90 MatMul, 0 Loop），推理时 prefill 用 chunkwise blocks、decode 仍用 Loop blocks。**Prefill 2x 加速**（49 tok: 11.6→23.6 tok/s）。Neumann series 8 步（支持 T≤256），需 FP32 精度（`compress_to_fp16=False` + `INFERENCE_PRECISION_HINT=f32`）。Prefill 后通过 `query_state()` 将 chunkwise explicit states 注入 stateful decode blocks

#### 已验证无效（Lunar Lake 统一内存架构下的反优化）

4. **GPU-side argmax** ✗ — 在 HeadWrapper IR 中加 `torch.argmax` 导致 Head 从 6ms → 34ms（+28ms）。Intel iGPU 的 argmax kernel 对 248320 元素极慢。而 Lunar Lake 统一内存下 GPU→CPU 传 1MB logits 几乎零成本（cache coherent，非 PCIe），CPU numpy argmax 仅 ~0.1ms。**结论：统一内存架构下 argmax 应留在 CPU**
5. **Embedding 上 GPU** ✗ — GPU Gather 子图 0.6ms vs CPU numpy 索引 0.02ms（慢 30x）。额外的 OV model launch + GPU kernel 调度开销远大于收益。且 485MB embedding 表常驻 GPU VRAM 增加显存压力。**结论：embedding 查表应留在 CPU numpy**
6. **NPU set_output_tensor 零拷贝 KV** ✗ — 对 NPU attn block 用 `set_output_tensor(1/2, kv_tensors)` 消除 KV memcpy，但因 input/output 指向同一 host buffer，NPU 插件内部增加了额外拷贝来处理 aliased memory，导致 decode 从 ~17 tok/s 降到 ~9 tok/s（2x 减速）。**结论：NPU 不适合 aliased input/output，保留显式 memcpy**。GPU `set_output_tensor` 无此问题（GPU 插件用独立 GPU 内存做中间计算）
7. **GPU attention prefill** ✗ — 在 HYBRID 模式 prefill 时将 attention 放 GPU（动态 shape，一次调用处理全 prompt）而非 NPU（chunked S=16/8/4/2）。结果 prefill 反而 0.79x 更慢。原因：GPU 已在跑 GDN（占 73% 时间），再加 attention 造成 GPU 争抢。NPU chunked prefill 已经和 GPU_ONLY 持平 (1.01x)。**结论：HYBRID prefill 时 attention 应保留在 NPU**。代码仍保留（`--no-attn-gpu-prefill` 默认关闭），可供对比
8. **PERFORMANCE_HINT: LATENCY** ✗ — 在 Python inference 中对 GPU/NPU 所有子图设置 `PERFORMANCE_HINT: LATENCY`（C++ 版已使用）。结果 Python 版 decode 从 ~14 tok/s 降到 ~9.7 tok/s（**30% 退化**）。C++ 用 LATENCY 有效可能因为 C++ 调用路径不同。**结论：Python OpenVINO API 不应设置 LATENCY hint**
9. **Prefill 后释放 GPU 内存** ✗ — Prefill 完成后释放 GDN prefill blocks + attention prefill blocks（~1.5GB GPU）。A/B 测试无改善（0.94x），GPU memory pressure 假说不成立。**结论：OpenVINO CompiledModel 内存不影响其他模型推理速度**
10. **GPU stateful attention for decode** ✗ — 将 decode 阶段的 attention 从 NPU（explicit I/O + memcpy）改为 GPU（MakeStateful，零 memcpy），期望消除 ~1MB KV cache memcpy/block。结果 decode 从 21 降到 10.5 tok/s（**50% 退化**）。原因：Lunar Lake iGPU 资源有限，额外 6 个 GPU attention 模型导致所有 GPU 操作（包括 GDN）变慢 2.2x。**结论：iGPU 上不能同时运行 GDN + attention on GPU，即使是串行执行。NPU attention 虽有 memcpy 开销，但整体更快**

#### 待尝试优化

所有主要优化方向已探索完毕。剩余方向均为增量改善：
- ~~**Async pipeline**~~ ✗ — 数据依赖链 `gdn_i → attn_i → gdn_{i+1}` 严格串行，无法并行
- ~~**KV cache INT8 量化**~~ → 权重量化已足够
- ✓ **Attention block 权重 INT4** → 已验证 INT4_SYM 安全
- ~~**Batch prefill**~~ → 已实现为 Chunked prefill
- ~~**GPU attention prefill**~~ ✗ → GPU 争抢
- ~~**GPU attention decode**~~ ✗ → GPU 资源争抢，2.2x 退化
- ~~**LATENCY hint**~~ ✗ → Python API 不兼容

#### 瓶颈分析（多轮 benchmark 实测，2026-03-14）

**C++ HYBRID decode**（3 轮平均 13.3 tok/s，量化 INT4+INT8）：
- GDN blocks x 6：63.2ms（**79.6%**）— 绝对主导瓶颈，~10.5ms/block
- Attn blocks x 6：12.6ms（15.8%）— NPU ~2.1ms/block（非常稳定）
- Head：3.5ms（4.4%）
- Overhead：<0.1ms — 零分配热循环、增量 mask、无 timing 开销
- **Total: 79.3ms/token（--timing 模式）**

**热节流影响**：同一配置 3 轮测试，decode 从 8.2 到 10.8 tok/s（30% 波动）。30 秒冷却后恢复。连续推理以 ~8-9 tok/s 为准。

**关键结论**：GDN 占 82%，是唯一值得优化的目标。GDN 需要 FP32 + Loop 在 GPU 上运行，优化空间接近极限。详细数据见 `benchmark.md`。

### 量化实验结果

**工具**:
- `qwen35/scripts/quantize_hybrid.py` — 对已导出的 hybrid IR 做 per-subgraph NNCF 权重压缩（含 GDN prefill blocks）
- `qwen35/scripts/test_quality.py` — 6 个多样 prompt 质量对比测试

**量化配方**:
```powershell
# 推荐：全量化（Attn INT4 + GDN INT8 + Head INT4）
uv run python -m qwen35.scripts.quantize_hybrid --attn-mode int4_sym --gdn-mode int8_sym --head-mode int4_sym
# 质量测试（与 FP16 baseline 对比）
uv run python -m qwen35.scripts.test_quality --model-dir models/qwen35/Qwen3.5-0.8B-hybrid-attn-int4sym-gdn-int8sym-head-int4sym --baseline-dir models/qwen35/Qwen3.5-0.8B-hybrid
```

**测试结果（6 prompt, HYBRID NPU+GPU, 80 tokens/prompt）**:

| 配置 | 模型大小 | C++ Decode avg | Python Decode avg | 质量偏差 |
|------|---------|---------------|------------------|---------|
| FP16 baseline | 5.5 GB | — | ~7.5 tok/s | — |
| **Attn INT4 + GDN INT8 + Head INT4** | **3.4 GB** | **13.3 tok/s** | **8.5 tok/s** | 2/6 divergent（语义正确） |
| Single-IR GPU INT4 | ~0.5 GB | — | 8.8 tok/s | 不拆图 baseline |

*以上为 3 轮平均，23 tok prompt → 80 tok decode。详细数据见 `benchmark.md`。*

**关键发现**:
- **GDN INT8_SYM 安全** — 递归精度未受影响，reasoning prompt（数学题）输出与 baseline 完全一致
- **Attn INT4_SYM 安全** — 标准 Transformer 对 INT4 鲁棒性好，NPU 上 INT4_SYM 是最优精度
- **Head INT4_SYM** — NNCF 对 248k vocab 投影做 INT4，减少 ~700MB
- **量化 HYBRID > FP16 HYBRID > GPU_ONLY** — NPU 分担 Attn + 权重压缩减少 GPU 带宽占用
- **divergent ≠ 错误** — 偏差仅在措辞上（如"北京"回答方式不同），语义完全正确
- **推荐配方**: `--attn-mode int4_sym --gdn-mode int8_sym --head-mode int4_sym`

### 模块说明

| 模块 | 职责 |
|------|------|
| `export_hybrid.py` | 导出 19 子图 IR（GDN decode/prefill + Attention + Head blocks） |
| `chunkwise_gdn.py` | `ChunkwiseRecurrentAttentionCell` — WY representation 并行 GDN prefill（无 Loop，纯 MatMul） |
| `inference_hybrid.py` | `Qwen35HybridModel` — 编排 19 子图推理，chunkwise GDN prefill + Loop GDN decode + chunked Attn |
| `cpp/` | C++ hybrid 推理（同架构，~9 tok/s avg decode），见下方 |
| `benchmark.md` | 多轮 benchmark 结果、时间分解、热节流分析 |
| `scripts/run_inference.py` | 统一入口，`--device HYBRID/GPU_ONLY/CPU_ONLY` |
| `scripts/quantize_hybrid.py` | Per-subgraph NNCF 权重量化（Attn INT4 / GDN INT8 / Head INT4） |
| `scripts/test_quality.py` | 6 prompt 质量对比测试，支持 baseline side-by-side |

### C++ Hybrid 推理

C++ 实现与 Python 使用相同的 19 子图架构（6 GDN decode + 6 GDN prefill + 6 Attn + 1 Head），额外优化：
- GPU `set_output_tensor` 零拷贝（GDN hidden 输出直接写入 host buffer）
- `NUM_REQUESTS=1` 减少资源争抢（`LATENCY` hint 默认关闭，可通过 `--latency` 开启）
- 预分配所有 tensor wrapper，decode 热循环零堆分配
- Chunkwise parallel GDN prefill（无 Loop，MatMul-based，需 FP32）
- Layer-major prefill + 只对最后一个 token 运行 Head
- 模型加载顺序优化：NPU 先（给 GPU 散热时间），GPU 后
- NPU `set_output_tensor` 会 2x 减速（KV aliased buffer 触发内部拷贝），保留 memcpy
- NPU hidden 输出 shape 与输入不同（S=1 时 [B,H] 而非 [B,1,H]），必须 memcpy

**性能（量化模型 INT4+INT8，3.4GB，3 轮平均，2026-03-14）：**
- **Decode**: 13.3 tok/s avg（冷态 16.7，热态 7.0）
- **Prefill**: 5 token prompt, ~900ms 冷态
- **时间分解**: GDN 63.2ms (80%) + Attn 12.6ms (16%) + Head 3.5ms (4%) = 79.3ms/tok

**注意**：Lunar Lake iGPU 热节流导致 2.4x 性能波动（冷态 16.7 vs 热态 7.0 tok/s）。30 秒冷却可恢复。连续推理以 sustained 为准。

**构建与运行：**
```powershell
.\qwen35\cpp\build.ps1                              # 编译
.\qwen35\cpp\run.ps1 -Prompt "Hello" -MaxTokens 50  # 运行 C++（默认量化模型）
.\qwen35\cpp\run-python.ps1 -Prompt "Hello"          # 对比 Python
.\qwen35\cpp\clear-cache.ps1                         # 清缓存（改编译参数后需要）
# 选项：--latency（启用 LATENCY hint）、--no-gdn-prefill（跳过 chunkwise prefill）
```

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
| `export_hybrid.py` | 导出 19 子图 IR（GDN decode/prefill + Attention + Head blocks），显式 I/O |
| `inference_hybrid.py` | `Qwen35HybridModel` — 编排 19 子图混合推理 |

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
├── Qwen3.5-0.8B-hybrid/          # 19 子图（Hybrid GPU+NPU 推理）
│   ├── gdn_block_{0-5}.xml/.bin  # 6 个 GDN decode 子图 (各 3 层, Loop, GPU stateful)
│   ├── gdn_prefill_block_{0-5}.xml/.bin # 6 个 GDN prefill 子图 (chunkwise parallel, GPU explicit I/O)
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

## 归档文档

以下规划/研究文档已完成使命，移至 `docs/archived/`：

| 文档 | 内容 | 归档原因 |
|------|------|---------|
| `mtp.md` | MTP speculative decoding 实现计划 | 已实现，经验记录在 `memory/mtp-speculative-decoding.md` |
| `new-optimization.md` | MTP + Loop-free S=1 优化方向 | MTP 无效，Loop-free 反慢 21% |
| `perf-analysis.md` | 早期 HYBRID vs GPU_ONLY 分析 | 数据已过时（HYBRID 现已优于 GPU_ONLY） |
| `chunkwise-parallel-gdn.md` | WY representation 并行 GDN 方案 | 已实现并集成到代码中 |
| `hybrid-gpu-npu-plan.md` | GPU+NPU 混合推理原始规划 | 所有 phase 已完成 |
| `attn-export-issue.md` | Attention 子图导出 bug report | 已修复 |
