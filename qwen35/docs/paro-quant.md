# PARO Quantization -- 实现文档

## 概述

PARO (Post-training Activation Rotation for Outlier-free quantization, ICLR 2026) 对权重做正交旋转，减少激活中的异常值（outlier），提升 INT4 量化精度。核心思想：

- **权重旋转**（离线）：`W_rot = W @ diag(1/cs) @ R^T`
- **激活旋转**（推理时）：`x_rot = x @ diag(cs) @ R^T`
- **数学等价**：`W_rot @ x_rot^T = W @ x^T`（旋转互相抵消）

旋转是 **block-diagonal** 的，group_size=128，每组独立的 128x128 正交矩阵。对于 Qwen3.5-0.8B（dim=1024），每个线性层有 8 个独立的旋转组。

## 架构

### RotatedLinear

`RotatedLinear`（`paro_rotation.py`）是 `nn.Linear` 的包装层，在前向传播中插入激活旋转。

**拆分 buffer 设计**：

```python
class RotatedLinear(nn.Module):
    def __init__(self, original_linear, R_blocks, channel_scales, group_size=128):
        super().__init__()
        # ...
        cs = np.asarray(channel_scales, dtype=np.float32).ravel()

        # 分离存储：channel_scales (1, dim) + R_blocks_T (ng, gs, gs)
        self.register_buffer(
            "channel_scales_buf", torch.from_numpy(cs).reshape(1, -1)
        )
        R_blocks_T = np.asarray(R_blocks, dtype=np.float32).transpose(0, 2, 1).copy()
        self.register_buffer(
            "R_blocks_T", torch.from_numpy(R_blocks_T)
        )

        # 离线预旋转权重
        W = original_linear.weight.data.float().numpy()
        W_rot = rotate_weight(W, R_blocks, cs, group_size)
        self.weight = nn.Parameter(torch.from_numpy(W_rot), requires_grad=False)
```

**Forward 路径**：

```python
def forward(self, x):
    shape = x.shape  # (..., dim)

    # Step 1: Channel scaling (逐元素乘)
    x_scaled = x * self.channel_scales_buf  # broadcast (1, dim)

    # Step 2: Block-diagonal rotation (BMM)
    x_flat = x_scaled.reshape(-1, self.num_groups, self.group_size)
    x_t = x_flat.permute(1, 0, 2)          # (ng, N, gs)
    x_rot = torch.bmm(x_t, self.R_blocks_T) # (ng, N, gs)
    x_rot = x_rot.permute(1, 0, 2)          # (N, ng, gs)

    # Step 3: 线性变换 (使用预旋转权重)
    x_out = x_rot.reshape(shape)
    return F.linear(x_out, self.weight, self.bias)
```

Tracing 后在 OpenVINO IR 中对应的 op 序列：`Multiply`（channel scales）-> `Reshape` -> `Transpose` -> `MatMul`（BMM rotation）-> `Transpose` -> `Reshape` -> `MatMul`（linear weight）。

**为什么拆分而非合并 `M_blocks = diag(cs) @ R^T`**：见下方 Bug 1。

### 权重预旋转

`rotate_weight()` 在导出前离线完成，推理时权重侧无额外计算：

```python
def rotate_weight(weight, R_blocks, channel_scales, group_size=128):
    """W_rot = W @ diag(1/cs) @ R^T (block-diagonal)."""
    W = np.asarray(weight, dtype=np.float64)  # FP64 精度计算
    W_rot = np.zeros_like(W)

    for g in range(num_groups):
        sl = slice(g * group_size, (g + 1) * group_size)
        inv_cs_g = inv_cs[sl]
        W_scaled = W[:, sl] * inv_cs_g[np.newaxis, :]
        W_rot[:, sl] = W_scaled @ R_blocks[g].T

    return W_rot.astype(np.float32)
```

关键点：

- 使用 **FP64** 计算旋转避免精度损失（Givens rotation 是连续左乘，误差会累积）
- 权重旋转后转回 FP32 存储
- `channel_scales` 的倒数 `1/cs` 应用在权重侧，`cs` 应用在激活侧，互相抵消

### PARO_SKIP_MODULES

```python
PARO_SKIP_MODULES = {"in_proj_a", "in_proj_b", "mlp.gate", "mlp.shared_expert_gate"}
```

这些层跳过旋转的原因：
- `in_proj_a`、`in_proj_b`：GDN 内部的低秩投影，维度小（非 group_size=128 的倍数）
- `mlp.gate`、`mlp.shared_expert_gate`：门控路由层，输出维度小且作为 sigmoid/softmax 的输入，对 outlier 不敏感

这些层在 PARO 模型的 `quantization_config.modules_to_not_convert` 中标记，不提供旋转参数。

## 导出流程

`export_hybrid.py` 中 PARO 相关步骤：

1. CLI 传入 `--paro-model` 指定 PARO 模型目录
2. 加载 PyTorch 模型后，调用 `extract_paro_params()` 提取旋转参数（theta、pairs、channel_scales）
3. 对每一层调用 `apply_paro_rotation_to_module()`，将 `nn.Linear` 替换为 `RotatedLinear`
4. 替换后的模型正常 trace 导出为 OpenVINO IR

```python
# export_hybrid.py L671-679
if paro_model:
    from qwen35.paro_rotation import extract_paro_params, apply_paro_rotation_to_module
    paro_params = extract_paro_params(paro_model)
    total_rotated = 0
    for i, layer in enumerate(model.model.layers):
        layer_prefix = f"layers.{i}"
        n = apply_paro_rotation_to_module(layer, paro_params, layer_prefix)
        total_rotated += n
```

**`--kv-update-method scatter_update_ext` 的重要性**：

导出 PARO 模型时，attention block 中的 KV cache 更新必须使用 `scatter_update_ext`（ConversionExtension -> ScatterUpdate-3），不能用默认的 `select`（torch.where）。`select` 方法内部的 `expand_as` 在 PARO 增加了 Reshape/Permute 节点后，OV 的 Broadcast shape 推导会失败。见 Bug 3。

完整导出命令：
```powershell
uv run --project qwen35 python -m qwen35.export_hybrid \
    --paro-model models/qwen35/Qwen3.5-0.8B-PARO \
    --kv-update-method scatter_update_ext
```

## 量化保护

### find_paro_rotation_ops()

PARO 的旋转矩阵 `R_blocks_T` 和 `channel_scales_buf` 在 IR 中表现为 Constant 节点。如果 `nncf.compress_weights()` 把它们当权重量化，旋转精度会被破坏。`find_paro_rotation_ops()` 检测这些节点并通过 `nncf.IgnoredScope` 排除。

```python
def find_paro_rotation_ops(model, group_size=128):
    """查找 PARO 旋转矩阵的 MatMul/Multiply ops。"""
    names = []
    for op in model.get_ops():
        op_type = op.get_type_name()
        if op_type not in ("MatMul", "Multiply"):
            continue
        for input_idx in range(op.get_input_size()):
            source = op.input(input_idx).get_source_output().get_node()
            # 穿透 Convert 节点链 (FP16 压缩插入的)
            while source.get_type_name() == "Convert":
                source = source.input(0).get_source_output().get_node()
            if source.get_type_name() != "Constant":
                continue
            shape = list(source.get_output_shape(0))
            # 匹配旋转矩阵: (num_groups, group_size, group_size)
            if (len(shape) == 3
                and shape[1] == group_size
                and shape[2] == group_size
                and shape[0] >= 1):
                names.append(op.get_friendly_name())
                break
            # 匹配 channel_scales: (1, dim) where dim % group_size == 0
            if (op_type == "Multiply"
                and len(shape) == 2
                and shape[0] == 1
                and shape[1] >= group_size
                and shape[1] % group_size == 0):
                names.append(op.get_friendly_name())
                break
    return names
```

检测逻辑：
- **旋转矩阵 `R_blocks_T`**：shape `(num_groups, 128, 128)`，作为 MatMul 的 Constant 输入
- **channel_scales**：shape `(1, dim)`，作为 Multiply 的 Constant 输入，`dim` 是 `group_size` 的倍数

### Convert 节点链处理

`ov.save_model(..., compress_to_fp16=True)` 在 FP32 Constant 节点后插入 `Convert(FP32->FP16)` 节点。检测时需要 **穿透 Convert 链** 才能找到底层 Constant：

```python
# 关键：while 循环穿透 Convert 链
while source.get_type_name() == "Convert":
    source = source.input(0).get_source_output().get_node()
```

### nncf.IgnoredScope 集成

`compress_ir()` 中的集成方式：

```python
# quantize_hybrid.py L140-143
rotation_ops = find_paro_rotation_ops(model)
if rotation_ops:
    kwargs["ignored_scope"] = nncf.IgnoredScope(names=rotation_ops)
    print(f" (protecting {len(rotation_ops)} PARO rotation ops)", end="")
```

### 每个 block 的 PARO op 数量

对于 Qwen3.5-0.8B：
- 每个 **Attention block**：包含 `q_proj`、`k_proj`、`v_proj`、`o_proj` + `gate_proj`、`up_proj`、`down_proj` = 7 个线性层，每层 2 个 PARO ops（1 Multiply + 1 MatMul），共 **14 个 PARO ops**
- 每个 **GDN block**（3 层）：每层有 `in_proj_qkv`、`out_proj` + `gate_proj`、`up_proj`、`down_proj` = 5 个旋转层（`in_proj_a`/`in_proj_b` 跳过），每层 10 ops，共 **30 个 PARO ops**
- **Head block**：`lm_head` 1 层，**2 个 PARO ops**

## NPU 静态 shape 适配

`inference_hybrid.py` 中 `_reshape_attn_static()` 对 NPU 编译的 attention IR 做静态 shape 设置。PARO 引入的 Reshape/Permute/BMM 节点使得 `ir.reshape()` 方法无法正确推导 Broadcast 节点的 shape，因此改用逐 input 的 `set_partial_shape()` + `validate_nodes_and_infer_types()`：

```python
def _reshape_attn_static(self, ir, past_seq: int, seq_len: int = 1):
    """Uses set_partial_shape() + validate_nodes_and_infer_types() instead of
    ir.reshape() to avoid breaking Broadcast nodes when PARO RotatedLinear
    adds reshape/permute/bmm ops that confuse OpenVINO's shape propagation."""
    B, S = 1, seq_len
    shape_map = {
        0: [B, S, self._hidden_size],
        1: [3, B, S],
        2: [B, self._num_kv_heads, past_seq, self._head_dim],
        3: [B, self._num_kv_heads, past_seq, self._head_dim],
        4: [S],
        5: [B, 1, S, past_seq],
    }
    for i, shape in shape_map.items():
        ir.inputs[i].get_node().set_partial_shape(ov.PartialShape(shape))
    ir.validate_nodes_and_infer_types()
    return ir
```

## 已修复的 Bug

### Bug 1: M_blocks FP16 精度损失

**问题**：最初实现将 `channel_scales` 和旋转矩阵合并为单一矩阵 `M_blocks = diag(cs) @ R^T`，作为 RotatedLinear 的唯一 buffer。导出 IR 时 `compress_to_fp16=True` 将 M_blocks 压缩到 FP16 后，24 层串联推理结果严重偏离 baseline。

**根因**：`channel_scales` 值范围 `[-5.4, 21.6]`，`R^T` 值范围 `[-1, 1]`。合并后 `M_blocks` 的元素混合了两种尺度。128 维点积在 FP16 下的累积误差约 0.038/element，经过 24 层串联后误差发散。

对比拆分方案：
- `channel_scales_buf`（值 ~20）：FP16 精度 ~0.07%（安全）
- `R_blocks_T`（值在 [-1, 1]）：FP16 精度 ~0.1%（安全）
- 合并的 `M_blocks`（值范围混合）：FP16 精度在点积中累积，~3.8% 误差/层

**修复**：拆分为两个独立 buffer `channel_scales_buf` + `R_blocks_T`，forward 中先 element-wise multiply 再 BMM。各自的值域在 FP16 范围内安全。

```python
# 修复前（合并）：
self.register_buffer("M_blocks", torch.from_numpy(M_blocks))  # diag(cs) @ R^T
# forward: x_rot = bmm(x, M_blocks)  <-- FP16 后累积误差

# 修复后（拆分）：
self.register_buffer("channel_scales_buf", ...)  # (1, dim), 值 ~20
self.register_buffer("R_blocks_T", ...)           # (ng, gs, gs), 值 [-1,1]
# forward: x_scaled = x * channel_scales_buf; x_rot = bmm(x_scaled, R_blocks_T)
```

### Bug 2: NNCF 量化旋转矩阵

**问题**：`nncf.compress_weights()` 将 `R_blocks_T` 旋转矩阵当作普通权重进行 INT4/INT8 压缩。旋转矩阵值在 `[-1, 1]` 范围内，量化后的反量化误差破坏正交性，导致 `W_rot @ x_rot^T != W @ x^T`。

**根因**：`find_paro_rotation_ops()` 最初直接查找 MatMul 输入的 Constant 节点。但 `ov.save_model(..., compress_to_fp16=True)` 在 Constant 和 MatMul 之间插入了 `Convert(FP32->FP16)` 节点，形成链式结构：

```
Constant(FP16) -> Convert(FP16->FP32) -> MatMul
```

检测代码只看 MatMul 的直接输入（Convert 节点），不是 Constant，所以匹配失败。旋转矩阵未被保护，被 NNCF 量化。

**修复**：添加 `while` 循环穿透任意长度的 Convert 链，找到底层 Constant 节点：

```python
# 修复前：只检查直接输入
source = op.input(input_idx).get_source_output().get_node()
if source.get_type_name() != "Constant":
    continue

# 修复后：穿透 Convert 链
source = op.input(input_idx).get_source_output().get_node()
while source.get_type_name() == "Convert":
    source = source.input(0).get_source_output().get_node()
if source.get_type_name() != "Constant":
    continue
```

### Bug 3: Broadcast shape 推导失败

**问题**：导出 PARO 模型时，如果使用默认的 `--kv-update-method select`（torch.where），attention block 导出在 `ov.convert_model()` 阶段报 Broadcast shape 错误。

**根因**：`select` 方法中的 `expand_as` 操作需要 OV 推导 Broadcast 的目标 shape。PARO 的 RotatedLinear 在计算路径中插入了 Reshape -> Permute -> BMM -> Permute -> Reshape 序列，使得 OV 的 shape 推导系统对 Broadcast 的兼容性判断失败。

同样，`_reshape_attn_static()` 原本使用的 `ir.reshape()` 方法也会在 PARO 模型上失败（Broadcast 节点无法推导新 shape），需要改用逐 input 的 `set_partial_shape()` + `validate_nodes_and_infer_types()`。

**修复**：导出时必须指定 `--kv-update-method scatter_update_ext`。`scatter_update_ext` 使用 ConversionExtension 直接生成 ScatterUpdate-3 op，不依赖 `expand_as`/Broadcast，避免 shape 推导问题。

```bash
# 错误（会触发 Broadcast shape 失败）：
uv run --project qwen35 python -m qwen35.export_hybrid --paro-model ...
# 默认 --kv-update-method select

# 正确：
uv run --project qwen35 python -m qwen35.export_hybrid \
    --paro-model ... --kv-update-method scatter_update_ext
```

## 性能数据

PARO 模型 vs baseline（Qwen3.5-0.8B-hybrid，HYBRID GPU+NPU，Python 推理）：

| 配置 | Decode 速度 | 说明 |
|------|------------|------|
| Baseline（无 PARO） | ~16.8 tok/s | FP16 hybrid |
| PARO | ~14.6 tok/s | 慢 ~13% |

额外开销来源：
- 每个旋转层增加 1 个 **Multiply**（channel scales broadcast）+ 1 个 **MatMul/BMM**（128x128 旋转）
- Qwen3.5-0.8B 共 ~120 个旋转层，decode 时每 token 增加 ~240 个额外 op

PARO 的价值不在于加速，而在于**更激进量化时的精度保持**。当对 GDN 做 INT4 量化时，PARO 预旋转减少了激活 outlier，使得量化误差更小、输出质量更高。

## 测试

`qwen35/scripts/test_paro_rotation.py` 提供 5 级渐进测试：

1. **旋转矩阵性质**：正交性（`R @ R^T = I`）、无 NaN/Inf、channel_scales 范围合理
2. **单层等价性**：`RotatedLinear(x)` 与 `original_linear(x)` 数值匹配（FP32，max_diff < 0.01）
3. **全模型等价性**：所有层旋转后 logits 与 baseline 的 cosine_sim > 0.99
4. **逐层误差累积**：逐层应用旋转，定位误差最大的层（仅在 Test 3 失败时运行）
5. **参数映射审计**：确认 PARO 参数覆盖所有应旋转的层，无遗漏

运行命令：
```powershell
uv run --project qwen35 python -m qwen35.scripts.test_paro_rotation
```
