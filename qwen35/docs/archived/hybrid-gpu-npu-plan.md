# Qwen3.5 0.8B GPU+NPU 混合推理方案

## 背景

Qwen3.5-0.8B 有 24 层，层模式为（每 4 层一组，重复 6 次）：

```
GDN GDN GDN Attn | GDN GDN GDN Attn | ... | GDN GDN GDN Attn
 0   1   2   3      4   5   6   7          20  21  22  23
```

- **18 层 GDN**（线性注意力）：含 Loop 节点（动态迭代），需 FP32 累加精度
- **6 层 Full Attention**（标准 Transformer）：标准 SDPA，FP16 精度足够

当前状态：
- GPU 单独跑：12-13 tok/s（INT4），输出正确
- NPU 单独跑：不可用（GDN FP16 精度 + Loop 编译双重障碍）

目标：GDN 层跑 GPU（FP32 精度 + 支持 Loop），Attention 层跑 NPU（FP16 够用，无 Loop）。

---

## HETERO 方案结论：不可用

已在 OpenVINO 2026.0.0 上测试了 8 种 HETERO 配置，全部失败：

| 配置 | 结果 | 错误 |
|------|------|------|
| 全 GPU | 成功 | — |
| Layer 3 → NPU | 失败 | undeclared parameters: inputs_embeds |
| Layer 3 → **CPU** | 失败 | undeclared parameters: inputs_embeds |
| 仅 SDPA (6 ops) → NPU | 失败 | undeclared parameters: inputs_embeds |
| Attn + shared ancestors → NPU | 失败 | Cannot sort subgraphs |
| Attn + ancestors, 无 state → NPU | 失败 | undeclared parameters: inputs_embeds |
| Auto HETERO:NPU,GPU | 失败 | dynamic shape (Loop) |

**根因**：HETERO 子图分割器在将 ops 分到不同设备时，不能正确地将原模型的 Parameter 节点替换为子模型的 bridge input。这不是 NPU 特有问题（CPU 替代也失败），而是 HETERO 插件在含 Loop + Stateful + dynamic shapes 的复杂模型上的局限。

---

## 手动子图拆分方案

### 核心思路

在 **PyTorch 导出阶段** 把模型拆成独立子图，每个子图导出为独立 IR，分别编译到 GPU 或 NPU。推理时 Python 循环串联。

### 子图划分

```
                   ┌─────────────┐
  embed_tokens     │ Python/numpy │  (现有方案，不变)
                   └──────┬──────┘
                          │ inputs_embeds [1, seq, 1024]
                   ┌──────▼──────┐
  GDN Block 0      │ layers 0,1,2│ → GPU
  (3 GDN layers)   │ + Loop×3    │
                   └──────┬──────┘
                          │ hidden_state [1, seq, 1024]
                   ┌──────▼──────┐
  Attn Block 0     │ layer 3     │ → NPU
  (1 Attn layer)   │ SDPA + KV   │
                   └──────┬──────┘
                          │ hidden_state [1, seq, 1024]
                   ┌──────▼──────┐
  GDN Block 1      │ layers 4,5,6│ → GPU
                   └──────┬──────┘
                          │
                         ...  (重复 6 组)
                          │
                   ┌──────▼──────┐
  Attn Block 5     │ layer 23    │ → NPU
                   └──────┬──────┘
                          │ hidden_state [1, seq, 1024]
                   ┌──────▼──────┐
  Head             │ final_norm   │ → GPU
                   │ + lm_head    │
                   └──────┬──────┘
                          │ logits [1, seq, vocab]
```

总计 **13 个子图**：6 GDN blocks (GPU) + 6 Attention blocks (NPU) + 1 Head (GPU)。

### 每个子图的接口

#### GDN Block (3 layers) — GPU

```
输入:
  hidden_state      [B, seq_len, 1024]    float32
  attention_mask     [B, seq_len]           int64       (batch prefill mask)
  conv_state_0..2   [B, 6144, 4]           float32     ×3
  recurrent_state_0..2 [B, 16, 128, 128]   float32     ×3

输出:
  hidden_state      [B, seq_len, 1024]    float32
  conv_state_0..2   [B, 6144, 4]           float32     ×3
  recurrent_state_0..2 [B, 16, 128, 128]   float32     ×3
```

内部包含 3 层 GDN，每层：input_layernorm → conv1d → GDN recurrence (Loop) → norm → MLP → residual add。

#### Attention Block (1 layer) — NPU

```
输入:
  hidden_state      [B, seq_len, 1024]    float32
  attention_mask     [B, 1, seq_len, past+seq]  float32  (causal mask)
  position_ids      [3, B, seq_len]        int64       (mRoPE)
  key_cache         [B, 4, past_seq, 256]  float32
  value_cache       [B, 4, past_seq, 256]  float32

输出:
  hidden_state      [B, seq_len, 1024]    float32
  key_cache         [B, 4, past+seq, 256]  float32
  value_cache       [B, 4, past+seq, 256]  float32
```

内部包含 1 层标准 attention：input_layernorm → QKV proj → RoPE → SDPA → output proj → residual → MLP → residual。

**NPU 关键约束**：
- NPU 偏好静态 shape → `past_seq` 需要 pad 到固定最大值（如 256/512）
- 无 Loop 节点（纯标准 Transformer 算子）→ NPU 编译通过
- FP16 精度对标准 attention 完全足够

#### Head — GPU

```
输入:
  hidden_state      [B, seq_len, 1024]    float32

输出:
  logits            [B, seq_len, 248064]   float32
```

内部：RMSNorm → linear (vocab projection)。

### 导出改动

修改 `export.py`，新增 `export_split_subgraphs()` 函数：

```python
def export_split_subgraphs(hf_model, text_cfg, output_dir):
    """导出 13 个独立子图 IR。"""

    layer_types = text_cfg.layer_types  # ['linear_attention', ..., 'full_attention', ...]

    group_idx = 0
    gdn_layers = []

    for i, layer in enumerate(hf_model.model.layers):
        if layer_types[i] == "linear_attention":
            gdn_layers.append(layer)
        else:
            # 遇到 full_attention 层：先导出前面积攒的 GDN block
            if gdn_layers:
                export_gdn_block(gdn_layers, group_idx, text_cfg, output_dir)
                gdn_layers = []

            # 导出 attention block
            export_attn_block(layer, group_idx, text_cfg, output_dir)
            group_idx += 1

    # 最后一组 GDN（如果有）
    if gdn_layers:
        export_gdn_block(gdn_layers, group_idx, text_cfg, output_dir)

    # 导出 head
    export_head(hf_model.model.norm, hf_model.lm_head, text_cfg, output_dir)
```

每个 `export_gdn_block()` 内部：
1. 创建 wrapper Module 封装 3 层 GDN
2. Patch GDN 层（RecurrentAttentionCell → Loop，同现有逻辑）
3. `ov.convert_model()` with ModuleExtension/ConversionExtension
4. **不做** stateful 转换（state 是显式 input/output）
5. FP16 压缩保存

每个 `export_attn_block()` 内部：
1. 创建 wrapper Module 封装 1 层 Attention
2. 包含 RoPE 计算（从 position_ids 生成 cos/sin）
3. KV cache 作为显式 input/output（concat 式）
4. `ov.convert_model()` 标准导出
5. FP16 压缩保存

### 推理改动

新增 `Qwen35HybridModel` 类：

```python
class Qwen35HybridModel:
    def __init__(self, model_dir, ov_config=None):
        core = ov.Core()
        self.gdn_blocks = []   # 6 CompiledModel on GPU
        self.attn_blocks = []  # 6 CompiledModel on NPU
        self.head = None       # 1 CompiledModel on GPU

        for i in range(6):
            gdn_ir = core.read_model(f"{model_dir}/gdn_block_{i}.xml")
            self.gdn_blocks.append(core.compile_model(gdn_ir, "GPU"))

            attn_ir = core.read_model(f"{model_dir}/attn_block_{i}.xml")
            self.attn_blocks.append(core.compile_model(attn_ir, "NPU"))

        head_ir = core.read_model(f"{model_dir}/head.xml")
        self.head = core.compile_model(head_ir, "GPU")

    def forward(self, input_ids, past_states):
        hidden = self.embed_table[input_ids]  # [1, seq, 1024]

        for i in range(6):
            # GDN block: 3 layers on GPU
            gdn_out = self.gdn_blocks[i].infer({
                "hidden_state": hidden,
                "conv_state_0": past_states.conv[i*3],
                "conv_state_1": past_states.conv[i*3+1],
                "conv_state_2": past_states.conv[i*3+2],
                "recurrent_state_0": past_states.recurrent[i*3],
                "recurrent_state_1": past_states.recurrent[i*3+1],
                "recurrent_state_2": past_states.recurrent[i*3+2],
            })
            hidden = gdn_out["hidden_state"]
            # update conv/recurrent states...

            # Attention block: 1 layer on NPU
            attn_out = self.attn_blocks[i].infer({
                "hidden_state": hidden,
                "attention_mask": self.build_causal_mask(...),
                "position_ids": self.build_position_ids(...),
                "key_cache": past_states.key[i],
                "value_cache": past_states.value[i],
            })
            hidden = attn_out["hidden_state"]
            # update kv cache...

        logits = self.head.infer({"hidden_state": hidden})
        return logits["logits"]
```

### 模型文件布局

```
models/qwen35/Qwen3.5-0.8B-hybrid/
├── gdn_block_0.xml/.bin       # layers 0,1,2 → GPU
├── gdn_block_1.xml/.bin       # layers 4,5,6 → GPU
├── gdn_block_2.xml/.bin       # layers 8,9,10 → GPU
├── gdn_block_3.xml/.bin       # layers 12,13,14 → GPU
├── gdn_block_4.xml/.bin       # layers 16,17,18 → GPU
├── gdn_block_5.xml/.bin       # layers 20,21,22 → GPU
├── attn_block_0.xml/.bin      # layer 3 → NPU
├── attn_block_1.xml/.bin      # layer 7 → NPU
├── attn_block_2.xml/.bin      # layer 11 → NPU
├── attn_block_3.xml/.bin      # layer 15 → NPU
├── attn_block_4.xml/.bin      # layer 19 → NPU
├── attn_block_5.xml/.bin      # layer 23 → NPU
├── head.xml/.bin              # final_norm + lm_head → GPU
├── embed_tokens.npy           # 同现有
├── config.json
└── tokenizer.json
```

---

## 历史经验：NPU v2 多子图实验

之前做过类似拆分实验（已删除的 `export_npu_v2.py`/`inference_npu_v2.py`）：

| 配置 | 速度 | 输出 | 说明 |
|------|------|------|------|
| 单体 CPU | 7.8 tok/s | 正确 | 基线 |
| 6 子图 CPU | 4.2 tok/s | 正确 | **46% 慢** — Python 循环 + FP32 casting 开销 |
| 6 子图 NPU | 8.1 tok/s | 乱码 | GDN FP16 精度不足 |
| 单体 GPU (INT4) | 12-13 tok/s | 正确 | **当前最佳** |

**46% 慢的原因**：
1. 6 次 Host↔Device 往返 + Python 循环调度
2. 每次子图间 `astype(np.float32)` casting（FP32 shadow state）
3. 0.8B 模型太小，子图计算时间短，开销占比大

**本次不同**：
- 无需 FP32 casting（GDN 已在 GPU 上，天然 FP32）
- 统一内存不需要物理拷贝（只需同步）
- 但子图从 6 增加到 13，Python 循环多一倍

---

## 风险与预期

### 性能预估（Phase 1 实测后更新）

Phase 1 实测数据（单层 attention, `test_single_attn_npu.py`）：

| 设备 | 编译时间 | 延迟 (past=4) | 延迟 (past=128) |
|------|---------|---------------|-----------------|
| CPU | 0.1s | 2.22ms | — |
| GPU | 0.3s | 1.31ms | — |
| **NPU** | **1.1s** | **1.09ms** | **1.47ms** |

NPU 比 GPU 快 ~20%！基于实测修正预估：

| 项目 | 估算 |
|------|------|
| GDN block GPU infer (3 层) | ~10ms × 6 = 60ms |
| Attn block **NPU** infer (1 层) | **~1.5ms × 6 = 9ms** |
| Python 循环 + tensor 传递 | ~1-3ms × 12 = 12-36ms |
| GPU/NPU 同步延迟 | ~0.1ms × 12 = 1.2ms |
| **总计** | **~82-106ms/tok ≈ 9-12 tok/s** |

对比：单体 GPU 12-13 tok/s。混合方案可能接近甚至持平，如果 GDN block 的 GPU 推理本身够快。

### NPU 约束（Phase 1 确认）

1. **必须静态 shape** — NPU 编译器不接受全动态维度（`[?,?,?,?]`），需要 reshape 到固定尺寸
2. **KV cache 需要 padding** — past_seq 必须固定（如 256 或 512），推理时 pad 到该长度
3. **编译时间 ~1s/模型** — 6 个 attention block 编译 ~6s，一次性开销可接受
4. **精度OK** — FP16 max diff vs CPU 仅 0.002，对 attention 层完全够用

### 主要风险（更新）

1. **Python 循环开销** — 13 次 infer() 调用，每次有 dict 构建、numpy copy 等开销（最大风险）
2. ~~NPU 编译能否成功~~ → **Phase 1 已确认：成功**
3. ~~NPU KV cache 静态 shape~~ → **Phase 1 已确认：需要 padding，但可行**
4. **0.8B 太小** — 计算量不够大，开销占比高，收益可能为负

### 成功标准

- 输出与 GPU-only 一致（argmax 匹配）
- 速度 ≥ 10 tok/s（不低于 GPU-only 的 80%）
- NPU 利用率 > 0（证明 attention 确实跑在 NPU 上）

---

## 实施步骤

### Phase 1: 验证 NPU 能跑单层 attention ✅ 完成

测试脚本：`qwen35/scripts/test_single_attn_npu.py`

1. ✅ 从 PyTorch 导出 layer 3（first full_attention）为独立 IR
2. ✅ 确认无 Loop 节点（280 ops，标准 SDPA）
3. ✅ NPU 编译 + 推理成功（需 static shape）
4. ✅ NPU 单层延迟 1.09ms（比 GPU 1.31ms 还快 20%）

结论：**NPU 可以处理 attention 子图，混合方案可行。**

### Phase 2: 完整子图导出

1. `export.py` 新增 `export_split_subgraphs()`
2. GDN block 导出（复用现有 RecurrentAttentionCell + Loop 逻辑）
3. Attention block 导出（标准 SDPA，RoPE 内置，KV cache 显式 I/O）
4. Head 导出
5. CPU 验证：13 个子图串联输出与单体模型一致

注意事项（Phase 1 发现）：
- KV cache 输出需要显式 cat（不能依赖 Python list mutation 在 OV 中被 trace）
- NPU attention block 需要编译为固定 past_seq shape
- position_ids 格式为 [3, B, seq_len]（mRoPE），需要正确传递

### Phase 3: GPU+NPU 混合推理

1. `inference.py` 新增 `Qwen35HybridModel` 类
2. GDN blocks → GPU 编译（动态 shape，含 Loop）
3. Attention blocks → NPU 编译（静态 shape，padded KV cache）
4. 推理循环 + state 管理 + KV cache padding/unpadding
5. 对比输出正确性 + 测速

### Phase 4: 优化（如果 Phase 3 速度不达标）

1. 减少 Python 开销：预分配 tensor，复用 InferRequest
2. 合并 GDN + Attention 为 4 层组（减少到 7 子图），GDN 部分 GPU，attention 部分仍 NPU
3. NPU async infer：`start_async()` + pipeline
4. 最终评估是否值得（vs GPU-only 12-13 tok/s）
