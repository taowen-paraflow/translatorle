# Attention 子图导出问题：KV cache concat 丢失 + 输出维度错误

## 背景

我们把 Qwen3.5-0.8B（24 层 Transformer 变体）拆成 13 个独立子图，分别导出为 OpenVINO IR。其中 6 个 attention 子图的导出有问题。

**环境**: OpenVINO 2026.0.0, PyTorch (transformers 5.3), Python 3.12

### Attention 子图做什么

每个 attention 子图包装 1 个标准 Transformer DecoderLayer（含 RMSNorm + Multi-Head Attention + MLP），KV cache 作为显式输入/输出（不用 OpenVINO Stateful 机制）。

PyTorch wrapper 代码：

```python
class KVCache:
    """单层 KV cache，用 list + torch.cat 实现。"""
    def __init__(self, k, v):
        self.key_cache = [k]
        self.value_cache = [v]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        v = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        self.key_cache[layer_idx] = k   # Python list mutation
        self.value_cache[layer_idx] = v
        return k, v

    def get_seq_length(self, layer_idx=0):
        return self.key_cache[layer_idx].shape[2]

class AttnBlockWrapper(nn.Module):
    def __init__(self, layer, rotary_emb):
        super().__init__()
        self.layer = layer        # HuggingFace Qwen3_5DecoderLayer
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states, position_ids, key_cache, value_cache):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cache = KVCache(key_cache, value_cache)

        out = self.layer(
            hidden_states,
            position_embeddings=(cos, sin),
            past_key_value=cache,
            use_cache=True,
        )

        return out[0], cache.key_cache[0], cache.value_cache[0]
```

导出代码：

```python
B, S, P = 1, 1, 2   # batch=1, seq_len=1, past_seq=2
dummy = {
    "hidden_states": torch.randn(B, S, 1024),
    "position_ids": torch.full((3, B, S), P, dtype=torch.int64),
    "key_cache": torch.randn(B, 4, P, 256) * 0.01,
    "value_cache": torch.randn(B, 4, P, 256) * 0.01,
}

with torch.no_grad():
    ov_model = ov.convert_model(wrapper, example_input=dummy)
```

导出之后，还调用了一个函数设置动态 shape 和重命名 I/O：

```python
def _make_dynamic_shapes_attn(ov_model, in_names, out_names):
    batch_sym = Symbol()
    seq_sym = Symbol()
    past_sym = Symbol()

    for i, name in enumerate(in_names):
        ps = ov_model.inputs[i].partial_shape
        if name == "in_hidden":
            # 设置 batch 和 seq 为动态
            ps[0] = Dimension(-1); ps[0].set_symbol(batch_sym)
            ps[1] = Dimension(-1); ps[1].set_symbol(seq_sym)
        elif name == "in_position_ids":
            ps[1] = Dimension(-1); ps[1].set_symbol(batch_sym)
            ps[2] = Dimension(-1); ps[2].set_symbol(seq_sym)
        elif name in ("in_key_cache", "in_value_cache"):
            ps[0] = Dimension(-1); ps[0].set_symbol(batch_sym)
            ps[2] = Dimension(-1); ps[2].set_symbol(past_sym)
        ov_model.inputs[i].get_node().set_partial_shape(ps)
        ov_model.inputs[i].get_tensor().set_names({name})

    for i, name in enumerate(out_names):
        ov_model.outputs[i].get_tensor().set_names({name})

    ov_model.validate_nodes_and_infer_types()

# 调用：
in_names = ["in_hidden", "in_position_ids", "in_key_cache", "in_value_cache"]
out_names = ["out_hidden", "out_key_cache", "out_value_cache"]
_make_dynamic_shapes_attn(ov_model, in_names, out_names)
```

## 问题现象

### 问题 1: KV cache 输出没有 Concat — 输出等于输入

导出的 IR 中 **没有 Concat 算子**。KV cache 输出和输入完全一致：

```
输入 key_cache:  shape=(1, 4, 1, 256)   # past_seq=1
输出 key_cache:  shape=(1, 4, 1, 256)   # 期望 (1, 4, 2, 256)，应该是 past+new
```

`torch.cat([past_kv, new_kv], dim=2)` 没有被 `ov.convert_model` 捕获到 IR 中。

### 问题 2: 输出 hidden_states 维度从 3D 变成 2D

IR 输出 `out_hidden` 的 partial shape 是 `[?,1024]` (2D)，期望 `[?,?,1024]` (3D)。

实际运行：
- 输入 hidden `(1, 5, 1024)` → 输出 `(5, 1024)` — batch 维度丢了
- 输入 hidden `(1, 1, 1024)` → 输出 `(1, 1024)` — batch 和 seq 被合并

### 问题 3: I/O tensor 对象共享 + 名字互相覆盖

检查发现，IR 中多个 input/output port 共享同一个底层 OV Tensor 对象：

```python
# 检查代码
for inp in ir.inputs:
    t = inp.get_tensor()
    print(f"{inp.get_any_name()}: tensor id={id(t)}, names={t.names}")
for out in ir.outputs:
    t = out.get_tensor()
    print(f"{out.get_any_name()}: tensor id={id(t)}, names={t.names}")
```

结果：
```
Input 0 (in_hidden):       tensor id=599600, names={'in_hidden'}
Input 1 (in_position_ids): tensor id=090160, names={'in_position_ids'}
Input 2:                    tensor id=599600, names={'out_key_cache'}   ← 与 Input 0 共享!
Input 3:                    tensor id=090160, names={'out_value_cache'} ← 与 Input 1 共享!

Output 0 (out_hidden):      tensor id=599600, names={'out_hidden'}
Output 1 (out_key_cache):   tensor id=090160, names={'out_key_cache'}
Output 2 (out_value_cache): tensor id=599600, names={'out_value_cache'}
```

Input 2 想命名为 `in_key_cache`，但因为和 Output 1 共享 tensor 对象，最后被 Output 1 的 `set_names({"out_key_cache"})` 覆盖了。

## 对比：Phase 1 测试可以正常工作

之前做过几乎一样的测试（`test_single_attn_npu.py`），用同样的 `KVCache` 类和 `AttnLayerWrapper`，导出后 IR 中 **有 Concat 算子**，KV cache 输出正确增长，hidden 输出是 3D。

Phase 1 和当前代码的 **唯一区别**：Phase 1 导出后直接 `ov.save_model()`，**没有**调用 `_make_dynamic_shapes_attn()`（即没有修改 partial shape、没有重命名、没有调用 `validate_nodes_and_infer_types()`）。

## 需要帮忙确认的问题

**Q1**: `ov.convert_model` 之后调用 `set_partial_shape()` + `validate_nodes_and_infer_types()` 是否可能改变图的结构（比如删除 Concat 算子、合并维度）？还是只做 shape 推导？

**Q2**: 多个 input/output port 共享同一个 OV Tensor 对象是正常现象吗？正确的命名方式是什么？`set_names()` 会互相覆盖吗？

**Q3**: 对于 KV cache 这种 "输入一个 tensor，内部做 cat 生成新 tensor，输出 cat 后的结果" 的 pattern，推荐的导出方式是什么？是否应该避免 `ov.convert_model` trace Python list mutation（`cache.key_cache[0] = cat_result`），改用其他方式？

## 文件位置

- 当前导出代码: `qwen35/export_hybrid.py` (第 123-148 行 wrapper，第 302-345 行导出函数)
- Phase 1 测试 (可工作): `qwen35/scripts/test_single_attn_npu.py` (第 58-93 行 wrapper)
- 验证脚本: `qwen35/scripts/_check_attn_shapes.py`, `qwen35/scripts/_check_attn_ops.py`
