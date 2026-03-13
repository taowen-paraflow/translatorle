# TTS Code Predictor 优化方案

## 现状

当前 TTS pipeline 三阶段耗时分布（NPU FP16 模式）：

| 阶段 | 设备 | 每步耗时 | 占比 |
|------|------|---------|------|
| Talker (28 层) | NPU | ~30ms | ~23% |
| **Code Predictor (5 层)** | **CPU** | **~100ms** | **~77%** |
| Speech Decoder | NPU | <1ms | ~0% |

**整体 RTF: 2.1~2.4x（未达实时）**

CP 是绝对瓶颈。每个 talker token 需要跑 CP 15 步（1 次 prefill + 14 次 decode），当前实现每步 ~7ms，总计 ~100ms。

## 对标：BabelVox 的 CP 性能

BabelVox 同模型（Qwen3-TTS 0.6B，5 层 CP）在相同硬件上：

| 指标 | Translatorle | BabelVox |
|------|-------------|----------|
| CP 15 步总耗时 | ~100ms | **~20ms** |
| KV Cache 方式 | Stateful API (内部管理) | 显式 numpy 传递 |
| 模型格式 | 单个 stateful IR | prefill + decode 两个 IR |
| Buffer 策略 | 动态 (OpenVINO 管理) | 固定大小 (BUF_LEN=20) |

**差距 ~5x，完全来自 KV 管理方式的不同。**

## 根因分析

### 当前实现 (`ov_code_predictor.py`)

```python
def predict(self, talker_hidden, layer0_token, ...):
    self._request.reset_state()          # ← 每步都清空重建 VariableState
    self._request.infer(prefill_inputs)   # ← 2 token prefill
    for group_idx in range(1, 14):
        self._request.infer(decode_inputs)  # ← 14 次 stateful decode
```

问题：
1. **`reset_state()` 开销** — 每个 talker token 都调用，清空/重建 5 层 × 2 (K+V) = 10 个 VariableState 容器
2. **Stateful API 黑盒开销** — ReadValue/Assign 节点的内部状态转换，对 5 层小模型占比极高
3. **InferRequest 状态机** — 每次 infer() 都要经过状态检查、状态同步等内部流程

### BabelVox 的做法 (`pipeline.py`)

```python
# Prefill: 输出显式 KV
cp_hidden, cp_kv_k, cp_kv_v, cp_pos = self._run_cp_prefill(cp_input)

# Decode: KV 作为 numpy 输入输出传递
for cp_step in range(1, 14):
    cp_hidden, cp_kv_k, cp_kv_v = self._run_cp_decode(
        next_embed, cp_pos, cp_kv_k, cp_kv_v)
```

优势：
1. **零状态管理开销** — 无 reset_state()，无 VariableState
2. **固定 buffer** — KV shape 固定为 `(5, 1, 8, 20, 128)`，BUF_LEN=20 覆盖最大 16 步
3. **两个专用模型** — prefill 模型只输出 KV，decode 模型用 scatter 原地更新 KV buffer

## 优化计划

### 第一步：导出显式 KV 的 CP 模型

参考 BabelVox 的 `tools/export_cp_kvcache.py`，导出两个 ONNX/IR 模型：

**CP Prefill 模型：**
- 输入：`inputs_embeds` (1, 2, 1024)
- 输出：`hidden_states` (1, 2, 1024), `present_keys` (5, 1, 8, 2, 128), `present_values` (5, 1, 8, 2, 128)

**CP Decode 模型：**
- 输入：`inputs_embeds` (1, 1, 1024), `cache_position` (1,), `attention_mask` (1, 1, 1, 20), `past_keys` (5, 1, 8, 20, 128), `past_values` (5, 1, 8, 20, 128)
- 输出：`hidden_states` (1, 1, 1024), `present_keys` (5, 1, 8, 20, 128), `present_values` (5, 1, 8, 20, 128)

关键参数：
- `BUF_LEN = 20`（CP 最多 16 步，留 4 步余量）
- KV head 数 = 8（GQA，query 16 heads / KV 8 heads）
- Head dim = 128

### 第二步：改写 CP 推理代码

将 `ov_code_predictor.py` 从 stateful 改为显式 KV：

```python
class OVCodePredictor:
    def __init__(self, prefill_model_path, decode_model_path):
        self._prefill = core.compile_model(prefill_model_path, "CPU")
        self._decode = core.compile_model(decode_model_path, "CPU")
        # 预分配固定 buffer
        self._kv_k = np.zeros((5, 1, 8, 20, 128), dtype=np.float32)
        self._kv_v = np.zeros((5, 1, 8, 20, 128), dtype=np.float32)

    def predict(self, talker_hidden, layer0_token, ...):
        # Prefill (无需 reset_state)
        result = self._prefill({...})
        self._kv_k[:, :, :, :2, :] = result["present_keys"]
        self._kv_v[:, :, :, :2, :] = result["present_values"]

        # Decode
        for step in range(1, 14):
            result = self._decode({
                "past_keys": self._kv_k,
                "past_values": self._kv_v,
                "cache_position": np.array([step + 1]),
                ...
            })
            self._kv_k = result["present_keys"]
            self._kv_v = result["present_values"]
```

### 第三步：验证与调优

1. **正确性验证** — 对比 stateful 版本和显式 KV 版本的输出，确保 logits 一致
2. **性能基准** — 跑 `benchmark_tts.py`，目标 CP 从 ~100ms 降到 ~20ms
3. **整体 RTF** — Talker 30ms + CP 20ms = 50ms/step → RTF ≈ 1.6x
4. **采样优化** — 检查 `np.random.choice` / `np.argpartition` 是否还有额外开销

## 预期收益

| 指标 | 优化前 | 优化后（预期） |
|------|-------|--------------|
| CP 每步耗时 | ~100ms | ~20ms |
| 总步耗时 | ~130ms | ~50ms |
| RTF | 2.1~2.4x | **~1.6x** |

结合后续 multi-bucket Talker 优化（BabelVox 的另一个关键技术），最终目标 **RTF ≈ 1.0x**。
