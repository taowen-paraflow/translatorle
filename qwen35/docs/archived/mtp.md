# MTP Speculative Decoding 实现计划

Qwen3.5-0.8B 内置 MTP (Multi-Token Prediction) 模块，可用于 speculative decoding 加速推理。

## 目标

当前 C++ hybrid decode: **~21 tok/s peak, 10-14 sustained**
目标: 通过 MTP 每步多出 1-2 个 token，有效吞吐提升 **30-60%**。

---

## 1. MTP 模块结构

### 架构

MTP 是一个轻量级的 "next-next-token predictor"，结构:

```
Input:
  hidden_states [B, 1, 1024]  ← 主模型最后一层输出 (head block 之前的 hidden)
  input_ids [B, 1]            ← 主模型刚预测的 token

Forward:
  emb = embed_tokens(input_ids)              # [B, 1, 1024], 共享 embedding
  emb = RMSNorm_embedding(emb)               # pre_fc_norm_embedding
  hid = RMSNorm_hidden(hidden_states)        # pre_fc_norm_hidden
  x = Linear(cat(emb, hid), dim=-1)          # fc: [B, 1, 2048] → [B, 1, 1024]
  x = DecoderLayer(x, position_ids, kv_cache)  # 一个完整 attention+MLP 层
  x = RMSNorm(x)                             # final norm
  logits = lm_head(x)                        # 共享权重 = embed_tokens.T

Output:
  logits [B, 1, 248320]  ← 下下个 token 的预测
  hidden [B, 1, 1024]    ← 可用于迭代预测第 3 个 token
```

### 权重 (15 tensors, ~20M params, ~36MB FP16)

```
mtp.pre_fc_norm_embedding.weight   [1024]
mtp.pre_fc_norm_hidden.weight      [1024]
mtp.fc.weight                      [1024, 2048]
mtp.layers.0.input_layernorm.weight       [1024]
mtp.layers.0.self_attn.q_proj.weight      [4096, 1024]   # 8 heads × 256 × 2 (query+gate)
mtp.layers.0.self_attn.k_proj.weight      [512, 1024]    # 2 KV heads × 256
mtp.layers.0.self_attn.v_proj.weight      [512, 1024]
mtp.layers.0.self_attn.o_proj.weight      [1024, 2048]   # 8×256 → 1024
mtp.layers.0.self_attn.q_norm.weight      [256]
mtp.layers.0.self_attn.k_norm.weight      [256]
mtp.layers.0.post_attention_layernorm.weight  [1024]
mtp.layers.0.mlp.gate_proj.weight         [3584, 1024]
mtp.layers.0.mlp.up_proj.weight           [3584, 1024]
mtp.layers.0.mlp.down_proj.weight         [1024, 3584]
mtp.norm.weight                           [1024]
```

### 关键特性

- **共享 embed_tokens** (`mtp_use_dedicated_embeddings=false`) — 不需要额外 embedding 表
- **共享 lm_head** (`tie_word_embeddings=true`) — lm_head = embed_tokens.T，复用 head block
- **独立 KV cache** — MTP attention 层有自己的 KV cache，与主模型的 6 个 attention 层独立
- **可迭代** — MTP 输出的 hidden state 可以再次输入 MTP，预测第 3、4 个 token（vLLM 用 `spec_step_idx` 复用同一层）
- **Attention 结构** — 与主模型完全相同: GQA 8Q:2KV, head_dim=256, output gate (sigmoid), partial RoPE (25%), QK-norms

---

## 2. Speculative Decoding 算法

### 每个 decode step 的流程

```
=== 普通 decode (当前) ===
1. forward(token) → hidden, logits → next_token       # ~48ms
2. repeat

=== MTP speculative decode (目标) ===
1. forward(token) → hidden_0, logits_0 → T[0]         # ~48ms (主模型)
2. mtp(hidden_0, T[0]) → hidden_1, logits_1 → T[1]    # ~8ms  (draft token 1)
3. mtp(hidden_1, T[1]) → hidden_2, logits_2 → T[2]    # ~8ms  (draft token 2)
4. verify([T[0], T[1], T[2]]) via prefill path         # ~20ms (批量验证 3 tokens)
5. 接受匹配的 tokens, 从第一个不匹配处截断
6. repeat
```

### 验证逻辑

验证使用**主模型的 prefill 路径**（不是逐 token decode）:
- 把 `[T[0], T[1], T[2]]` 作为长度 3 的序列送入主模型
- 主模型对每个位置产生 logits
- 比较: `argmax(verify_logits[i]) == T[i+1]`
- 第一个不匹配处截断，接受之前的所有 token

### 状态管理 (关键难点)

**GDN 状态回滚**:
- Draft 阶段: MTP 不改变主模型的 GDN 状态（MTP 只用自己的 attention KV cache）
- Verify 阶段: prefill 路径处理 N 个 token，会更新 GDN conv/rec 状态
- 如果部分 token 被拒绝: 需要回滚 GDN 状态到最后接受的位置
- 方案: verify 前用 `query_state()` 保存快照，拒绝时用 `set_state()` 恢复

**Attention KV cache 回滚**:
- NPU attention 用 explicit I/O + ScatterUpdate-3
- `past_length_` 追踪当前位置
- 拒绝时只需 reset `past_length_` 到最后接受位置
- 被拒绝的 KV entries 会被后续 ScatterUpdate 覆盖，无需清除

**MTP 自身的 KV cache**:
- MTP attention 有独立的 KV cache
- 每次 draft 后需要 reset（下一轮 draft 从新的 hidden_state 开始）
- 最简方案: MTP 用 stateful model，每轮 draft 前 reset states

---

## 3. 实现步骤

### Step 1: Python — 导出 MTP block IR

**文件**: `qwen35/export_hybrid.py` (或新建 `qwen35/export_mtp.py`)

**任务**:
1. 从 safetensors 手动加载 15 个 MTP 权重（HF AutoModelForCausalLM 会忽略 `mtp.*`）
2. 构建 PyTorch MTP 模块:
   - 2× RMSNorm + Linear(2048→1024)
   - 1× Qwen3_5DecoderLayer (full attention)
   - 1× RMSNorm (final)
   - **不含** lm_head（复用已有的 head block）
3. Trace 为 OpenVINO IR，输入签名:
   ```
   hidden_states:  [1, 1, 1024]  f32   ← 主模型最后一层 hidden
   input_embeds:   [1, 1, 1024]  f32   ← embed_tokens(predicted_token)
   position_ids:   [3, 1, 1]     i64   ← mRoPE position
   attention_mask: [1, 1, 1, K]  f32   ← causal mask for MTP KV cache
   ```
4. MTP attention 的 KV cache 处理:
   - 方案 A: **Stateful** (GPU, 简单) — MakeStateful，KV 常驻 GPU
   - 方案 B: **Explicit I/O** (灵活) — 如果需要精确控制 reset
   - **推荐 A**: GPU stateful，每轮 draft 前 reset states
5. FP16 压缩，保存为 `mtp_block.xml/bin`
6. 可选: INT4 量化（通过 quantize_hybrid.py）

**输出**: `models/qwen35/Qwen3.5-0.8B-hybrid/mtp_block.xml`

### Step 2: Python — 验证 MTP 正确性

**文件**: `qwen35/inference_hybrid.py` 或新建测试脚本

**任务**:
1. 在 Python 推理中加载 MTP block
2. 对比 MTP 预测 vs 主模型实际输出的 top-1 token
3. 统计接受率（目标: 60-80%）
4. 验证 MTP KV cache reset 正确性

### Step 3: C++ — 加载 MTP block

**文件**: `qwen35/cpp/src/hybrid_model.h`, `hybrid_model.cpp`

**任务**:
1. 新增 `ov::InferRequest mtp_request_`
2. `load_models()` 中加载 `mtp_block.xml` (GPU, FP16)
3. 预分配 MTP I/O tensors:
   - `mtp_hidden_` [1, 1, 1024]
   - `mtp_embeds_` [1, 1, 1024]
   - `mtp_pos_` [3, 1, 1]
   - `mtp_mask_` [1, 1, 1, MAX_MTP_CACHE]
4. 新增 `has_mtp_` flag
5. 新增配置: `--mtp-steps N` (draft 步数, 默认 2)

### Step 4: C++ — Speculative decode 循环

**文件**: `qwen35/cpp/src/hybrid_forward.cpp`

**任务**:
1. `generate()` 循环改造:

```cpp
while (generated < max_tokens) {
    // --- 主模型 decode ---
    forward(&token, 1);  // → hidden_buf_ 有 hidden, logits 有 next token
    int accepted = 1;
    tokens[0] = argmax(logits);

    if (has_mtp_ && mtp_steps_ > 0) {
        // --- Draft ---
        save_gdn_states();     // 快照 GDN conv/rec states
        int draft_len = 0;
        int64_t draft_tokens[MAX_DRAFT];  // MAX_DRAFT = mtp_steps_ + 1
        draft_tokens[0] = tokens[0];      // 主模型预测的 token

        for (int d = 0; d < mtp_steps_; d++) {
            run_mtp(hidden_state, draft_tokens[d]);  // → mtp_hidden, mtp_logits
            draft_tokens[d + 1] = argmax(mtp_logits);
            hidden_state = mtp_hidden;
            draft_len++;
        }

        // --- Verify ---
        // 用 prefill 路径批量验证 draft_tokens[0..draft_len]
        forward(draft_tokens, draft_len + 1);  // S = draft_len + 1
        // 对比 verify_logits[i] 和 draft_tokens[i+1]
        for (int i = 0; i < draft_len; i++) {
            if (argmax(verify_logits[i]) == draft_tokens[i + 1]) {
                accepted++;
            } else {
                break;  // 从此处截断
            }
        }

        if (accepted < draft_len + 1) {
            // --- Rollback ---
            restore_gdn_states();  // 恢复 GDN states
            // 重新 forward 只接受的 tokens
            forward(draft_tokens, accepted);
            past_length_ -= (draft_len + 1 - accepted);  // 修正 KV cache position
        }

        reset_mtp_states();  // 清空 MTP KV cache
    }

    // output accepted tokens
    for (int i = 0; i < accepted; i++) {
        output(draft_tokens[i]);
    }
}
```

2. 新增辅助函数:
   - `save_gdn_states()` — 6 blocks × 6 states = 36 个 tensor 快照
   - `restore_gdn_states()` — 恢复快照
   - `run_mtp()` — 执行 MTP block + head block
   - `reset_mtp_states()` — 清空 MTP KV cache

### Step 5: 调优和测试

1. 测量 MTP draft 开销 (预计 ~6-8ms/step)
2. 测量 verify 开销 (prefill 3 tokens 预计 ~15-20ms)
3. 统计接受率
4. 调整 `mtp_steps` (1 vs 2 vs 3)
5. 考虑 MTP 放 NPU 还是 GPU (GPU 更灵活，NPU 可能释放 GPU 压力)

---

## 4. 性能估算

### 假设

- 主模型 decode: 48ms/token
- MTP draft: 8ms/step
- Verify (prefill N tokens): 15-20ms
- 接受率: 60-70%

### mtp_steps=2 (draft 2 extra tokens)

```
无 MTP:   48ms → 1 token  = 20.8 tok/s
有 MTP:   48ms + 16ms(draft) + 18ms(verify) = 82ms
  接受 3 tokens (100%): 82ms → 3 tokens = 36.6 tok/s (+76%)
  接受 2 tokens (67%):  82ms → 2 tokens = 24.4 tok/s (+17%)
  接受 1 token  (33%):  82ms → 1 token  = 12.2 tok/s (-41%) ← 比不用还慢!
```

### 关键观察

- **接受率 < 50% 时 MTP 反而更慢** — verify 开销不可忽略
- **mtp_steps=1 更安全**: draft 1 token, verify 2 tokens
  - 48ms + 8ms(draft) + 12ms(verify) = 68ms
  - 接受 2 tokens: 68ms → 2 tokens = 29.4 tok/s (+41%)
  - 接受 1 token: 68ms → 1 token = 14.7 tok/s (-29%)
  - 盈亏平衡: 接受率 > ~42%
- **实际接受率取决于任务** — 代码生成/翻译可能 70%+，创意写作可能 40%

### 建议

先从 `mtp_steps=1` 开始（最简单，风险最低），测量实际接受率后再决定是否增加到 2。

---

## 5. 验证优化: 避免重复计算

上面的伪代码有一个问题: verify 阶段重新计算了主模型已经算过的 token。优化:

### 方案 A: 只 verify draft tokens (跳过已确认的 token)

```
主模型: forward(T[-1]) → H[0], T[0]          # 正常 decode
MTP:    draft T[1], T[2]                      # 2 draft tokens
Verify: forward([T[0], T[1], T[2]], S=3)      # verify 全部 3 tokens
        但 T[0] 的 GDN/Attn 状态更新在 decode 时已经做了!
```

问题: verify 会**重复处理 T[0]**，导致 GDN 状态被双重更新。

### 方案 B: Verify 只处理 draft tokens

```
Verify: forward([T[1], T[2]], S=2)  # 只处理 draft 部分
```

但这需要 T[0] 的状态已经在 decode 时正确更新。而 decode 用的是 stateful 路径，verify 也用 stateful — 状态是连续的，不需要重复 T[0]。

**这是正确的方案**: decode 已经处理了 T[0] 并更新了所有状态，verify 只需要从 T[1] 开始。

### 优化后流程

```
1. decode(T[-1])     → hidden[0], T[0]        # 主模型, 48ms
2. draft(hidden[0], T[0]) → T[1]              # MTP step 1, 8ms
3. draft(hidden[1], T[1]) → T[2]              # MTP step 2, 8ms
4. save_gdn_states()                           # 快照
5. verify([T[1], T[2]], S=2)                   # 主模型 prefill, 12ms
6. compare verify_logits[0] vs T[2]            # 检查 T[1] 后的预测是否 == T[2]
7. 接受/拒绝 + rollback if needed
```

**但注意**: verify 的 logits[0] 对应的是 "给定 T[1], 主模型预测什么"，需要和 T[2] 比较。这是标准 speculative decoding 的 shifted comparison。

实际上更精确:
```
verify_logits = main_model.forward([T[1], T[2]])
  verify_logits[0] = P(next | ...T[0], T[1])  → 应该 == T[2] 才接受 T[1]
  verify_logits[1] = P(next | ...T[0], T[1], T[2])  → 如果 T[2] 也被接受，这就是 bonus token
```

所以 verify S=2 tokens, 最多产生 2 + 1 = 3 个新 token (T[0] 来自 decode, T[1] T[2] 来自 verify 接受, bonus token 来自 verify_logits[1])。

---

## 6. 文件变更清单

| 文件 | 变更 |
|------|------|
| `qwen35/export_mtp.py` (新建) | MTP 模块构建 + 权重加载 + OV IR 导出 |
| `qwen35/export_hybrid.py` | 在 `export_hybrid_subgraphs()` 中调用 MTP 导出 |
| `qwen35/scripts/quantize_hybrid.py` | 添加 `mtp_block.xml` 量化支持 |
| `qwen35/cpp/src/hybrid_model.h` | MTP request, tensors, state backup buffers |
| `qwen35/cpp/src/hybrid_model.cpp` | 加载 mtp_block, 分配 buffers |
| `qwen35/cpp/src/hybrid_forward.cpp` | speculative decode 循环, draft/verify/rollback |

---

## 7. 风险和注意事项

1. **MTP 接受率未知** — 0.8B 小模型的 MTP 接受率可能低于大模型（信息更少）。需实测。
2. **Verify 开销** — prefill 2-3 tokens 的实际耗时需测量。如果 > 20ms，mtp_steps=1 可能是唯一划算的选项。
3. **GPU 热节流** — MTP 增加 GPU 工作量，可能加剧 Lunar Lake 热节流。Sustained 性能提升可能小于 peak。
4. **GDN 状态快照开销** — 6 blocks × 6 states ≈ 36 tensors。每个 conv_state [1, D_conv-1, D_k, D_v]，rec_state [1, D_k, D_v]。总共约 **~1.2MB**，memcpy 开销 ~0.1ms，可忽略。
5. **NPU attention 静态 shape** — verify 时 seq_len=2 或 3，需要 NPU 有对应的预编译模型。当前只有 S=1,2,4,8,16。S=3 需要额外编译或 padding 到 S=4。
6. **MTP KV cache 大小** — MTP 只有 1 层 attention，KV cache 很小 (2 KV heads × 256 head_dim × MAX_LEN)。可以用小的 MAX_LEN (32-64)。
