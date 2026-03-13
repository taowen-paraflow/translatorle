# Qwen3.5-0.8B C++ Inference 优化方向

研究笔记 -- Intel Lunar Lake (NPU + GPU) 上的两个可行优化路径。

---

## Direction 1: MTP Speculative Decoding (高优先级)

### Qwen3.5 MTP 架构详情

- Qwen3.5 全系列内置 MTP (Multi-Token Prediction)，训练时就包含，权重在 safetensors 中
- Config 参数: `"mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false`
- MTP 模块是 **一个完整的 transformer decoder layer**（标准 multi-head attention + FFN），不是简单的线性投影头

### MTP 模块结构 (from vLLM Qwen3_5MultiTokenPredictor)

```
MTP Module:
  embed_tokens          -- 与主模型共享 (tie_word_embeddings=true)
  pre_fc_norm_embedding -- RMSNorm on next-token embedding
  pre_fc_norm_hidden    -- RMSNorm on last-layer hidden state
  fc                    -- Linear(hidden_size*2 -> hidden_size)
  layers[0]             -- One Qwen3_5DecoderLayer (full_attention type)
    input_layernorm
    self_attn (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm)
    post_attention_layernorm
    mlp (gate_proj, up_proj, down_proj)
  norm                  -- Final RMSNorm
  lm_head               -- 与主模型共享 (tie_word_embeddings=true)
```

### 权重名称 (safetensors 中的 15 个参数)

```
mtp.fc.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
mtp.norm.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.self_attn.q_proj.weight
mtp.layers.0.self_attn.k_proj.weight
mtp.layers.0.self_attn.v_proj.weight
mtp.layers.0.self_attn.o_proj.weight
mtp.layers.0.self_attn.q_norm.weight
mtp.layers.0.self_attn.k_norm.weight
mtp.layers.0.mlp.gate_proj.weight
mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.mlp.down_proj.weight
```

### 参数规模 (0.8B, hidden_size=1024, num_heads=8, head_dim=256, kv_heads=2, intermediate=3584)

- fc: 2048 x 1024 = 2.1M
- self_attn: q(2M) + k(0.5M) + v(0.5M) + o(2M) = 5M
- mlp: gate(3.7M) + up(3.7M) + down(3.7M) = 11M
- norms: ~5K
- **Total: ~18M params = ~36MB FP16 = ~9MB INT4**
- 约占总模型的 2%，非常轻量

### MTP 推理算法

1. 主模型生成 token T[0]，同时捕获最后一层的 hidden state H[0]
2. MTP head: `concat(RMSNorm(H[0]), RMSNorm(embed(T[0])))` -> Linear -> 一个 decoder layer -> 共享 lm_head -> 预测 T[1]
3. MTP head 可以迭代: 用 MTP 层输出 H[1]' 和 embed(T[1]) 继续预测 T[2]
4. 主模型通过 **prefill 路径**批量验证 T[1], T[2], ... (不是逐 token decode!)
5. 接受匹配的 tokens，从第一个不匹配处拒绝

### HuggingFace 注意事项

- 标准 `Qwen3_5ForCausalLM` 类会忽略 MTP 权重: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`
- vLLM/SGLang 有专门的 `Qwen3_5MTP` 模型类来加载这些权重
- 我们需要在导出脚本中自行加载 MTP 权重

### vLLM/SGLang 部署参考

```bash
# vLLM
vllm serve Qwen/Qwen3.5-0.8B --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3.5-0.8B \
    --speculative-algo NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

### OpenVINO 实现方案

1. **导出 MTP head** 为额外的 OpenVINO IR 模型 (mtp_block.xml)
   - 输入: hidden_state [1, 1, 1024] + token_embedding [1, 1, 1024]
   - 包含: 2x RMSNorm + Linear(2048->1024) + 一个标准 attention layer + RMSNorm
   - MTP 的 attention layer 需要自己的 KV cache
   - 输出: hidden_state [1, 1, 1024]
   - 复用已有的 head_block 做 lm_head

2. **验证用 prefill 路径**: 已有的 chunkwise parallel GDN prefill + chunked attention prefill 可以批量处理 2-3 个 draft tokens
   - 这是关键优势: 验证不需要逐 token decode，用 prefill 路径一次处理多个 token

3. **状态管理**:
   - Chunkwise prefill 使用 explicit I/O (非 stateful)，输出的 conv/rec states 可以在任意位置注入
   - 已有 `transfer_prefill_states_to_decode()` 基础设施
   - 验证失败时: 只需用正确位置的 states 重新注入即可

4. **预期收益**:
   - 每 decode 步预测 2 个额外 token，假设 60-80% 接受率
   - 有效速度提升 30-60%（从 ~21 tok/s -> ~28-34 tok/s peak）
   - MTP head 本身开销很小（~6-8ms per draft token）

---

## Direction 2: Loop-free GDN Decode (S=1)

### 问题分析

- 当前 GDN decode blocks 使用 Loop 节点，即使 S=1 decode 时 Loop 只执行一次
- OpenVINO GPU 插件中，每个 Loop 节点有 **3+ 次 host-GPU 同步往返**:
  1. `read_scalar_value(trip_count_mem)` -- 从 GPU 读 trip_count 到 host
  2. `read_scalar_value(first_execution_condition_mem)` -- 读初始执行条件
  3. `read_scalar_value(body_execution_condition_mem)` -- body 执行后检查条件
- 每个 GDN block 有 3 个 Loop (3 layers), 共 6 blocks = **18 个 Loop 执行/token = 54+ 次 host-GPU 同步**

### GPU_ENABLE_LOOP_UNROLLING 不起作用

- `ov::intel_gpu::enable_loop_unrolling` 只能展开 **静态 trip count < 16** 的 Loop
- GDN 的 Loop 有动态 trip_count (来自输入 port)，`get_num_iterations()` 返回 -1
- `UnrollTensorIterator` pass 遇到 -1 直接跳过，无论该标志是 true 还是 false

### 解决方案: 导出 Loop-free S=1 decode IR

- 对于 S=1 decode，GDN 递归简化为标准 rank-1 update
- 可以直接将 Loop body 内联为普通的 MatMul/Add/Mul 操作
- 消除所有 Loop 管理开销: 同步往返、内存预处理、event wait
- chunkwise_gdn.py 中的 WY representation 对 S=1 是数学精确的（不需要 Neumann 近似）

### 实验结果 — S1 blocks 反而更慢

A/B 测试（back-to-back, 相同热状态，INT4+INT8 量化模型）：

| 指标 | Loop-based | S1 no-Loop | 差异 |
|------|-----------|------------|------|
| GDN total/token | 46.2ms | 60.1ms | **+30% 更慢** |
| GDN/block avg | 7.7ms | 10.0ms | +30% |
| Decode tok/s | 16.6 | 13.1 | **-21%** |
| GDN compilation | 3380ms | 1094ms | S1 编译更快 |

**原因分析**：
- GPU 插件对 Loop body 内部做了 **kernel fusion**，将 RecurrentAttentionCell 的所有操作融合为高效的 GPU kernel
- S1 inlined flat ops（595 ops）没有 Loop 包装，GPU 插件无法识别它们属于同一个计算单元，fusion 机会减少
- Loop overhead（54 次 host-GPU sync ≈ 2-3ms）远小于 fusion 带来的收益（~14ms）
- S1 模型的 ops 略多（595 vs 579），额外 ops 来自 Unsqueeze/Squeeze/Reshape 操作

**结论**：Loop-free S=1 decode 不可行。GPU 插件的 Loop body fusion 优化 > Loop 管理开销。保留 Loop-based blocks 作为 decode 路径。

**代码处理**：S1 export 代码保留在 export_hybrid.py 中但不默认使用。C++ 侧 S1 加载逻辑保留 but 目前不会被触发（除非手动导出 S1 blocks）。

---

## GPU Plugin 可用的优化配置

| 属性 | 默认值 | 作用 |
|------|--------|------|
| `ov::intel_gpu::enable_loop_unrolling` | true | 展开静态 <16 iter 的 Loop（对 GDN 无效） |
| `ov::intel_gpu::hint::enable_sdpa_optimization` | true | SDPA 融合优化 |
| `ov::intel_gpu::hint::queue_throttle` | MEDIUM | OpenCL 队列节流 |
| `ov::intel_gpu::hint::queue_priority` | MEDIUM | OpenCL 队列优先级 |
| `ov::hint::inference_precision` | f16 | 模型精度 |
| `ov::hint::performance_mode` | LATENCY | 延迟 vs 吞吐 |
| `ov::hint::execution_mode` | PERFORMANCE | 允许不安全精度优化 |
| `ov::cache_dir` | "" | 模型缓存目录 |

---

## 优先级和风险评估

| 方向 | 预期收益 | 复杂度 | 风险 | 优先级 |
|------|---------|--------|------|--------|
| Loop-free S=1 decode | +30% 更慢（实测） | 中（修改导出） | 低 | ❌ 失败 |
| MTP speculative decoding | +30-60% decode | 高（新 pipeline） | 中（需实测验证） | ★★★★ 后做 |

---

## 其他考虑过但不推荐的方案

### N-gram Prompt-lookup

- 零额外模型开销，纯 CPU 字符串匹配
- 适合翻译任务（输出常复制输入中的实体/数字）
- 但验证仍需 prefill 路径，且命中率不稳定
- 可作为 MTP 的补充，但不作为主要方向

### GPU Stateful Attention for Decode

- 已尝试: 加载 6 个 GPU attention 模型导致 iGPU 资源争抢
- GDN 速度降 2.2x，整体从 21->10.5 tok/s
- 结论: NPU attention 是当前最优方案
