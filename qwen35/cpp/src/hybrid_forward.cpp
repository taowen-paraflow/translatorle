// hybrid_forward.cpp — Runtime inference: forward, generate, subgraph runners
//
// Hot-loop optimization: all ov::Tensor objects are pre-allocated in alloc_buffers().
// No heap allocations occur during decode or prefill forward passes.
#include "hybrid_model.h"
#include "tokenizer.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <set>

// ---------------------------------------------------------------------------
// GDN block inference (GPU, stateful)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::run_gdn_block(int block_idx, int seq_len) {
    auto& req = gdn_requests_[block_idx];

    bool is_decode = (seq_len == 1);
    auto& hidden = is_decode ? s1_hidden_ : sc_hidden_;
    req.set_input_tensor(0, hidden);
    req.set_input_tensor(1, is_decode ? s1_gdn_mask_ : sc_gdn_mask_);
    // Zero-copy: GPU copies input to GPU memory before processing, then writes
    // output from GPU memory back to host. Safe to share the same host buffer.
    req.set_output_tensor(0, hidden);

    req.infer();
}

// ---------------------------------------------------------------------------
// Attention block inference (NPU, explicit I/O, ScatterUpdate-3 fixed KV)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::run_attn_block(int block_idx, int seq_len, bool use_prefill) {
    auto& req = use_prefill ? attn_prefill_requests_[block_idx]
                            : attn_requests_[block_idx];
    bool is_decode = (seq_len == 1);

    auto& hidden = is_decode ? s1_hidden_ : sc_hidden_;
    req.set_input_tensor(0, hidden);

    for (int c = 0; c < 3; ++c) {
        for (int s = 0; s < seq_len; ++s) {
            pos_buf_[c * seq_len + s] = past_length_ + s;
        }
    }
    req.set_input_tensor(1, is_decode ? s1_pos_ : sc_pos_);

    req.set_input_tensor(2, kv_key_tensors_[block_idx]);
    req.set_input_tensor(3, kv_value_tensors_[block_idx]);

    for (int s = 0; s < seq_len; ++s) {
        cache_pos_buf_[s] = past_length_ + s;
    }
    req.set_input_tensor(4, is_decode ? s1_cache_pos_ : sc_cache_pos_);

    fill_attn_mask(seq_len);
    req.set_input_tensor(5, is_decode ? s1_attn_mask_ : sc_attn_mask_);

    req.infer();

    // Copy outputs back — NPU output shape for hidden may differ from input
    // shape (NPU squeezes S=1 dim to [B,H]), so memcpy is needed.
    // KV caches: NPU set_output_tensor with aliased input/output causes 2x
    // slowdown (NPU plugin adds internal copies for aliased buffers), so
    // explicit memcpy is faster.
    std::memcpy(hidden_buf_.data(),
                req.get_output_tensor(0).data<const float>(),
                seq_len * cfg_.hidden_size * sizeof(float));
    std::memcpy(kv_key_tensors_[block_idx].data<float>(),
                req.get_output_tensor(1).data<const float>(),
                kv_total_ * sizeof(float));
    std::memcpy(kv_value_tensors_[block_idx].data<float>(),
                req.get_output_tensor(2).data<const float>(),
                kv_total_ * sizeof(float));
}

// ---------------------------------------------------------------------------
// Attention mask (causal, fixed-size KV cache)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::fill_attn_mask(int seq_len) {
    const float MASK_VAL = -65504.0f;
    int P = attn_past_seq_;
    float* mask = attn_mask_buf_.data();

    // For each query q in [0..seq_len-1]:
    //   attend to positions [0..past_length_+q] (inclusive)
    //   mask positions [past_length_+q+1..P-1]
    for (int q = 0; q < seq_len; ++q) {
        float* row = mask + q * P;
        int valid = std::min(past_length_ + q + 1, P);
        // Attend region: 0.0f (IEEE 754 float zero = all bits zero)
        std::memset(row, 0, valid * sizeof(float));
        // Mask region: -65504.0f
        std::fill(row + valid, row + P, MASK_VAL);
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

const float* Qwen35HybridModel::forward(
    const int64_t* token_ids, int seq_len, bool run_head)
{
    int H = cfg_.hidden_size;

    // Embedding lookup into hidden_buf_ (CPU, ~0.02ms)
    for (int i = 0; i < seq_len; ++i) {
        std::memcpy(hidden_buf_.data() + i * H,
                    embed_table_.data() + token_ids[i] * H,
                    H * sizeof(float));
    }

    // Auto-detect prefill mode
    bool use_prefill = (seq_len == prefill_chunk_size_
                        && !attn_prefill_requests_.empty());

    // 6 × (GDN → Attn)
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        run_gdn_block(i, seq_len);
        run_attn_block(i, seq_len, use_prefill);
    }

    past_length_ += seq_len;

    if (!run_head) return nullptr;

    // For multi-token input (last prefill chunk), run head on LAST token only.
    // Saves (chunk_size-1) × matmul(hidden_size, vocab_size) on GPU (~5ms).
    if (seq_len > 1) {
        std::memmove(hidden_buf_.data(),
                     hidden_buf_.data() + (seq_len - 1) * H,
                     H * sizeof(float));
    }
    head_request_.set_input_tensor(0, s1_hidden_);
    head_request_.infer();

    return head_request_.get_output_tensor(0).data<const float>();
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

std::string Qwen35HybridModel::generate(const std::string& prompt, int max_new_tokens) {
    reset();

    if (!tokenizer_) {
        throw std::runtime_error(
            "Tokenizer not loaded. Provide --tokenizers-lib or run convert_tokenizer.py first.");
    }

    auto input_ids = tokenizer_->encode(prompt);
    int prompt_len = static_cast<int>(input_ids.size());
    int chunk_size = prefill_chunk_size_;
    log("Prompt tokens: " + std::to_string(prompt_len) +
        " (chunk=" + std::to_string(chunk_size) + ")");

    // --- Prefill: chunked (S=chunk_size) + remainder token-by-token (S=1) ---
    auto t0 = std::chrono::steady_clock::now();
    int pos = 0;
    const float* logits = nullptr;

    // Full chunks — skip Head for non-last chunks (saves ~6ms per chunk)
    while (pos + chunk_size <= prompt_len) {
        bool is_last = (pos + chunk_size >= prompt_len);
        logits = forward(input_ids.data() + pos, chunk_size, /*run_head=*/is_last);
        pos += chunk_size;
    }
    // Remainder tokens (S=1) — cannot pad because GDN Loop would process padding
    while (pos < prompt_len) {
        bool is_last = (pos + 1 >= prompt_len);
        logits = forward(input_ids.data() + pos, 1, /*run_head=*/is_last);
        pos += 1;
    }

    double prefill_ms = elapsed_ms(t0);
    log("Prefill: " + std::to_string(prompt_len) + " tokens in " +
        std::to_string(prefill_ms) + " ms (" +
        std::to_string(prompt_len / (prefill_ms / 1000.0)) + " tok/s, chunk=" +
        std::to_string(chunk_size) + ")");

    // --- Greedy decode ---
    int V = cfg_.vocab_size;
    int next_id = static_cast<int>(
        std::max_element(logits, logits + V) - logits);

    std::vector<int64_t> generated;
    generated.push_back(next_id);

    std::set<int> stop_ids = {151645, 151643, 248044};

    auto t_decode = std::chrono::steady_clock::now();
    for (int step = 0; step < max_new_tokens - 1; ++step) {
        if (stop_ids.count(next_id)) break;

        int64_t tid = static_cast<int64_t>(next_id);
        logits = forward(&tid, 1, /*run_head=*/true);
        next_id = static_cast<int>(
            std::max_element(logits, logits + V) - logits);
        generated.push_back(next_id);
    }
    double decode_ms = elapsed_ms(t_decode);
    int num_decoded = static_cast<int>(generated.size());

    if (decode_ms > 0) {
        log("Decode: " + std::to_string(num_decoded) + " tokens in " +
            std::to_string(decode_ms) + " ms (" +
            std::to_string(num_decoded / (decode_ms / 1000.0)) + " tok/s)");
    }

    if (!generated.empty() && stop_ids.count(static_cast<int>(generated.back()))) {
        generated.pop_back();
    }

    return tokenizer_->decode(generated);
}
