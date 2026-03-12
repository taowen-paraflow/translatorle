// hybrid_forward.cpp — Runtime inference: forward, generate, subgraph runners
#include "hybrid_model.h"
#include "tokenizer.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <set>

// ---------------------------------------------------------------------------
// GDN block inference (GPU, stateful)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::run_gdn_block(int block_idx, std::vector<float>& hidden, int seq_len) {
    auto& req = gdn_requests_[block_idx];

    // Input 0: hidden_states [1, seq_len, hidden_size]
    ov::Tensor h_tensor(ov::element::f32,
                        {1, static_cast<size_t>(seq_len),
                         static_cast<size_t>(cfg_.hidden_size)});
    std::memcpy(h_tensor.data<float>(), hidden.data(), hidden.size() * sizeof(float));
    req.set_input_tensor(0, h_tensor);

    // Input 1: attention_mask [1, seq_len] int64, all ones
    ov::Tensor mask_tensor(ov::element::i64,
                           {1, static_cast<size_t>(seq_len)});
    int64_t* mask_data = mask_tensor.data<int64_t>();
    for (int i = 0; i < seq_len; ++i) mask_data[i] = 1;
    req.set_input_tensor(1, mask_tensor);

    req.infer();

    const float* out = req.get_output_tensor(0).data<const float>();
    std::memcpy(hidden.data(), out, hidden.size() * sizeof(float));
}

// ---------------------------------------------------------------------------
// Attention block inference (NPU, explicit I/O)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::run_attn_block(int block_idx, std::vector<float>& hidden, int seq_len) {
    auto& req = attn_requests_[block_idx];
    auto& kv = kv_caches_[block_idx];
    int actual_cache_len = kv.len;
    int H = cfg_.num_kv_heads;
    int D = cfg_.head_dim;
    int P = attn_past_seq_;

    // Input 0: hidden_states [1, 1, hidden_size]
    ov::Tensor h_tensor(ov::element::f32,
                        {1, 1, static_cast<size_t>(cfg_.hidden_size)});
    std::memcpy(h_tensor.data<float>(), hidden.data(), hidden.size() * sizeof(float));
    req.set_input_tensor(0, h_tensor);

    // Input 1: position_ids [3, 1, 1] — mRoPE, all 3 components = past_length
    ov::Tensor pos_tensor(ov::element::i64, {3, 1, 1});
    int64_t* pos_data = pos_tensor.data<int64_t>();
    for (int c = 0; c < 3; ++c) pos_data[c] = past_length_;
    req.set_input_tensor(1, pos_tensor);

    // Input 2: key_cache padded [1, H, P, D]
    auto padded_key = pad_kv_cache(kv.key, H, actual_cache_len, P, D, true);
    ov::Tensor key_tensor(ov::element::f32,
                          {1, static_cast<size_t>(H),
                           static_cast<size_t>(P),
                           static_cast<size_t>(D)});
    std::memcpy(key_tensor.data<float>(), padded_key.data(), padded_key.size() * sizeof(float));
    req.set_input_tensor(2, key_tensor);

    // Input 3: value_cache padded [1, H, P, D]
    auto padded_value = pad_kv_cache(kv.value, H, actual_cache_len, P, D, false);
    ov::Tensor val_tensor(ov::element::f32,
                          {1, static_cast<size_t>(H),
                           static_cast<size_t>(P),
                           static_cast<size_t>(D)});
    std::memcpy(val_tensor.data<float>(), padded_value.data(), padded_value.size() * sizeof(float));
    req.set_input_tensor(3, val_tensor);

    // Input 4: attention_mask [1, 1, 1, P+seq_len]
    int key_seq = P + seq_len;
    auto mask = build_attn_mask(actual_cache_len, P, seq_len);
    ov::Tensor mask_tensor(ov::element::f32,
                           {1, 1, 1, static_cast<size_t>(key_seq)});
    std::memcpy(mask_tensor.data<float>(), mask.data(), mask.size() * sizeof(float));
    req.set_input_tensor(4, mask_tensor);

    req.infer();

    // Output 0: hidden [1, 1, hidden_size]
    const float* out_h = req.get_output_tensor(0).data<const float>();
    std::memcpy(hidden.data(), out_h, hidden.size() * sizeof(float));

    // Output 1,2: new KV [1, H, P+seq_len, D] — compact to remove padding
    const float* out_key = req.get_output_tensor(1).data<const float>();
    const float* out_val = req.get_output_tensor(2).data<const float>();

    int new_len = actual_cache_len + seq_len;
    int out_stride = (P + seq_len) * D;

    kv.key.resize(H * new_len * D);
    kv.value.resize(H * new_len * D);

    for (int h = 0; h < H; ++h) {
        // Real past entries [0..actual_cache_len)
        std::memcpy(kv.key.data() + h * new_len * D,
                    out_key + h * out_stride,
                    actual_cache_len * D * sizeof(float));
        // New token(s) from position P
        std::memcpy(kv.key.data() + h * new_len * D + actual_cache_len * D,
                    out_key + h * out_stride + P * D,
                    seq_len * D * sizeof(float));

        std::memcpy(kv.value.data() + h * new_len * D,
                    out_val + h * out_stride,
                    actual_cache_len * D * sizeof(float));
        std::memcpy(kv.value.data() + h * new_len * D + actual_cache_len * D,
                    out_val + h * out_stride + P * D,
                    seq_len * D * sizeof(float));
    }
    kv.len = new_len;
}

// ---------------------------------------------------------------------------
// Attention mask (NPU explicit mode)
// ---------------------------------------------------------------------------

std::vector<float> Qwen35HybridModel::build_attn_mask(
    int actual_cache_len, int padded_cache_len, int seq_len)
{
    const float MASK_VAL = -65504.0f;
    int key_seq = padded_cache_len + seq_len;
    std::vector<float> mask(key_seq, MASK_VAL);

    // Unmask real past tokens (positions 1..actual_cache_len-1)
    for (int i = 1; i < actual_cache_len; ++i) mask[i] = 0.0f;
    // Unmask new token positions
    for (int i = padded_cache_len; i < padded_cache_len + seq_len; ++i) mask[i] = 0.0f;

    return mask;
}

// ---------------------------------------------------------------------------
// KV cache padding (NPU explicit mode)
// ---------------------------------------------------------------------------

std::vector<float> Qwen35HybridModel::pad_kv_cache(
    const std::vector<float>& cache, int num_heads,
    int current_len, int target_len, int head_dim, bool is_key)
{
    if (current_len >= target_len) {
        std::vector<float> out(num_heads * target_len * head_dim);
        for (int h = 0; h < num_heads; ++h) {
            std::memcpy(out.data() + h * target_len * head_dim,
                        cache.data() + h * current_len * head_dim,
                        target_len * head_dim * sizeof(float));
        }
        return out;
    }

    int pad_len = target_len - current_len;
    float pad_val = is_key ? -1e4f : 0.0f;

    std::vector<float> out(num_heads * target_len * head_dim);
    for (int h = 0; h < num_heads; ++h) {
        std::memcpy(out.data() + h * target_len * head_dim,
                    cache.data() + h * current_len * head_dim,
                    current_len * head_dim * sizeof(float));
        float* pad_start = out.data() + h * target_len * head_dim + current_len * head_dim;
        std::fill(pad_start, pad_start + pad_len * head_dim, pad_val);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

std::vector<float> Qwen35HybridModel::forward(const std::vector<int64_t>& token_ids) {
    int seq_len = static_cast<int>(token_ids.size());
    int H = cfg_.hidden_size;

    // Embedding lookup
    std::vector<float> hidden(seq_len * H);
    for (int i = 0; i < seq_len; ++i) {
        std::memcpy(hidden.data() + i * H,
                    embed_table_.data() + token_ids[i] * H,
                    H * sizeof(float));
    }

    // 6 × (GDN → Attn)
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        run_gdn_block(i, hidden, seq_len);
        run_attn_block(i, hidden, seq_len);
    }

    // Head: hidden → logits
    ov::Tensor h_tensor(ov::element::f32,
                        {1, static_cast<size_t>(seq_len),
                         static_cast<size_t>(H)});
    std::memcpy(h_tensor.data<float>(), hidden.data(), hidden.size() * sizeof(float));
    head_request_.set_input_tensor(0, h_tensor);
    head_request_.infer();

    auto out_tensor = head_request_.get_output_tensor(0);
    const float* logits_ptr = out_tensor.data<const float>();
    int total = seq_len * cfg_.vocab_size;
    std::vector<float> logits(logits_ptr, logits_ptr + total);

    past_length_ += seq_len;
    return logits;
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
    log("Prompt tokens: " + std::to_string(prompt_len));

    // Prefill — token by token (NPU attention requires S=1)
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits;
    for (int i = 0; i < prompt_len; ++i) {
        logits = forward({input_ids[i]});
    }
    double prefill_ms = elapsed_ms(t0);
    log("Prefill: " + std::to_string(prompt_len) + " tokens in " +
        std::to_string(prefill_ms) + " ms (" +
        std::to_string(prompt_len / (prefill_ms / 1000.0)) + " tok/s)");

    // Greedy decode
    int V = cfg_.vocab_size;
    const float* last_logits = logits.data() + (logits.size() - V);
    int next_id = static_cast<int>(
        std::max_element(last_logits, last_logits + V) - last_logits);

    std::vector<int64_t> generated;
    generated.push_back(next_id);

    std::set<int> stop_ids = {151645, 151643, 248044};

    auto t_decode = std::chrono::steady_clock::now();
    for (int step = 0; step < max_new_tokens - 1; ++step) {
        if (stop_ids.count(next_id)) break;

        logits = forward({static_cast<int64_t>(next_id)});
        last_logits = logits.data() + (logits.size() - V);
        next_id = static_cast<int>(
            std::max_element(last_logits, last_logits + V) - last_logits);
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
