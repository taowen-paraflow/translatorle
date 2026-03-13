// hybrid_forward.cpp — Runtime inference: forward, generate, subgraph runners
//
// Hot-loop optimization: all ov::Tensor objects are pre-allocated in alloc_buffers().
// No heap allocations occur during decode or prefill forward passes.
// Decode attention mask is incremental (1 float write/step vs 1KB memset).
// Timing instrumentation is off by default (--timing flag to enable).
#include "hybrid_model.h"
#include "tokenizer.h"
#include "utils.h"

#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// Stop token check (inline, no std::set overhead)
// ---------------------------------------------------------------------------

static inline bool is_stop_token(int id) {
    return id == 151645 || id == 151643 || id == 248044;
}

// ---------------------------------------------------------------------------
// GDN block inference (GPU, stateful)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::run_gdn_block(int block_idx, int seq_len) {
    auto& req = gdn_requests_[block_idx];
    bool is_decode = (seq_len == 1);
    auto& hidden = is_decode ? s1_hidden_ : sc_hidden_.at(seq_len);
    req.set_input_tensor(0, hidden);
    req.set_input_tensor(1, is_decode ? s1_gdn_mask_ : sc_gdn_mask_.at(seq_len));
    // Zero-copy: GPU copies input to GPU memory before processing, then writes
    // output from GPU memory back to host. Safe to share the same host buffer.
    req.set_output_tensor(0, hidden);

    req.infer();
}

// ---------------------------------------------------------------------------
// GDN prefill block inference (GPU, explicit I/O, chunkwise parallel)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::run_gdn_prefill_block(
    int block_idx, ov::Tensor& hidden_tensor, ov::Tensor& mask_tensor)
{
    auto& req = gdn_prefill_requests_[block_idx];

    // Input 0: hidden [1, S, H], Input 1: mask [1, S]
    req.set_input_tensor(0, hidden_tensor);
    req.set_input_tensor(1, mask_tensor);

    // Inputs 2-7: conv0, rec0, conv1, rec1, conv2, rec2
    for (int j = 0; j < 3; ++j) {
        req.set_input_tensor(2 + j * 2, gdn_prefill_conv_states_[block_idx][j]);
        req.set_input_tensor(3 + j * 2, gdn_prefill_rec_states_[block_idx][j]);
    }

    // Output 0: hidden (set to same tensor for zero-copy on GPU)
    req.set_output_tensor(0, hidden_tensor);

    req.infer();

    // Copy output states back (outputs 1-6: conv0, rec0, conv1, rec1, conv2, rec2)
    for (int j = 0; j < 3; ++j) {
        auto out_conv = req.get_output_tensor(1 + j * 2);
        auto out_rec = req.get_output_tensor(2 + j * 2);
        std::memcpy(gdn_prefill_conv_states_[block_idx][j].data<float>(),
                    out_conv.data<const float>(),
                    out_conv.get_byte_size());
        std::memcpy(gdn_prefill_rec_states_[block_idx][j].data<float>(),
                    out_rec.data<const float>(),
                    out_rec.get_byte_size());
    }
}

// ---------------------------------------------------------------------------
// Attention block inference (NPU, explicit I/O, ScatterUpdate-3 fixed KV)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::run_attn_block(int block_idx, int seq_len, bool use_prefill) {
    auto& req = use_prefill ? attn_prefill_requests_.at(seq_len)[block_idx]
                            : attn_requests_[block_idx];
    bool is_decode = (seq_len == 1);

    auto& hidden = is_decode ? s1_hidden_ : sc_hidden_.at(seq_len);
    req.set_input_tensor(0, hidden);

    for (int c = 0; c < 3; ++c) {
        for (int s = 0; s < seq_len; ++s) {
            pos_buf_[c * seq_len + s] = past_length_ + s;
        }
    }
    req.set_input_tensor(1, is_decode ? s1_pos_ : sc_pos_.at(seq_len));

    req.set_input_tensor(2, kv_key_tensors_[block_idx]);
    req.set_input_tensor(3, kv_value_tensors_[block_idx]);

    for (int s = 0; s < seq_len; ++s) {
        cache_pos_buf_[s] = past_length_ + s;
    }
    req.set_input_tensor(4, is_decode ? s1_cache_pos_ : sc_cache_pos_.at(seq_len));

    fill_attn_mask(seq_len);
    req.set_input_tensor(5, is_decode ? s1_attn_mask_ : sc_attn_mask_.at(seq_len));

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
// Used by prefill and multi-token paths. Decode uses incremental mask instead.
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
    bool is_decode = (seq_len == 1);

    // Embedding lookup into hidden_buf_ (CPU, ~0.02ms)
    for (int i = 0; i < seq_len; ++i) {
        std::memcpy(hidden_buf_.data() + i * H,
                    embed_table_.data() + token_ids[i] * H,
                    H * sizeof(float));
    }

    if (is_decode) {
        // --- Decode fast path: tensors pre-bound in bind_decode_tensors() ---
        // Only need to update position_ids, cache_position, and attn_mask.
        // Hidden content updated by embedding lookup above (same buffer).
        for (int c = 0; c < 3; ++c) pos_buf_[c] = past_length_;
        cache_pos_buf_[0] = past_length_;
        // Incremental mask: init_decode_attn_mask() set the base,
        // now just unmask the current position (1 float write vs 1KB memset)
        attn_mask_buf_[past_length_] = 0.0f;

        if (timing_) {
            // Timed decode path: 12 chrono::now() calls per token
            for (int i = 0; i < cfg_.num_blocks; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                gdn_requests_[i].infer();
                decode_gdn_ms_[i] += elapsed_ms(t0);

                t0 = std::chrono::steady_clock::now();
                attn_requests_[i].infer();
                std::memcpy(hidden_buf_.data(),
                            attn_requests_[i].get_output_tensor(0).data<const float>(),
                            H * sizeof(float));
                std::memcpy(kv_key_tensors_[i].data<float>(),
                            attn_requests_[i].get_output_tensor(1).data<const float>(),
                            kv_total_ * sizeof(float));
                std::memcpy(kv_value_tensors_[i].data<float>(),
                            attn_requests_[i].get_output_tensor(2).data<const float>(),
                            kv_total_ * sizeof(float));
                decode_attn_ms_[i] += elapsed_ms(t0);
            }

            past_length_ += 1;
            if (!run_head) return nullptr;

            auto t0 = std::chrono::steady_clock::now();
            head_request_.infer();
            decode_head_ms_ += elapsed_ms(t0);
            decode_steps_++;
        } else {
            // Fast decode path: no timing overhead
            for (int i = 0; i < cfg_.num_blocks; ++i) {
                gdn_requests_[i].infer();
                attn_requests_[i].infer();
                std::memcpy(hidden_buf_.data(),
                            attn_requests_[i].get_output_tensor(0).data<const float>(),
                            H * sizeof(float));
                std::memcpy(kv_key_tensors_[i].data<float>(),
                            attn_requests_[i].get_output_tensor(1).data<const float>(),
                            kv_total_ * sizeof(float));
                std::memcpy(kv_value_tensors_[i].data<float>(),
                            attn_requests_[i].get_output_tensor(2).data<const float>(),
                            kv_total_ * sizeof(float));
            }

            past_length_ += 1;
            if (!run_head) return nullptr;

            head_request_.infer();
            decode_steps_++;
        }

        return head_request_.get_output_tensor(0).data<const float>();
    }

    // --- Multi-token path (used by chunk-based prefill fallback) ---
    bool use_prefill = (attn_prefill_requests_.count(seq_len) > 0);

    for (int i = 0; i < cfg_.num_blocks; ++i) {
        run_gdn_block(i, seq_len);
        run_attn_block(i, seq_len, use_prefill);
    }

    past_length_ += seq_len;

    if (!run_head) return nullptr;

    if (seq_len > 1) {
        std::memmove(hidden_buf_.data(),
                     hidden_buf_.data() + (seq_len - 1) * H,
                     H * sizeof(float));
    }
    head_request_.infer();

    return head_request_.get_output_tensor(0).data<const float>();
}

// ---------------------------------------------------------------------------
// Layer-major prefill: full-batch GDN (GPU) + chunked attention (NPU)
// ---------------------------------------------------------------------------

const float* Qwen35HybridModel::prefill(const int64_t* token_ids, int prompt_len) {
    int H = cfg_.hidden_size;
    int P = attn_past_seq_;
    int C = prefill_chunk_size_;
    auto prefill_t0 = std::chrono::steady_clock::now();

    // Compute chunk boundaries for NPU attention
    struct Chunk { int start; int len; };
    std::vector<Chunk> chunks;
    int pos = 0;
    while (pos < prompt_len) {
        int remaining = prompt_len - pos;
        int cs = C;
        while (cs > remaining) cs /= 2;
        if (cs < 1) cs = 1;
        chunks.push_back({pos, cs});
        pos += cs;
    }

    // Use pre-allocated prefill buffer (no heap alloc)
    float* full_hidden = prefill_hidden_buf_.data();
    int64_t* full_gdn_mask = prefill_gdn_mask_.data();

    // Embedding lookup
    for (int i = 0; i < prompt_len; ++i) {
        std::memcpy(full_hidden + i * H,
                    embed_table_.data() + token_ids[i] * H,
                    H * sizeof(float));
    }

    // Create tensor wrappers for full-prompt GDN (wraps pre-allocated buffer)
    ov::Tensor full_hidden_tensor(ov::element::f32,
        {1, static_cast<size_t>(prompt_len), static_cast<size_t>(H)},
        full_hidden);
    ov::Tensor full_mask_tensor(ov::element::i64,
        {1, static_cast<size_t>(prompt_len)},
        full_gdn_mask);

    // --- Timing arrays (always needed for prefill report) ---
    std::vector<double> gdn_times(cfg_.num_blocks, 0.0);
    std::vector<double> attn_times(cfg_.num_blocks, 0.0);
    std::vector<std::vector<double>> attn_chunk_times(
        cfg_.num_blocks, std::vector<double>(chunks.size(), 0.0));

    for (int layer = 0; layer < cfg_.num_blocks; ++layer) {
        // --- GDN block (GPU): full prompt at once ---
        auto gdn_t0 = std::chrono::steady_clock::now();
        if (has_gdn_prefill_) {
            run_gdn_prefill_block(layer, full_hidden_tensor, full_mask_tensor);
        } else {
            auto& gdn_req = gdn_requests_[layer];
            gdn_req.set_input_tensor(0, full_hidden_tensor);
            gdn_req.set_input_tensor(1, full_mask_tensor);
            gdn_req.set_output_tensor(0, full_hidden_tensor);
            gdn_req.infer();
        }
        gdn_times[layer] = elapsed_ms(gdn_t0);

        // --- Attention block (NPU): chunked ---
        auto attn_layer_t0 = std::chrono::steady_clock::now();
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            auto& chunk = chunks[ci];
            int cs = chunk.len;
            past_length_ = chunk.start;

            std::memcpy(hidden_buf_.data(),
                        full_hidden + chunk.start * H,
                        cs * H * sizeof(float));

            auto attn_chunk_t0 = std::chrono::steady_clock::now();
            bool use_prefill = attn_prefill_requests_.count(cs) > 0;
            run_attn_block(layer, cs, use_prefill);
            attn_chunk_times[layer][ci] = elapsed_ms(attn_chunk_t0);

            std::memcpy(full_hidden + chunk.start * H,
                        hidden_buf_.data(),
                        cs * H * sizeof(float));
        }
        attn_times[layer] = elapsed_ms(attn_layer_t0);
    }

    // Set past_length for decode phase
    past_length_ = prompt_len;

    // Transfer chunkwise prefill states to stateful decode blocks
    if (has_gdn_prefill_) {
        transfer_prefill_states_to_decode();
    }

    // Restore decode tensor bindings + initialize incremental decode mask
    bind_decode_tensors();

    // Head: last token only
    std::memcpy(hidden_buf_.data(),
                full_hidden + (prompt_len - 1) * H,
                H * sizeof(float));
    head_request_.set_input_tensor(0, s1_hidden_);
    auto head_t0 = std::chrono::steady_clock::now();
    head_request_.infer();
    double head_ms = elapsed_ms(head_t0);

    double total_ms = elapsed_ms(prefill_t0);

    // --- Print timing breakdown ---
    double gdn_total = 0, attn_total = 0;
    std::string gdn_line = "  GDN : ";
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        gdn_total += gdn_times[i];
        gdn_line += "[" + std::to_string(i) + "]=" + fmt_ms(gdn_times[i]) + "ms ";
    }
    gdn_line += " total=" + fmt_ms(gdn_total) + "ms";

    std::string attn_line = "  Attn: ";
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        attn_total += attn_times[i];
        attn_line += "[" + std::to_string(i) + "]=" + fmt_ms(attn_times[i]) + "ms ";
    }
    attn_line += " total=" + fmt_ms(attn_total) + "ms";

    std::string chunk_hdr = "  Chunks: ";
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        chunk_hdr += "S=" + std::to_string(chunks[ci].len);
        if (ci + 1 < chunks.size()) chunk_hdr += ",";
    }

    std::string chunk_detail;
    for (int layer = 0; layer < cfg_.num_blocks; ++layer) {
        chunk_detail += "  Attn[" + std::to_string(layer) + "] chunks: ";
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            chunk_detail += "S" + std::to_string(chunks[ci].len) + "=" +
                            fmt_ms(attn_chunk_times[layer][ci]) + "ms ";
        }
        chunk_detail += "\n";
    }

    log("=== Prefill Timing (" + std::to_string(prompt_len) + " tokens) ===");
    log(gdn_line);
    log(attn_line);
    log(chunk_hdr);
    log(chunk_detail);
    log("  Head: " + fmt_ms(head_ms) + "ms");
    log("  Total prefill: " + fmt_ms(total_ms) + "ms");
    log("  Breakdown: GDN " + fmt_pct(gdn_total / total_ms * 100) + "% + Attn " +
        fmt_pct(attn_total / total_ms * 100) + "% + Head " +
        fmt_pct(head_ms / total_ms * 100) + "%");

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

    // Initialize decode timing accumulators
    decode_gdn_ms_.assign(cfg_.num_blocks, 0.0);
    decode_attn_ms_.assign(cfg_.num_blocks, 0.0);
    decode_head_ms_ = 0;
    decode_steps_ = 0;

    // --- Prefill: layer-major (full-batch GDN + chunked NPU attention) ---
    auto t0 = std::chrono::steady_clock::now();
    const float* logits = prefill(input_ids.data(), prompt_len);

    double prefill_ms = elapsed_ms(t0);
    log("Prefill: " + std::to_string(prompt_len) + " tokens in " +
        fmt_ms(prefill_ms) + " ms (" +
        fmt_ms(prompt_len / (prefill_ms / 1000.0)) + " tok/s, chunk=" +
        std::to_string(chunk_size) + ", layer-major)");

    // --- Greedy decode ---
    int V = cfg_.vocab_size;
    int next_id = static_cast<int>(
        std::max_element(logits, logits + V) - logits);

    std::vector<int64_t> generated;
    generated.push_back(next_id);

    auto t_decode = std::chrono::steady_clock::now();

    for (int step = 0; step < max_new_tokens - 1; ++step) {
        if (is_stop_token(next_id)) break;

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
            fmt_ms(decode_ms) + " ms (" +
            fmt_ms(num_decoded / (decode_ms / 1000.0)) + " tok/s)");

        if (timing_ && decode_steps_ > 0) {
            double gdn_total = 0, attn_total = 0;
            for (int i = 0; i < cfg_.num_blocks; ++i) {
                gdn_total += decode_gdn_ms_[i];
                attn_total += decode_attn_ms_[i];
            }
            double per_tok = decode_ms / decode_steps_;
            log("=== Decode Timing Breakdown (avg per token: " + fmt_ms(per_tok) + "ms) ===");
            for (int i = 0; i < cfg_.num_blocks; ++i) {
                double gdn_avg = decode_gdn_ms_[i] / decode_steps_;
                double attn_avg = decode_attn_ms_[i] / decode_steps_;
                log("  Block " + std::to_string(i) + ": GDN=" + fmt_ms(gdn_avg) +
                    "ms  Attn=" + fmt_ms(attn_avg) + "ms");
            }
            double gdn_avg = gdn_total / decode_steps_;
            double attn_avg = attn_total / decode_steps_;
            double head_avg = decode_head_ms_ / decode_steps_;
            log("  Total: GDN=" + fmt_ms(gdn_avg) + "ms (" +
                fmt_pct(gdn_total / decode_ms * 100) + "%)  Attn=" +
                fmt_ms(attn_avg) + "ms (" + fmt_pct(attn_total / decode_ms * 100) +
                "%)  Head=" + fmt_ms(head_avg) + "ms (" +
                fmt_pct(decode_head_ms_ / decode_ms * 100) + "%)");
        }
    }

    if (!generated.empty() && is_stop_token(static_cast<int>(generated.back()))) {
        generated.pop_back();
    }

    return tokenizer_->decode(generated);
}
