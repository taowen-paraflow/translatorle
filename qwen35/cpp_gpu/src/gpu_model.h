#ifndef GPU_MODEL_H
#define GPU_MODEL_H

#include <string>
#include <vector>
#include <memory>
#include <openvino/openvino.hpp>

class OVTokenizer;

/// GPU-only inference for Qwen3.5 using a single stateful OpenVINO IR.
///
/// Much simpler than the 19-subgraph hybrid approach:
///   - One compiled model on GPU (stateful: conv/recurrent/KV states internal)
///   - Prefill: one infer() call with full prompt
///   - Decode: one infer() call per token
///   - Supports INT8 quantized embeddings (embed_tokens_int8.npy + scales)
class Qwen35GPUModel {
public:
    Qwen35GPUModel(const std::string& model_dir,
                   const std::string& tokenizers_lib = "",
                   bool timing = false);
    ~Qwen35GPUModel();

    std::string generate(const std::string& prompt, int max_new_tokens = 100);

private:
    void load_config(const std::string& model_dir);
    void load_model(const std::string& model_dir);
    void load_embeddings(const std::string& model_dir);
    void reset();

    // Config
    int hidden_size_ = 0;
    int vocab_size_ = 0;
    int past_length_ = 0;
    bool timing_ = false;

    // OpenVINO
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest request_;

    // Input name detection
    bool has_attention_mask_ = false;
    bool has_position_ids_ = false;
    bool has_beam_idx_ = false;

    // Embedding table — one of two modes:
    // Mode 1: FP16 -> FP32 (embed_tokens.npy)
    std::vector<float> embed_fp32_;

    // Mode 2: INT8 + scales (embed_tokens_int8.npy + embed_tokens_scales.npy)
    std::vector<int8_t> embed_int8_;
    std::vector<float> embed_scales_;  // per-row scale (FP32)
    bool use_int8_embed_ = false;

    // Pre-allocated buffers
    std::vector<float> hidden_buf_;     // max_seq * hidden_size
    std::vector<int64_t> mask_buf_;     // max_seq (attention mask, all ones)
    std::vector<int64_t> pos_buf_;      // 3 * max_seq (mRoPE position IDs)

    // Tokenizer
    std::unique_ptr<OVTokenizer> tokenizer_;

    // Embedding lookup: writes seq_len * hidden_size floats into hidden_buf_
    void embed_lookup(const int64_t* token_ids, int seq_len);
};

#endif // GPU_MODEL_H
