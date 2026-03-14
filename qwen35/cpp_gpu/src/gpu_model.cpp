// gpu_model.cpp — Single-IR GPU-only inference for Qwen3.5
#include "gpu_model.h"
#include "tokenizer.h"
#include "npy_reader.h"
#include "utils.h"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

static inline bool is_stop_token(int id) {
    return id == 151645 || id == 151643 || id == 248044;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

Qwen35GPUModel::Qwen35GPUModel(
    const std::string& model_dir,
    const std::string& tokenizers_lib,
    bool timing)
    : timing_(timing)
{
    load_config(model_dir);

    std::string cache_dir = model_dir + "/cache";
    core_.set_property(ov::cache_dir(cache_dir));
    log("Model cache: " + cache_dir);

    load_model(model_dir);
    load_embeddings(model_dir);

    // Pre-allocate buffers for max sequence length
    // Use 512 as max for pre-allocation (covers most prompts + generation)
    const int MAX_SEQ = 512;
    hidden_buf_.resize(MAX_SEQ * hidden_size_, 0.0f);
    mask_buf_.assign(MAX_SEQ, 1LL);
    pos_buf_.resize(3 * MAX_SEQ, 0LL);

    // Initialize tokenizer
    std::string tok_xml = model_dir + "/openvino_tokenizer.xml";
    if (fs::exists(tok_xml) && !tokenizers_lib.empty()) {
        tokenizer_ = std::make_unique<OVTokenizer>(core_, model_dir, tokenizers_lib);
        log("Tokenizer loaded from OV IR");
    } else if (fs::exists(tok_xml)) {
        log("WARNING: Tokenizer IR found but no --tokenizers-lib specified.");
    } else {
        log("WARNING: No openvino_tokenizer.xml found.");
    }
}

Qwen35GPUModel::~Qwen35GPUModel() = default;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

void Qwen35GPUModel::load_config(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + model_dir + "/config.json");

    json root = json::parse(f);
    json cfg = root.contains("text_config") ? root["text_config"] : root;

    hidden_size_ = cfg["hidden_size"].get<int>();
    vocab_size_ = cfg.value("vocab_size", 248320);

    log("Config: hidden=" + std::to_string(hidden_size_) +
        " vocab=" + std::to_string(vocab_size_));
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

void Qwen35GPUModel::load_model(const std::string& model_dir) {
    std::string xml_path = model_dir + "/openvino_model.xml";
    if (!fs::exists(xml_path))
        throw std::runtime_error("Model not found: " + xml_path);

    log("Loading and compiling model on GPU...");
    auto t0 = std::chrono::steady_clock::now();

    auto model = core_.read_model(xml_path);

    // GPU outputs FP16 by default — add FP32 conversion
    {
        ov::preprocess::PrePostProcessor ppp(model);
        for (size_t i = 0; i < model->outputs().size(); ++i) {
            ppp.output(i).tensor().set_element_type(ov::element::f32);
        }
        model = ppp.build();
    }

    // Detect available inputs
    for (auto& inp : model->inputs()) {
        std::string name = inp.get_any_name();
        if (name == "attention_mask") has_attention_mask_ = true;
        else if (name == "position_ids") has_position_ids_ = true;
        else if (name == "beam_idx") has_beam_idx_ = true;
    }

    ov::AnyMap gpu_config = {
        {ov::hint::num_requests.name(), uint32_t(1)},
    };
    compiled_ = core_.compile_model(model, "GPU", gpu_config);
    request_ = compiled_.create_infer_request();

    log("Model compiled in " + std::to_string(elapsed_ms(t0)) + " ms");
    log("Inputs: inputs_embeds" +
        std::string(has_attention_mask_ ? " attention_mask" : "") +
        std::string(has_position_ids_ ? " position_ids" : "") +
        std::string(has_beam_idx_ ? " beam_idx" : ""));
}

// ---------------------------------------------------------------------------
// Embedding loading (supports both FP16 and INT8)
// ---------------------------------------------------------------------------

void Qwen35GPUModel::load_embeddings(const std::string& model_dir) {
    log("Loading embeddings...");
    auto t0 = std::chrono::steady_clock::now();

    std::string int8_path = model_dir + "/embed_tokens_int8.npy";
    std::string scales_path = model_dir + "/embed_tokens_scales.npy";
    std::string fp16_path = model_dir + "/embed_tokens.npy";

    if (fs::exists(int8_path) && fs::exists(scales_path)) {
        // INT8 mode
        std::vector<size_t> shape;
        embed_int8_ = load_npy_int8(int8_path, shape);

        std::vector<size_t> scales_shape;
        embed_scales_ = load_npy_fp16_as_fp32(scales_path, scales_shape);

        use_int8_embed_ = true;
        log("  INT8 embeddings: [" + std::to_string(shape[0]) + ", " +
            std::to_string(shape[1]) + "] + scales [" +
            std::to_string(scales_shape[0]) + "]");
    } else if (fs::exists(fp16_path)) {
        // FP16 mode
        std::vector<size_t> shape;
        embed_fp32_ = load_npy_fp16_as_fp32(fp16_path, shape);

        use_int8_embed_ = false;
        log("  FP16 embeddings: [" + std::to_string(shape[0]) + ", " +
            std::to_string(shape[1]) + "]");
    } else {
        throw std::runtime_error("No embedding files found in " + model_dir);
    }

    log("  Loaded in " + std::to_string(elapsed_ms(t0)) + " ms");
}

// ---------------------------------------------------------------------------
// Embedding lookup
// ---------------------------------------------------------------------------

void Qwen35GPUModel::embed_lookup(const int64_t* token_ids, int seq_len) {
    int H = hidden_size_;

    if (use_int8_embed_) {
        // INT8 dequantize: result = int8_val * scale
        for (int i = 0; i < seq_len; ++i) {
            int64_t tid = token_ids[i];
            float scale = embed_scales_[tid];
            const int8_t* row = embed_int8_.data() + tid * H;
            float* dst = hidden_buf_.data() + i * H;
            for (int j = 0; j < H; ++j) {
                dst[j] = static_cast<float>(row[j]) * scale;
            }
        }
    } else {
        // FP32 copy
        for (int i = 0; i < seq_len; ++i) {
            std::memcpy(hidden_buf_.data() + i * H,
                        embed_fp32_.data() + token_ids[i] * H,
                        H * sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void Qwen35GPUModel::reset() {
    request_.reset_state();
    past_length_ = 0;
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

std::string Qwen35GPUModel::generate(const std::string& prompt, int max_new_tokens) {
    reset();

    if (!tokenizer_) {
        throw std::runtime_error(
            "Tokenizer not loaded. Provide --tokenizers-lib.");
    }

    auto input_ids = tokenizer_->encode(prompt);
    int prompt_len = static_cast<int>(input_ids.size());
    log("Prompt tokens: " + std::to_string(prompt_len));

    int H = hidden_size_;
    int V = vocab_size_;

    // === Prefill ===
    auto t_prefill = std::chrono::steady_clock::now();

    embed_lookup(input_ids.data(), prompt_len);

    // Create input tensors for prefill
    ov::Tensor hidden_tensor(ov::element::f32,
        {1, static_cast<size_t>(prompt_len), static_cast<size_t>(H)},
        hidden_buf_.data());

    request_.set_tensor("inputs_embeds", hidden_tensor);

    if (has_attention_mask_) {
        ov::Tensor mask_tensor(ov::element::i64,
            {1, static_cast<size_t>(prompt_len)},
            mask_buf_.data());
        request_.set_tensor("attention_mask", mask_tensor);
    }

    if (has_position_ids_) {
        // Text-only: all 3 mRoPE dims identical
        for (int c = 0; c < 3; ++c) {
            for (int s = 0; s < prompt_len; ++s) {
                pos_buf_[c * prompt_len + s] = s;
            }
        }
        ov::Tensor pos_tensor(ov::element::i64,
            {3, 1, static_cast<size_t>(prompt_len)},
            pos_buf_.data());
        request_.set_tensor("position_ids", pos_tensor);
    }

    // beam_idx must be function-scoped (not block-scoped) so the tensor
    // memory stays alive through infer().  IR port type is i32.
    int32_t prefill_beam = 0;
    if (has_beam_idx_) {
        ov::Tensor beam_tensor(ov::element::i32, {1}, &prefill_beam);
        request_.set_tensor("beam_idx", beam_tensor);
    }

    request_.infer();
    past_length_ = prompt_len;

    double prefill_ms = elapsed_ms(t_prefill);
    log("Prefill: " + std::to_string(prompt_len) + " tokens in " +
        fmt_ms(prefill_ms) + " ms (" +
        fmt_ms(prompt_len / (prefill_ms / 1000.0)) + " tok/s)");

    // Get first token from prefill logits
    const float* logits = request_.get_tensor("logits").data<const float>();
    // logits shape: [1, prompt_len, vocab_size] — take last token
    const float* last_logits = logits + (prompt_len - 1) * V;
    int next_id = static_cast<int>(
        std::max_element(last_logits, last_logits + V) - last_logits);

    std::vector<int64_t> generated;
    generated.push_back(next_id);

    // === Decode loop ===
    auto t_decode = std::chrono::steady_clock::now();

    // Pre-allocate decode tensors (S=1)
    ov::Tensor s1_hidden(ov::element::f32, {1, 1, static_cast<size_t>(H)}, hidden_buf_.data());
    int64_t pos_val[3] = {0, 0, 0};
    ov::Tensor s1_pos(ov::element::i64, {3, 1, 1}, pos_val);
    int32_t beam_val = 0;
    ov::Tensor s1_beam(ov::element::i32, {1}, &beam_val);

    // Bind decode tensors once
    request_.set_tensor("inputs_embeds", s1_hidden);
    if (has_position_ids_) request_.set_tensor("position_ids", s1_pos);
    if (has_beam_idx_) request_.set_tensor("beam_idx", s1_beam);

    for (int step = 0; step < max_new_tokens - 1; ++step) {
        if (is_stop_token(next_id)) break;

        // Embed single token
        int64_t tid = static_cast<int64_t>(next_id);
        embed_lookup(&tid, 1);

        // Update attention mask (grows by 1 each step)
        if (has_attention_mask_) {
            int total_len = past_length_ + 1;
            ov::Tensor mask_tensor(ov::element::i64,
                {1, static_cast<size_t>(total_len)},
                mask_buf_.data());
            request_.set_tensor("attention_mask", mask_tensor);
        }

        // Update position
        if (has_position_ids_) {
            pos_val[0] = past_length_;
            pos_val[1] = past_length_;
            pos_val[2] = past_length_;
        }

        request_.infer();
        past_length_ += 1;

        // Argmax on logits [1, 1, V]
        logits = request_.get_tensor("logits").data<const float>();
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
    }

    // Remove trailing stop token
    if (!generated.empty() && is_stop_token(static_cast<int>(generated.back()))) {
        generated.pop_back();
    }

    return tokenizer_->decode(generated);
}
