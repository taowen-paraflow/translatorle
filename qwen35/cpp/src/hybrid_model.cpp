// hybrid_model.cpp — Construction, config loading, model compilation, state init
#include "hybrid_model.h"
#include "tokenizer.h"
#include "npy_reader.h"
#include "utils.h"

#include <fstream>
#include <cstring>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <openvino/pass/make_stateful.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

Qwen35HybridModel::Qwen35HybridModel(
    const std::string& model_dir,
    int attn_past_seq,
    int prefill_chunk_size,
    const std::string& tokenizers_lib)
    : attn_past_seq_(attn_past_seq),
      prefill_chunk_size_(prefill_chunk_size),
      past_length_(0),
      kv_total_(0)
{
    cfg_ = load_config(model_dir);
    log("Config: hidden=" + std::to_string(cfg_.hidden_size) +
        " kv_heads=" + std::to_string(cfg_.num_kv_heads) +
        " head_dim=" + std::to_string(cfg_.head_dim) +
        " blocks=" + std::to_string(cfg_.num_blocks) +
        " prefill_chunk=" + std::to_string(prefill_chunk_size));

    // Enable model caching — NPU compilation is ~10s/block (60s total).
    // After first run, cached blobs make subsequent starts <1s.
    std::string cache_dir = model_dir + "/cache";
    core_.set_property(ov::cache_dir(cache_dir));
    log("Model cache: " + cache_dir);

    load_gdn_blocks(model_dir);
    load_attn_blocks(model_dir);
    load_head(model_dir);
    load_embeddings(model_dir);
    alloc_buffers();

    // Initialize tokenizer
    std::string tok_xml = model_dir + "/openvino_tokenizer.xml";
    if (fs::exists(tok_xml)) {
        if (!tokenizers_lib.empty()) {
            tokenizer_ = std::make_unique<OVTokenizer>(core_, model_dir, tokenizers_lib);
            log("Tokenizer loaded from OV IR");
        } else {
            log("WARNING: Tokenizer IR found but no --tokenizers-lib specified. "
                "Run with --tokenizers-lib path/to/openvino_tokenizers.dll");
        }
    } else {
        log("WARNING: No openvino_tokenizer.xml found. Run convert_tokenizer.py first.");
    }
}

Qwen35HybridModel::~Qwen35HybridModel() = default;

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

ModelConfig Qwen35HybridModel::load_config(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + model_dir + "/config.json");

    json root = json::parse(f);
    json cfg = root.contains("text_config") ? root["text_config"] : root;

    ModelConfig c{};
    c.hidden_size = cfg["hidden_size"].get<int>();
    c.num_kv_heads = cfg["num_key_value_heads"].get<int>();
    int num_attn_heads = cfg["num_attention_heads"].get<int>();
    c.head_dim = cfg.value("head_dim", c.hidden_size / num_attn_heads);
    c.num_v_heads = cfg["linear_num_value_heads"].get<int>();
    c.k_head_dim = cfg["linear_key_head_dim"].get<int>();
    c.v_head_dim = cfg["linear_value_head_dim"].get<int>();
    int num_k_heads = cfg["linear_num_key_heads"].get<int>();
    c.conv_dim = num_k_heads * c.k_head_dim * 2 + c.num_v_heads * c.v_head_dim;
    c.conv_kernel = cfg["linear_conv_kernel_dim"].get<int>();
    c.vocab_size = cfg.value("vocab_size", 248320);

    c.num_blocks = 0;
    while (fs::exists(model_dir + "/gdn_block_" + std::to_string(c.num_blocks) + ".xml"))
        c.num_blocks++;

    return c;
}

// ---------------------------------------------------------------------------
// FP32 output conversion (GPU/NPU output FP16 by default)
// ---------------------------------------------------------------------------

std::shared_ptr<ov::Model> Qwen35HybridModel::add_f32_output_conversion(
    std::shared_ptr<ov::Model> model)
{
    ov::preprocess::PrePostProcessor ppp(model);
    for (size_t i = 0; i < model->outputs().size(); ++i) {
        ppp.output(i).tensor().set_element_type(ov::element::f32);
    }
    return ppp.build();
}

// ---------------------------------------------------------------------------
// GDN block loading (GPU, stateful)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_gdn_blocks(const std::string& model_dir) {
    log("Compiling " + std::to_string(cfg_.num_blocks) + " GDN blocks on GPU (stateful)...");
    auto t0 = std::chrono::steady_clock::now();

    // 3 layers per block × (conv + rec) = 6 state pairs
    std::map<std::string, std::string> state_map;
    for (int j = 0; j < 3; ++j) {
        state_map["in_conv" + std::to_string(j)] = "out_conv" + std::to_string(j);
        state_map["in_rec" + std::to_string(j)] = "out_rec" + std::to_string(j);
    }

    for (int i = 0; i < cfg_.num_blocks; ++i) {
        std::string xml = model_dir + "/gdn_block_" + std::to_string(i) + ".xml";
        auto model = core_.read_model(xml);
        ov::pass::MakeStateful(state_map).run_on_model(model);
        model = add_f32_output_conversion(model);
        auto compiled = core_.compile_model(model, "GPU",
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::hint::num_requests(1));
        gdn_models_.push_back(compiled);
        gdn_requests_.push_back(compiled.create_infer_request());
    }

    init_gdn_states();
    log("  GDN compilation: " + std::to_string(elapsed_ms(t0)) + " ms");
}

// ---------------------------------------------------------------------------
// Attention block loading (NPU, explicit I/O with ScatterUpdate-3 fixed KV)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_attn_blocks(const std::string& model_dir) {
    int N = cfg_.num_blocks;
    int H = cfg_.num_kv_heads;
    int P = attn_past_seq_;
    int D = cfg_.head_dim;
    int C = prefill_chunk_size_;
    int hidden = cfg_.hidden_size;

    // --- Decode models (S=1) ---
    log("Compiling " + std::to_string(N) + " Attn blocks on NPU (S=1, KV=" +
        std::to_string(P) + ")...");
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < N; ++i) {
        std::string xml = model_dir + "/attn_block_" + std::to_string(i) + ".xml";
        auto model = core_.read_model(xml);

        // Reshape to static shapes for NPU — 6 inputs (ScatterUpdate-3 IR):
        //   0: hidden [1,1,hidden_size]
        //   1: position_ids [3,1,1]
        //   2: key_cache [1,H,P,D]
        //   3: value_cache [1,H,P,D]
        //   4: cache_position [1]
        //   5: attention_mask [1,1,1,P]
        std::map<std::string, ov::PartialShape> shapes;
        auto inputs = model->inputs();
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            std::string name = inputs[idx].get_any_name();
            switch (idx) {
                case 0: shapes[name] = {1, 1, hidden}; break;
                case 1: shapes[name] = {3, 1, 1}; break;
                case 2: case 3: shapes[name] = {1, H, P, D}; break;
                case 4: shapes[name] = {1}; break;
                case 5: shapes[name] = {1, 1, 1, P}; break;
            }
        }
        model->reshape(shapes);
        model = add_f32_output_conversion(model);

        ov::AnyMap npu_config = {
            {"NPU_COMPILER_TYPE", "PREFER_PLUGIN"},
            {ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY},
            {ov::hint::num_requests.name(), uint32_t(1)}
        };
        auto compiled = core_.compile_model(model, "NPU", npu_config);
        attn_models_.push_back(compiled);
        attn_requests_.push_back(compiled.create_infer_request());
    }
    log("  Attn decode compilation: " + std::to_string(elapsed_ms(t0)) + " ms");

    // --- Prefill models (S=chunk_size) ---
    if (C > 1) {
        log("Compiling " + std::to_string(N) + " prefill Attn blocks on NPU (S=" +
            std::to_string(C) + ")...");
        t0 = std::chrono::steady_clock::now();

        for (int i = 0; i < N; ++i) {
            std::string xml = model_dir + "/attn_block_" + std::to_string(i) + ".xml";
            auto model = core_.read_model(xml);

            std::map<std::string, ov::PartialShape> shapes;
            auto inputs = model->inputs();
            for (size_t idx = 0; idx < inputs.size(); ++idx) {
                std::string name = inputs[idx].get_any_name();
                switch (idx) {
                    case 0: shapes[name] = {1, C, hidden}; break;
                    case 1: shapes[name] = {3, 1, C}; break;
                    case 2: case 3: shapes[name] = {1, H, P, D}; break;
                    case 4: shapes[name] = {C}; break;
                    case 5: shapes[name] = {1, 1, C, P}; break;
                }
            }
            model->reshape(shapes);
            model = add_f32_output_conversion(model);

            ov::AnyMap npu_config = {
                {"NPU_COMPILER_TYPE", "PREFER_PLUGIN"},
                {ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY},
                {ov::hint::num_requests.name(), uint32_t(1)}
            };
            auto compiled = core_.compile_model(model, "NPU", npu_config);
            attn_prefill_models_.push_back(compiled);
            attn_prefill_requests_.push_back(compiled.create_infer_request());
        }
        log("  Attn prefill compilation: " + std::to_string(elapsed_ms(t0)) + " ms");
    }

    init_kv_caches();
}

// ---------------------------------------------------------------------------
// Head block loading (GPU, stateless)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_head(const std::string& model_dir) {
    log("Compiling Head on GPU...");
    auto t0 = std::chrono::steady_clock::now();

    auto model = core_.read_model(model_dir + "/head.xml");
    model = add_f32_output_conversion(model);
    head_model_ = core_.compile_model(model, "GPU",
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
        ov::hint::num_requests(1));
    head_request_ = head_model_.create_infer_request();

    log("  Head compilation: " + std::to_string(elapsed_ms(t0)) + " ms");
}

// ---------------------------------------------------------------------------
// Embedding loading
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_embeddings(const std::string& model_dir) {
    log("Loading embed_tokens.npy...");
    auto t0 = std::chrono::steady_clock::now();

    std::vector<size_t> shape;
    embed_table_ = load_npy_fp16_as_fp32(model_dir + "/embed_tokens.npy", shape);

    log("  Embeddings: [" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
        "] loaded in " + std::to_string(elapsed_ms(t0)) + " ms");
}

// ---------------------------------------------------------------------------
// KV cache initialization (fixed-size, zero-filled)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::init_kv_caches() {
    int N = cfg_.num_blocks;
    int H = cfg_.num_kv_heads;
    int P = attn_past_seq_;
    int D = cfg_.head_dim;
    kv_total_ = H * P * D;

    kv_key_tensors_.clear();
    kv_value_tensors_.clear();
    for (int i = 0; i < N; ++i) {
        ov::Tensor key(ov::element::f32,
                       {1, static_cast<size_t>(H),
                        static_cast<size_t>(P),
                        static_cast<size_t>(D)});
        ov::Tensor val(ov::element::f32,
                       {1, static_cast<size_t>(H),
                        static_cast<size_t>(P),
                        static_cast<size_t>(D)});
        std::memset(key.data<float>(), 0, kv_total_ * sizeof(float));
        std::memset(val.data<float>(), 0, kv_total_ * sizeof(float));
        kv_key_tensors_.push_back(std::move(key));
        kv_value_tensors_.push_back(std::move(val));
    }
}

// ---------------------------------------------------------------------------
// Pre-allocate buffers and tensor wrappers
// ---------------------------------------------------------------------------

void Qwen35HybridModel::alloc_buffers() {
    int max_seq = std::max(1, prefill_chunk_size_);
    int H = cfg_.hidden_size;
    int P = attn_past_seq_;
    int C = prefill_chunk_size_;

    // Allocate raw buffers
    hidden_buf_.resize(max_seq * H, 0.0f);
    gdn_mask_buf_.assign(max_seq, 1LL);    // all ones, never changes
    pos_buf_.resize(3 * max_seq, 0LL);
    cache_pos_buf_.resize(max_seq, 0LL);
    attn_mask_buf_.resize(max_seq * P, 0.0f);

    // Create tensor wrappers for decode (S=1) — wraps raw buffers, no copy
    s1_hidden_ = ov::Tensor(ov::element::f32, {1, 1, static_cast<size_t>(H)}, hidden_buf_.data());
    s1_gdn_mask_ = ov::Tensor(ov::element::i64, {1, 1}, gdn_mask_buf_.data());
    s1_pos_ = ov::Tensor(ov::element::i64, {3, 1, 1}, pos_buf_.data());
    s1_cache_pos_ = ov::Tensor(ov::element::i64, {1}, cache_pos_buf_.data());
    s1_attn_mask_ = ov::Tensor(ov::element::f32,
                                {1, 1, 1, static_cast<size_t>(P)},
                                attn_mask_buf_.data());

    // Create tensor wrappers for prefill (S=chunk_size)
    if (C > 1) {
        sc_hidden_ = ov::Tensor(ov::element::f32,
                                 {1, static_cast<size_t>(C), static_cast<size_t>(H)},
                                 hidden_buf_.data());
        sc_gdn_mask_ = ov::Tensor(ov::element::i64,
                                    {1, static_cast<size_t>(C)},
                                    gdn_mask_buf_.data());
        sc_pos_ = ov::Tensor(ov::element::i64,
                               {3, 1, static_cast<size_t>(C)},
                               pos_buf_.data());
        sc_cache_pos_ = ov::Tensor(ov::element::i64,
                                     {static_cast<size_t>(C)},
                                     cache_pos_buf_.data());
        sc_attn_mask_ = ov::Tensor(ov::element::f32,
                                     {1, 1, static_cast<size_t>(C), static_cast<size_t>(P)},
                                     attn_mask_buf_.data());
    }
}

// ---------------------------------------------------------------------------
// GDN state initialization
// ---------------------------------------------------------------------------

void Qwen35HybridModel::init_gdn_states() {
    int conv_size = cfg_.conv_dim * cfg_.conv_kernel;
    int rec_size = cfg_.num_v_heads * cfg_.k_head_dim * cfg_.v_head_dim;

    for (auto& req : gdn_requests_) {
        for (auto& s : req.query_state()) {
            std::string name = s.get_name();
            if (name.find("conv") != std::string::npos) {
                ov::Tensor t(ov::element::f32,
                             {1, static_cast<size_t>(cfg_.conv_dim),
                              static_cast<size_t>(cfg_.conv_kernel)});
                std::memset(t.data<float>(), 0, conv_size * sizeof(float));
                s.set_state(t);
            } else if (name.find("rec") != std::string::npos) {
                ov::Tensor t(ov::element::f32,
                             {1, static_cast<size_t>(cfg_.num_v_heads),
                              static_cast<size_t>(cfg_.k_head_dim),
                              static_cast<size_t>(cfg_.v_head_dim)});
                std::memset(t.data<float>(), 0, rec_size * sizeof(float));
                s.set_state(t);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reset all states for a new generation
// ---------------------------------------------------------------------------

void Qwen35HybridModel::reset() {
    init_gdn_states();
    // Reset KV caches to zeros (fixed-size, stays allocated)
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        std::memset(kv_key_tensors_[i].data<float>(), 0, kv_total_ * sizeof(float));
        std::memset(kv_value_tensors_[i].data<float>(), 0, kv_total_ * sizeof(float));
    }
    past_length_ = 0;
}
