// hybrid_model.cpp — Construction, config loading, model compilation, state init
#include "hybrid_model.h"
#include "gdn_fuse_pass.h"
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
    const std::string& tokenizers_lib,
    bool use_latency_hint,
    bool no_gdn_prefill,
    bool timing)
    : attn_past_seq_(attn_past_seq),
      prefill_chunk_size_(prefill_chunk_size),
      past_length_(0),
      kv_total_(0),
      use_latency_hint_(use_latency_hint),
      no_gdn_prefill_(no_gdn_prefill),
      timing_(timing)
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

    // Load order matters for thermal management on Lunar Lake iGPU:
    // NPU models first (doesn't heat GPU), then GPU models.
    // This gives the GPU cooling time between compilation and inference.
    load_attn_blocks(model_dir);         // NPU — no GPU heat
    load_gdn_blocks(model_dir);          // GPU — decode (Loop fallback)
    load_gdn_noloop_blocks(model_dir);   // GPU — decode (flat ops, preferred)
    load_head(model_dir);                // GPU — decode critical
    if (!no_gdn_prefill) {
        load_gdn_prefill_blocks(model_dir);  // GPU — prefill only, loaded last
    } else {
        log("Skipping GDN prefill blocks (--no-gdn-prefill)");
    }
    load_embeddings(model_dir);
    alloc_buffers();
    bind_decode_tensors();

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
    log("Compiling " + std::to_string(cfg_.num_blocks) +
        " GDN Loop-based blocks on GPU (stateful)...");
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
        // Stateful: state lives in GPU VRAM (faster ~15%)
        ov::pass::MakeStateful(state_map).run_on_model(model);
        model = add_f32_output_conversion(model);
        ov::AnyMap gpu_config = {
            {ov::hint::num_requests.name(), uint32_t(1)}
        };
        if (use_latency_hint_) {
            gpu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        }
        auto compiled = core_.compile_model(model, "GPU", gpu_config);
        gdn_models_.push_back(compiled);
        gdn_requests_.push_back(compiled.create_infer_request());
    }

    init_gdn_states();
    log("  GDN compilation: " + std::to_string(elapsed_ms(t0)) + " ms (Loop-based)");
}

// ---------------------------------------------------------------------------
// GDN noloop block loading (GPU, stateful, flat ops — no Loop node)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_gdn_noloop_blocks(const std::string& model_dir) {
    // Prefer quantized gdn_s1_block_*, fall back to gdn_noloop_block_*
    std::string prefix = "gdn_s1_block_";
    std::string test_xml = model_dir + "/" + prefix + "0.xml";
    if (!fs::exists(test_xml)) {
        prefix = "gdn_noloop_block_";
        test_xml = model_dir + "/" + prefix + "0.xml";
    }
    if (!fs::exists(test_xml)) {
        log("No noloop GDN blocks found, using Loop-based decode");
        return;
    }

    log("Compiling " + std::to_string(cfg_.num_blocks) +
        " GDN noloop blocks on GPU (stateful, no Loop)...");
    auto t0 = std::chrono::steady_clock::now();

    // Same state map as Loop blocks — same I/O naming convention
    std::map<std::string, std::string> state_map;
    for (int j = 0; j < 3; ++j) {
        state_map["in_conv" + std::to_string(j)] = "out_conv" + std::to_string(j);
        state_map["in_rec" + std::to_string(j)] = "out_rec" + std::to_string(j);
    }

    // Locate kernel files for fused GDN recurrence (if available)
    std::string config_xml = model_dir + "/fused_gdn_recurrence.xml";
    std::string kernel_cl = model_dir + "/gdn_recurrence.cl";
    bool try_fuse = fs::exists(config_xml) && fs::exists(kernel_cl);
    if (try_fuse) {
        log("  Found fused GDN kernel config: " + config_xml);
    }

    for (int i = 0; i < cfg_.num_blocks; ++i) {
        std::string xml = model_dir + "/" + prefix + std::to_string(i) + ".xml";
        auto model = core_.read_model(xml);

        // Try to fuse recurrence subgraphs before MakeStateful
        int fused = 0;
        if (try_fuse) {
            fused = fuse_gdn_recurrence(model);
            if (fused > 0) {
                log("  Block " + std::to_string(i) + ": fused " +
                    std::to_string(fused) + " recurrence subgraphs");
            }
        }

        ov::pass::MakeStateful(state_map).run_on_model(model);
        model = add_f32_output_conversion(model);
        ov::AnyMap gpu_config = {
            {ov::hint::num_requests.name(), uint32_t(1)}
        };
        if (use_latency_hint_) {
            gpu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        }
        if (fused > 0) {
            gpu_config["CONFIG_FILE"] = config_xml;
        }
        auto compiled = core_.compile_model(model, "GPU", gpu_config);
        gdn_noloop_models_.push_back(compiled);
        gdn_noloop_requests_.push_back(compiled.create_infer_request());
    }

    has_gdn_noloop_ = true;

    // Initialize states (same layout as Loop blocks)
    int conv_size = cfg_.conv_dim * cfg_.conv_kernel;
    int rec_size = cfg_.num_v_heads * cfg_.k_head_dim * cfg_.v_head_dim;
    for (auto& req : gdn_noloop_requests_) {
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

    log("  GDN noloop compilation: " + std::to_string(elapsed_ms(t0)) + " ms");
}

// ---------------------------------------------------------------------------
// GDN prefill block loading (GPU, explicit I/O, chunkwise parallel)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_gdn_prefill_blocks(const std::string& model_dir) {
    std::string test_xml = model_dir + "/gdn_prefill_block_0.xml";
    if (!fs::exists(test_xml)) {
        log("No chunkwise GDN prefill blocks found, using Loop-based prefill");
        return;
    }

    log("Compiling " + std::to_string(cfg_.num_blocks) +
        " GDN prefill blocks on GPU (explicit I/O, FP32)...");
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < cfg_.num_blocks; ++i) {
        std::string xml = model_dir + "/gdn_prefill_block_" + std::to_string(i) + ".xml";
        auto model = core_.read_model(xml);
        model = add_f32_output_conversion(model);

        // Force FP32 inference: the Neumann series (7 matrix squarings)
        // accumulates catastrophic precision loss in FP16.
        ov::AnyMap gpu_config = {
            {ov::hint::num_requests.name(), uint32_t(1)},
            {"INFERENCE_PRECISION_HINT", "f32"}
        };
        if (use_latency_hint_) {
            gpu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        }
        auto compiled = core_.compile_model(model, "GPU", gpu_config);
        gdn_prefill_models_.push_back(compiled);
        gdn_prefill_requests_.push_back(compiled.create_infer_request());
    }

    has_gdn_prefill_ = true;
    init_gdn_prefill_states();
    log("  GDN prefill compilation: " + std::to_string(elapsed_ms(t0)) + " ms");
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
            {ov::hint::num_requests.name(), uint32_t(1)}
        };
        if (use_latency_hint_) {
            npu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        }
        auto compiled = core_.compile_model(model, "NPU", npu_config);
        attn_models_.push_back(compiled);
        attn_requests_.push_back(compiled.create_infer_request());
    }
    log("  Attn decode compilation: " + std::to_string(elapsed_ms(t0)) + " ms");

    // --- Prefill models (descending powers of 2: 16, 8, 4, 2) ---
    if (C > 1) {
        // Collect chunk sizes: e.g. 16 -> {16, 8, 4, 2}
        std::vector<int> chunk_sizes;
        for (int cs = C; cs >= 2; cs /= 2)
            chunk_sizes.push_back(cs);

        std::string sizes_str;
        for (size_t si = 0; si < chunk_sizes.size(); ++si) {
            if (si > 0) sizes_str += ",";
            sizes_str += std::to_string(chunk_sizes[si]);
        }
        log("Compiling " + std::to_string(N) + " prefill Attn blocks on NPU for S=[" +
            sizes_str + "]...");
        t0 = std::chrono::steady_clock::now();

        for (int cs : chunk_sizes) {
            std::vector<ov::CompiledModel> models;
            std::vector<ov::InferRequest> requests;
            for (int i = 0; i < N; ++i) {
                std::string xml = model_dir + "/attn_block_" + std::to_string(i) + ".xml";
                auto model = core_.read_model(xml);

                std::map<std::string, ov::PartialShape> shapes;
                auto inputs = model->inputs();
                for (size_t idx = 0; idx < inputs.size(); ++idx) {
                    std::string name = inputs[idx].get_any_name();
                    switch (idx) {
                        case 0: shapes[name] = {1, cs, hidden}; break;
                        case 1: shapes[name] = {3, 1, cs}; break;
                        case 2: case 3: shapes[name] = {1, H, P, D}; break;
                        case 4: shapes[name] = {cs}; break;
                        case 5: shapes[name] = {1, 1, cs, P}; break;
                    }
                }
                model->reshape(shapes);
                model = add_f32_output_conversion(model);

                ov::AnyMap npu_config = {
                    {"NPU_COMPILER_TYPE", "PREFER_PLUGIN"},
                    {ov::hint::num_requests.name(), uint32_t(1)}
                };
                if (use_latency_hint_) {
                    npu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
                }
                auto compiled = core_.compile_model(model, "NPU", npu_config);
                models.push_back(compiled);
                requests.push_back(compiled.create_infer_request());
            }
            attn_prefill_models_[cs] = std::move(models);
            attn_prefill_requests_[cs] = std::move(requests);
        }
        log("  Attn prefill compilation (" + std::to_string(chunk_sizes.size()) +
            " sizes): " + std::to_string(elapsed_ms(t0)) + " ms");
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
    ov::AnyMap gpu_config = {
        {ov::hint::num_requests.name(), uint32_t(1)}
    };
    if (use_latency_hint_) {
        gpu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    }
    head_model_ = core_.compile_model(model, "GPU", gpu_config);
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

    // Pre-allocate prefill buffers (max prompt = attn_past_seq tokens, one alloc at init)
    prefill_hidden_buf_.resize(P * H, 0.0f);
    prefill_gdn_mask_.assign(P, 1LL);

    // Create tensor wrappers for decode (S=1) — wraps raw buffers, no copy
    s1_hidden_ = ov::Tensor(ov::element::f32, {1, 1, static_cast<size_t>(H)}, hidden_buf_.data());
    s1_gdn_mask_ = ov::Tensor(ov::element::i64, {1, 1}, gdn_mask_buf_.data());
    s1_pos_ = ov::Tensor(ov::element::i64, {3, 1, 1}, pos_buf_.data());
    s1_cache_pos_ = ov::Tensor(ov::element::i64, {1}, cache_pos_buf_.data());
    s1_attn_mask_ = ov::Tensor(ov::element::f32,
                                {1, 1, 1, static_cast<size_t>(P)},
                                attn_mask_buf_.data());

    // Create tensor wrappers for prefill — one set per chunk size (16, 8, 4, 2)
    // All share the same underlying buffers (hidden_buf_, etc.) since only one
    // chunk size is active at a time during prefill.
    for (int cs = C; cs >= 2; cs /= 2) {
        size_t S = static_cast<size_t>(cs);
        sc_hidden_[cs] = ov::Tensor(ov::element::f32,
                                     {1, S, static_cast<size_t>(H)},
                                     hidden_buf_.data());
        sc_gdn_mask_[cs] = ov::Tensor(ov::element::i64,
                                        {1, S},
                                        gdn_mask_buf_.data());
        sc_pos_[cs] = ov::Tensor(ov::element::i64,
                                   {3, 1, S},
                                   pos_buf_.data());
        sc_cache_pos_[cs] = ov::Tensor(ov::element::i64,
                                         {S},
                                         cache_pos_buf_.data());
        sc_attn_mask_[cs] = ov::Tensor(ov::element::f32,
                                         {1, 1, S, static_cast<size_t>(P)},
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
// GDN prefill state initialization (explicit buffers, zero-filled)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::init_gdn_prefill_states() {
    // Guard: only allocate once (shared by decode S1 + prefill)
    if (!gdn_prefill_conv_states_.empty()) return;

    int conv_dim = cfg_.conv_dim;
    int conv_kernel = cfg_.conv_kernel;
    int num_v = cfg_.num_v_heads;
    int kd = cfg_.k_head_dim;
    int vd = cfg_.v_head_dim;

    gdn_prefill_conv_states_.resize(cfg_.num_blocks);
    gdn_prefill_rec_states_.resize(cfg_.num_blocks);

    for (int i = 0; i < cfg_.num_blocks; ++i) {
        gdn_prefill_conv_states_[i].resize(3);
        gdn_prefill_rec_states_[i].resize(3);
        for (int j = 0; j < 3; ++j) {
            gdn_prefill_conv_states_[i][j] = ov::Tensor(ov::element::f32,
                {1, static_cast<size_t>(conv_dim), static_cast<size_t>(conv_kernel)});
            std::memset(gdn_prefill_conv_states_[i][j].data<float>(), 0,
                        conv_dim * conv_kernel * sizeof(float));

            gdn_prefill_rec_states_[i][j] = ov::Tensor(ov::element::f32,
                {1, static_cast<size_t>(num_v), static_cast<size_t>(kd),
                 static_cast<size_t>(vd)});
            std::memset(gdn_prefill_rec_states_[i][j].data<float>(), 0,
                        num_v * kd * vd * sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer chunkwise prefill states to stateful decode GDN blocks
// ---------------------------------------------------------------------------

void Qwen35HybridModel::transfer_prefill_states_to_decode() {
    // Transfer to Loop-based decode blocks
    for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
        for (auto& s : gdn_requests_[blk].query_state()) {
            std::string name = s.get_name();
            for (int j = 0; j < 3; ++j) {
                if (name.find("conv" + std::to_string(j)) != std::string::npos) {
                    s.set_state(gdn_prefill_conv_states_[blk][j]);
                    break;
                }
                if (name.find("rec" + std::to_string(j)) != std::string::npos) {
                    s.set_state(gdn_prefill_rec_states_[blk][j]);
                    break;
                }
            }
        }
    }

    // Transfer to noloop decode blocks (if available)
    if (has_gdn_noloop_) {
        for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
            for (auto& s : gdn_noloop_requests_[blk].query_state()) {
                std::string name = s.get_name();
                for (int j = 0; j < 3; ++j) {
                    if (name.find("conv" + std::to_string(j)) != std::string::npos) {
                        s.set_state(gdn_prefill_conv_states_[blk][j]);
                        break;
                    }
                    if (name.find("rec" + std::to_string(j)) != std::string::npos) {
                        s.set_state(gdn_prefill_rec_states_[blk][j]);
                        break;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Initialize decode attention mask for incremental updates.
// Sets positions [0..past_length_] to 0.0 (attend) and rest to -65504.0 (mask).
// After this, each decode step only writes one float: mask[past_length_] = 0.0f.
// ---------------------------------------------------------------------------

void Qwen35HybridModel::init_decode_attn_mask() {
    const float MASK_VAL = -65504.0f;
    int P = attn_past_seq_;
    float* mask = attn_mask_buf_.data();
    int valid = std::min(past_length_ + 1, P);
    std::memset(mask, 0, valid * sizeof(float));
    std::fill_n(mask + valid, P - valid, MASK_VAL);
}

// ---------------------------------------------------------------------------
// Pre-bind decode tensors (S=1) — call once after alloc_buffers()
// Avoids 55+ set_input/output_tensor calls per decode step.
// Only position_ids, cache_position, and attn_mask content change per step.
// ---------------------------------------------------------------------------

void Qwen35HybridModel::bind_decode_tensors() {
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        auto& gdn_req = gdn_requests_[i];
        gdn_req.set_input_tensor(0, s1_hidden_);
        gdn_req.set_input_tensor(1, s1_gdn_mask_);
        gdn_req.set_output_tensor(0, s1_hidden_);

        // Noloop blocks have same I/O after MakeStateful: hidden + mask → hidden
        if (has_gdn_noloop_) {
            auto& noloop_req = gdn_noloop_requests_[i];
            noloop_req.set_input_tensor(0, s1_hidden_);
            noloop_req.set_input_tensor(1, s1_gdn_mask_);
            noloop_req.set_output_tensor(0, s1_hidden_);
        }

        auto& attn_req = attn_requests_[i];
        attn_req.set_input_tensor(0, s1_hidden_);
        attn_req.set_input_tensor(1, s1_pos_);
        attn_req.set_input_tensor(2, kv_key_tensors_[i]);
        attn_req.set_input_tensor(3, kv_value_tensors_[i]);
        attn_req.set_input_tensor(4, s1_cache_pos_);
        attn_req.set_input_tensor(5, s1_attn_mask_);
    }
    head_request_.set_input_tensor(0, s1_hidden_);

    // Initialize mask for incremental decode updates
    init_decode_attn_mask();
}

// ---------------------------------------------------------------------------
// Reset all states for a new generation
// ---------------------------------------------------------------------------

void Qwen35HybridModel::reset() {
    init_gdn_states();
    // Reset noloop block states
    if (has_gdn_noloop_) {
        int conv_size = cfg_.conv_dim * cfg_.conv_kernel;
        int rec_size = cfg_.num_v_heads * cfg_.k_head_dim * cfg_.v_head_dim;
        for (auto& req : gdn_noloop_requests_) {
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
    if (has_gdn_prefill_) {
        // Prefill blocks have separate state buffers — zero them
        for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
            for (int j = 0; j < 3; ++j) {
                std::memset(gdn_prefill_conv_states_[blk][j].data<float>(), 0,
                            gdn_prefill_conv_states_[blk][j].get_byte_size());
                std::memset(gdn_prefill_rec_states_[blk][j].data<float>(), 0,
                            gdn_prefill_rec_states_[blk][j].get_byte_size());
            }
        }
    }
    // Reset KV caches to zeros (fixed-size, stays allocated)
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        std::memset(kv_key_tensors_[i].data<float>(), 0, kv_total_ * sizeof(float));
        std::memset(kv_value_tensors_[i].data<float>(), 0, kv_total_ * sizeof(float));
    }
    past_length_ = 0;
}
