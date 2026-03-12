#include "tokenizer.h"

#include <stdexcept>

OVTokenizer::OVTokenizer(ov::Core& core,
                         const std::string& model_dir,
                         const std::string& tokenizers_lib) {
    // Register the openvino_tokenizers extension so that custom string ops
    // (StringTensorUnpack, BPETokenizer, etc.) are available.
    core.add_extension(tokenizers_lib);

    // Compile tokenizer (encode) model on CPU -- string ops are CPU-only.
    auto tok_model = core.read_model(model_dir + "/openvino_tokenizer.xml");
    auto tok_compiled = core.compile_model(tok_model, "CPU");
    tok_request_ = tok_compiled.create_infer_request();

    // Compile detokenizer (decode) model on CPU.
    auto detok_model = core.read_model(model_dir + "/openvino_detokenizer.xml");
    auto detok_compiled = core.compile_model(detok_model, "CPU");
    detok_request_ = detok_compiled.create_infer_request();
}

std::vector<int64_t> OVTokenizer::encode(const std::string& text) const {
    // Build a string tensor of shape {1} as input.
    ov::Tensor input_tensor(ov::element::string, ov::Shape{1});
    std::string* str_data = input_tensor.data<std::string>();
    str_data[0] = text;

    tok_request_.set_input_tensor(input_tensor);
    tok_request_.infer();

    // The first output tensor contains int64 token IDs with shape {1, seq_len}.
    ov::Tensor output_tensor = tok_request_.get_output_tensor(0);
    const int64_t* ids = output_tensor.data<int64_t>();
    size_t count = output_tensor.get_size();

    return std::vector<int64_t>(ids, ids + count);
}

std::string OVTokenizer::decode(const std::vector<int64_t>& token_ids) const {
    if (token_ids.empty()) {
        return {};
    }

    size_t n = token_ids.size();

    // Build an int64 tensor of shape {1, N}.
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, n});
    int64_t* dst = input_tensor.data<int64_t>();
    std::copy(token_ids.begin(), token_ids.end(), dst);

    detok_request_.set_input_tensor(input_tensor);
    detok_request_.infer();

    // The output is a string tensor of shape {1}.
    ov::Tensor output_tensor = detok_request_.get_output_tensor(0);
    std::string* out_str = output_tensor.data<std::string>();

    return out_str[0];
}
