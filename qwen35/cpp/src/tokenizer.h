#pragma once

#include <string>
#include <vector>

#include <openvino/openvino.hpp>

/// Wrapper around OpenVINO tokenizer / detokenizer IR models.
///
/// Requires the openvino_tokenizers extension library to be loaded so that
/// the custom string-processing operations are registered with the Core.
class OVTokenizer {
public:
    /// @param core        Shared OpenVINO Core instance (must outlive this object).
    /// @param model_dir   Directory containing openvino_tokenizer.xml/.bin and
    ///                    openvino_detokenizer.xml/.bin.
    /// @param tokenizers_lib  Path to the openvino_tokenizers shared library
    ///                        (.dll on Windows, .so on Linux).
    OVTokenizer(ov::Core& core,
                const std::string& model_dir,
                const std::string& tokenizers_lib);

    /// Encode a UTF-8 string into a sequence of token IDs.
    std::vector<int64_t> encode(const std::string& text) const;

    /// Decode a sequence of token IDs back into a UTF-8 string.
    std::string decode(const std::vector<int64_t>& token_ids) const;

private:
    mutable ov::InferRequest tok_request_;
    mutable ov::InferRequest detok_request_;
};
