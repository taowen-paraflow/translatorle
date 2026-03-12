#include "npy_reader.h"
#include "half.h"

#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <sstream>

// ---------------------------------------------------------------------------
// NPY v1.0/v2.0 format:
//   6 bytes  magic: \x93NUMPY
//   1 byte   major version
//   1 byte   minor version
//   2 bytes  (v1) or 4 bytes (v2) header_len (little-endian)
//   header_len bytes: Python dict literal, padded with spaces + '\n'
//   raw data follows immediately
//
// We only support descr='<f2' (little-endian float16).
// ---------------------------------------------------------------------------

static uint16_t read_le16(const char* p) {
    auto u = reinterpret_cast<const uint8_t*>(p);
    return static_cast<uint16_t>(u[0]) | (static_cast<uint16_t>(u[1]) << 8);
}

static uint32_t read_le32(const char* p) {
    auto u = reinterpret_cast<const uint8_t*>(p);
    return static_cast<uint32_t>(u[0])
         | (static_cast<uint32_t>(u[1]) << 8)
         | (static_cast<uint32_t>(u[2]) << 16)
         | (static_cast<uint32_t>(u[3]) << 24);
}

// Extract the string value for a given key from a Python dict literal.
// E.g., for key="descr" in "{'descr': '<f2', ...}", returns "<f2".
static std::string extract_string_value(const std::string& header,
                                        const std::string& key) {
    std::string search = "'" + key + "'";
    auto pos = header.find(search);
    if (pos == std::string::npos) {
        throw std::runtime_error("npy header: missing key '" + key + "'");
    }
    // Skip past the key and find the opening quote of the value.
    pos = header.find('\'', pos + search.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("npy header: malformed value for '" + key + "'");
    }
    auto end = header.find('\'', pos + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("npy header: unterminated string for '" + key + "'");
    }
    return header.substr(pos + 1, end - pos - 1);
}

// Extract the boolean value for 'fortran_order' from the header.
static bool extract_fortran_order(const std::string& header) {
    auto pos = header.find("'fortran_order'");
    if (pos == std::string::npos) {
        throw std::runtime_error("npy header: missing 'fortran_order'");
    }
    // Look for True or False after the colon.
    auto colon = header.find(':', pos);
    if (colon == std::string::npos) {
        throw std::runtime_error("npy header: malformed 'fortran_order'");
    }
    auto rest = header.substr(colon + 1);
    if (rest.find("True") < rest.find("False")) {
        return true;
    }
    return false;
}

// Extract the shape tuple, e.g., "(248320, 1024)" or "(100,)" for 1D.
static std::vector<size_t> extract_shape(const std::string& header) {
    auto pos = header.find("'shape'");
    if (pos == std::string::npos) {
        throw std::runtime_error("npy header: missing 'shape'");
    }
    auto open = header.find('(', pos);
    auto close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("npy header: malformed 'shape'");
    }
    std::string inside = header.substr(open + 1, close - open - 1);

    std::vector<size_t> shape;
    std::istringstream iss(inside);
    std::string token;
    while (std::getline(iss, token, ',')) {
        // Trim whitespace.
        size_t start = token.find_first_not_of(" \t");
        if (start == std::string::npos) continue; // trailing comma
        size_t end = token.find_last_not_of(" \t");
        std::string num = token.substr(start, end - start + 1);
        if (num.empty()) continue;
        shape.push_back(static_cast<size_t>(std::stoull(num)));
    }
    return shape;
}

std::vector<float> load_npy_fp16_as_fp32(const std::string& path,
                                         std::vector<size_t>& shape_out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // 1. Read and verify magic bytes: \x93NUMPY
    char magic[6];
    file.read(magic, 6);
    if (!file || magic[0] != '\x93' || std::strncmp(magic + 1, "NUMPY", 5) != 0) {
        throw std::runtime_error("Not a valid .npy file: " + path);
    }

    // 2. Read version.
    uint8_t major = 0, minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    if (!file) {
        throw std::runtime_error("Failed to read npy version: " + path);
    }

    // 3. Read header length.
    uint32_t header_len = 0;
    if (major == 1) {
        char buf[2];
        file.read(buf, 2);
        if (!file) throw std::runtime_error("Failed to read v1 header length: " + path);
        header_len = read_le16(buf);
    } else if (major == 2) {
        char buf[4];
        file.read(buf, 4);
        if (!file) throw std::runtime_error("Failed to read v2 header length: " + path);
        header_len = read_le32(buf);
    } else {
        throw std::runtime_error("Unsupported npy version " + std::to_string(major) +
                                 "." + std::to_string(minor) + ": " + path);
    }

    // 4. Read header dict string.
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);
    if (!file) {
        throw std::runtime_error("Failed to read npy header: " + path);
    }

    // 5. Parse header fields.
    std::string descr = extract_string_value(header, "descr");
    if (descr != "<f2") {
        throw std::runtime_error("Unsupported dtype '" + descr +
                                 "', expected '<f2' (float16): " + path);
    }

    if (extract_fortran_order(header)) {
        throw std::runtime_error("Fortran order not supported: " + path);
    }

    shape_out = extract_shape(header);

    // Compute total number of elements.
    size_t num_elements = 1;
    for (size_t dim : shape_out) {
        num_elements *= dim;
    }
    if (num_elements == 0) {
        return {};
    }

    // 6. Read raw FP16 data into a uint16_t buffer.
    size_t data_bytes = num_elements * sizeof(uint16_t);
    std::vector<uint16_t> fp16_buf(num_elements);
    file.read(reinterpret_cast<char*>(fp16_buf.data()), static_cast<std::streamsize>(data_bytes));
    if (!file) {
        throw std::runtime_error("Failed to read " + std::to_string(data_bytes) +
                                 " bytes of data from: " + path);
    }

    // 7. Convert FP16 -> FP32.
    std::vector<float> fp32_out(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        fp32_out[i] = half_to_float(fp16_buf[i]);
    }

    return fp32_out;
}
