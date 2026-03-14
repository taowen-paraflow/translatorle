#include "npy_reader.h"
#include "half.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sstream>

// ---------------------------------------------------------------------------
// NPY header parsing utilities
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

static std::string extract_string_value(const std::string& header,
                                        const std::string& key) {
    std::string search = "'" + key + "'";
    auto pos = header.find(search);
    if (pos == std::string::npos)
        throw std::runtime_error("npy header: missing key '" + key + "'");
    pos = header.find('\'', pos + search.size());
    if (pos == std::string::npos)
        throw std::runtime_error("npy header: malformed value for '" + key + "'");
    auto end = header.find('\'', pos + 1);
    if (end == std::string::npos)
        throw std::runtime_error("npy header: unterminated string for '" + key + "'");
    return header.substr(pos + 1, end - pos - 1);
}

static std::vector<size_t> extract_shape(const std::string& header) {
    auto pos = header.find("'shape'");
    if (pos == std::string::npos)
        throw std::runtime_error("npy header: missing 'shape'");
    auto open = header.find('(', pos);
    auto close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos)
        throw std::runtime_error("npy header: malformed 'shape'");
    std::string inside = header.substr(open + 1, close - open - 1);

    std::vector<size_t> shape;
    std::istringstream iss(inside);
    std::string token;
    while (std::getline(iss, token, ',')) {
        size_t start = token.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        size_t end = token.find_last_not_of(" \t");
        std::string num = token.substr(start, end - start + 1);
        if (num.empty()) continue;
        shape.push_back(static_cast<size_t>(std::stoull(num)));
    }
    return shape;
}

// Read npy header and return (header_string, data_offset)
static std::string read_npy_header(std::ifstream& file, const std::string& path) {
    char magic[6];
    file.read(magic, 6);
    if (!file || magic[0] != '\x93' || std::strncmp(magic + 1, "NUMPY", 5) != 0)
        throw std::runtime_error("Not a valid .npy file: " + path);

    uint8_t major = 0, minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    if (!file)
        throw std::runtime_error("Failed to read npy version: " + path);

    uint32_t header_len = 0;
    if (major == 1) {
        char buf[2];
        file.read(buf, 2);
        if (!file) throw std::runtime_error("Failed to read v1 header: " + path);
        header_len = read_le16(buf);
    } else if (major == 2) {
        char buf[4];
        file.read(buf, 4);
        if (!file) throw std::runtime_error("Failed to read v2 header: " + path);
        header_len = read_le32(buf);
    } else {
        throw std::runtime_error("Unsupported npy version: " + path);
    }

    std::string header(header_len, '\0');
    file.read(&header[0], header_len);
    if (!file)
        throw std::runtime_error("Failed to read npy header: " + path);

    return header;
}

// ---------------------------------------------------------------------------
// FP16 -> FP32 loader
// ---------------------------------------------------------------------------

std::vector<float> load_npy_fp16_as_fp32(const std::string& path,
                                         std::vector<size_t>& shape_out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open: " + path);

    std::string header = read_npy_header(file, path);
    std::string descr = extract_string_value(header, "descr");
    if (descr != "<f2")
        throw std::runtime_error("Expected '<f2', got '" + descr + "': " + path);

    shape_out = extract_shape(header);

    size_t num_elements = 1;
    for (size_t d : shape_out) num_elements *= d;
    if (num_elements == 0) return {};

    std::vector<uint16_t> fp16_buf(num_elements);
    file.read(reinterpret_cast<char*>(fp16_buf.data()),
              static_cast<std::streamsize>(num_elements * sizeof(uint16_t)));
    if (!file)
        throw std::runtime_error("Failed to read data: " + path);

    std::vector<float> fp32_out(num_elements);
    for (size_t i = 0; i < num_elements; ++i)
        fp32_out[i] = half_to_float(fp16_buf[i]);

    return fp32_out;
}

// ---------------------------------------------------------------------------
// INT8 loader
// ---------------------------------------------------------------------------

std::vector<int8_t> load_npy_int8(const std::string& path,
                                   std::vector<size_t>& shape_out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open: " + path);

    std::string header = read_npy_header(file, path);
    std::string descr = extract_string_value(header, "descr");
    if (descr != "|i1")
        throw std::runtime_error("Expected '|i1', got '" + descr + "': " + path);

    shape_out = extract_shape(header);

    size_t num_elements = 1;
    for (size_t d : shape_out) num_elements *= d;
    if (num_elements == 0) return {};

    std::vector<int8_t> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(num_elements));
    if (!file)
        throw std::runtime_error("Failed to read data: " + path);

    return data;
}
