#ifndef NPY_READER_H
#define NPY_READER_H

#include <string>
#include <vector>
#include <cstddef>

// Load a .npy file containing FP16 ('<f2') data and return it as FP32.
// Populates shape_out with the array dimensions (e.g., {248320, 1024}).
// Throws std::runtime_error on any format or I/O error.
std::vector<float> load_npy_fp16_as_fp32(const std::string& path,
                                         std::vector<size_t>& shape_out);

// Load a .npy file containing FP32 ('<f4') data.
// Populates shape_out with the array dimensions (e.g., {1024}).
// Throws std::runtime_error on any format or I/O error.
std::vector<float> load_npy_fp32(const std::string& path,
                                 std::vector<size_t>& shape_out);

#endif // NPY_READER_H
