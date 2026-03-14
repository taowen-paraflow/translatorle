#ifndef NPY_READER_H
#define NPY_READER_H

#include <string>
#include <vector>
#include <cstdint>

// Load a .npy file containing FP16 ('<f2') data and return it as FP32.
std::vector<float> load_npy_fp16_as_fp32(const std::string& path,
                                         std::vector<size_t>& shape_out);

// Load a .npy file containing INT8 ('|i1') data.
std::vector<int8_t> load_npy_int8(const std::string& path,
                                   std::vector<size_t>& shape_out);

#endif // NPY_READER_H
