#pragma once
#include "rans.h"
#include <memory>
#include <pybind11/numpy.h>

namespace py = pybind11;
using Int16Array = py::array_t<int16_t, py::array::c_style | py::array::forcecast>;
using Int32Array = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using ByteArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

class RansEncoder {
public:
  RansEncoder(bool multiThread, int streamPart);
  void encode_with_indexes(const Int16Array &symbols, const Int16Array &indexes,
                           const Int32Array &cdfs, const Int32Array &sizes,
                           const Int32Array &offsets);
  void flush();
  ByteArray get_encoded_stream();
  void reset();
private:
  RansEncoderLib encoder;
};

class RansDecoder {
public:
  explicit RansDecoder(int streamPart);
  void set_stream(const ByteArray &encoded);
  Int16Array decode_stream(const Int16Array &indexes, const Int32Array &cdfs,
                          const Int32Array &sizes, const Int32Array &offsets);
private:
  RansDecoderLib decoder;
};
