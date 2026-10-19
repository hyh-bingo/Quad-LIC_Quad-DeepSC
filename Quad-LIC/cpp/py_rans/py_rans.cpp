#include "py_rans.h"
#include <cstring>
#include <stdexcept>

namespace {
struct Tables {
  std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs;
  std::shared_ptr<std::vector<int32_t>> sizes;
  std::shared_ptr<std::vector<int32_t>> offsets;
};

Tables tables(const Int32Array &cdfs, const Int32Array &sizes, const Int32Array &offsets) {
  if (cdfs.ndim() != 2 || sizes.ndim() != 1 || offsets.ndim() != 1 || sizes.size() == 0 ||
      cdfs.shape(0) != sizes.size() || offsets.size() != sizes.size())
    throw std::invalid_argument("CDF arrays have incompatible shapes");
  Tables result;
  result.cdfs = std::make_shared<std::vector<std::vector<int32_t>>>();
  result.sizes = std::make_shared<std::vector<int32_t>>(sizes.data(), sizes.data() + sizes.size());
  result.offsets = std::make_shared<std::vector<int32_t>>(offsets.data(), offsets.data() + offsets.size());
  auto rows = cdfs.unchecked<2>();
  for (ssize_t i = 0; i < sizes.size(); ++i) {
    int length = sizes.data()[i];
    if (length < 3 || length > cdfs.shape(1) || rows(i, 0) != 0 || rows(i, length - 1) != 65536)
      throw std::invalid_argument("CDF must start at zero and end at 65536");
    for (int j = 1; j < length; ++j)
      if (rows(i, j) <= rows(i, j - 1)) throw std::invalid_argument("CDF must be strictly increasing");
    if (offsets.data()[i] < -65535 || offsets.data()[i] > 65535)
      throw std::invalid_argument("CDF offset exceeds the supported symbol range");
    result.cdfs->emplace_back(rows.data(i, 0), rows.data(i, 0) + length);
  }
  return result;
}

std::shared_ptr<std::vector<int16_t>> index_vector(const Int16Array &indexes, size_t count) {
  if (indexes.ndim() != 1) throw std::invalid_argument("Indexes must be one-dimensional");
  auto result = std::make_shared<std::vector<int16_t>>(indexes.data(), indexes.data() + indexes.size());
  for (int16_t index : *result)
    if (index < 0 || static_cast<size_t>(index) >= count) throw std::invalid_argument("Invalid CDF index");
  return result;
}
}

RansEncoder::RansEncoder(bool multiThread, int streamPart) {
  if (multiThread || streamPart != 1)
    throw std::invalid_argument("This inference backend supports one synchronous stream");
}

void RansEncoder::encode_with_indexes(const Int16Array &symbols, const Int16Array &indexes,
                                     const Int32Array &cdfs, const Int32Array &sizes,
                                     const Int32Array &offsets) {
  if (symbols.ndim() != 1 || symbols.size() != indexes.size())
    throw std::invalid_argument("Symbols and indexes must be matching one-dimensional arrays");
  auto c = tables(cdfs, sizes, offsets);
  auto indexes_copy = index_vector(indexes, c.sizes->size());
  auto symbols_copy = std::make_shared<std::vector<int16_t>>(symbols.data(), symbols.data() + symbols.size());
  encoder.encode_with_indexes(symbols_copy, indexes_copy, c.cdfs, c.sizes, c.offsets);
}

void RansEncoder::flush() { encoder.flush(); }
void RansEncoder::reset() { encoder.reset(); }

ByteArray RansEncoder::get_encoded_stream() {
  auto bytes = encoder.get_encoded_stream();
  ByteArray output(bytes.size() + 1);
  output.mutable_data()[0] = 1;
  if (!bytes.empty()) std::memcpy(output.mutable_data() + 1, bytes.data(), bytes.size());
  return output;
}

RansDecoder::RansDecoder(int streamPart) {
  if (streamPart != 1) throw std::invalid_argument("This inference backend supports one stream");
}

void RansDecoder::set_stream(const ByteArray &encoded) {
  if (encoded.ndim() != 1 || encoded.size() < 9 || encoded.data()[0] != 1)
    throw std::invalid_argument("Invalid single-stream rANS header");
  auto bytes = std::make_shared<std::vector<uint8_t>>(encoded.data() + 1, encoded.data() + encoded.size());
  decoder.set_stream(bytes);
}

Int16Array RansDecoder::decode_stream(const Int16Array &indexes, const Int32Array &cdfs,
                                     const Int32Array &sizes, const Int32Array &offsets) {
  auto c = tables(cdfs, sizes, offsets);
  auto values = decoder.decode_stream(index_vector(indexes, c.sizes->size()), c.cdfs, c.sizes, c.offsets);
  Int16Array output(values.size());
  if (!values.empty()) std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(int16_t));
  return output;
}

PYBIND11_MODULE(MLCodec_rans, m) {
  py::class_<RansEncoder>(m, "RansEncoder")
      .def(py::init<bool, int>())
      .def("encode_with_indexes", &RansEncoder::encode_with_indexes)
      .def("flush", &RansEncoder::flush)
      .def("get_encoded_stream", &RansEncoder::get_encoded_stream)
      .def("reset", &RansEncoder::reset);
  py::class_<RansDecoder>(m, "RansDecoder")
      .def(py::init<int>())
      .def("set_stream", &RansDecoder::set_stream)
      .def("decode_stream", &RansDecoder::decode_stream);
}
