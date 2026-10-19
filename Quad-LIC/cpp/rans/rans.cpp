



#include "rans.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <limits>


constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4; 
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

namespace {


inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));

  
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    Rans64Assert(x < x_max);
  }

  
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits, const uint32_t *end) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  
  x = x >> n_bits;
  if (x < RANS64_L) {
    if (*pptr >= end) throw std::invalid_argument("Truncated rANS stream");
    x = (x << 32) | **pptr;
    *pptr += 1;
    if (x < RANS64_L) throw std::invalid_argument("Invalid rANS state");
    Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}
} 

void RansEncoderLib::encode_with_indexes(
    const std::shared_ptr<std::vector<int16_t>> symbols,
    const std::shared_ptr<std::vector<int16_t>> indexes,
    const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
    const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
    const std::shared_ptr<std::vector<int32_t>> offsets) {

  
  const int16_t *symbols_ptr = symbols->data();
  const int16_t *indexes_ptr = indexes->data();
  const int32_t *cdfs_sizes_ptr = cdfs_sizes->data();
  const int32_t *offsets_ptr = offsets->data();
  const int symbol_size = static_cast<int>(symbols->size());
  _syms.reserve(symbol_size * 3 / 2);
  for (int i = 0; i < symbol_size; ++i) {
    const int32_t cdf_idx = indexes_ptr[i];
    if (cdf_idx < 0) {
      continue;
    }
    const int32_t *cdf = cdfs->at(cdf_idx).data();
    const int32_t max_value = cdfs_sizes_ptr[cdf_idx] - 2;
    int32_t value = symbols_ptr[i] - offsets_ptr[cdf_idx];

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    _syms.push_back({static_cast<uint16_t>(cdf[value]),
                     static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                     false});

    
    if (value == max_value) {
      
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
      }

      
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val1 =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back({static_cast<uint16_t>(val1),
                         static_cast<uint16_t>(val1 + 1), true});
      }
    }
  }
}

void RansEncoderLib::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  std::vector<uint32_t> output(_syms.size() + 2); 
  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  const int nbytes = static_cast<int>(
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t));

  _stream.resize(nbytes);
  memcpy(_stream.data(), ptr, nbytes);
}

std::vector<uint8_t> RansEncoderLib::get_encoded_stream() { return _stream; }

void RansEncoderLib::reset() { _syms.clear(); _stream.clear(); }

void RansDecoderLib::set_stream(
    const std::shared_ptr<std::vector<uint8_t>> encoded) {
  if (encoded->size() < 8 || encoded->size() % 4 != 0)
    throw std::invalid_argument("Invalid rANS stream length");
  _stream = encoded;
  _ptr = reinterpret_cast<uint32_t *>(_stream->data());
  _end = _ptr + _stream->size() / 4;
  Rans64DecInit(&_rans, &_ptr);
  if (_rans < RANS64_L) throw std::invalid_argument("Invalid initial rANS state");
}

std::vector<int16_t> RansDecoderLib::decode_stream(
    const std::shared_ptr<std::vector<int16_t>> indexes,
    const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
    const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
    const std::shared_ptr<std::vector<int32_t>> offsets) {
  if (!_stream) throw std::invalid_argument("Call set_stream before decoding");
  int index_size = static_cast<int>(indexes->size());
  std::vector<int16_t> output(index_size);

  int16_t *outout_ptr = output.data();
  const int16_t *indexes_ptr = indexes->data();
  const int32_t *cdfs_sizes_ptr = cdfs_sizes->data();
  const int32_t *offsets_ptr = offsets->data();
  for (int i = 0; i < index_size; ++i) {
    const int32_t cdf_idx = indexes_ptr[i];
    if (cdf_idx < 0 || static_cast<size_t>(cdf_idx) >= cdfs->size())
      throw std::invalid_argument("Invalid CDF index");
    const int32_t offset = offsets_ptr[cdf_idx];
    const int32_t *cdf = cdfs->at(cdf_idx).data();
    const int32_t max_value = cdfs_sizes_ptr[cdf_idx] - 2;
    const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

    const auto cdf_end = cdf + cdfs_sizes_ptr[cdf_idx];
    const auto it = std::find_if(cdf, cdf_end, [cum_freq](int v) {
      return static_cast<uint32_t>(v) > cum_freq;
    });
    const uint32_t s = static_cast<uint32_t>(std::distance(cdf, it) - 1);

    uint64_t next = static_cast<uint64_t>(cdf[s + 1] - cdf[s]) * (_rans >> precision) + (_rans & 65535) - cdf[s];
    if (next < RANS64_L) {
      if (_ptr >= _end) throw std::invalid_argument("Truncated rANS stream");
      next = (next << 32) | *_ptr++;
      if (next < RANS64_L) throw std::invalid_argument("Invalid rANS state");
    }
    _rans = next;

    int64_t value = static_cast<int64_t>(s);

    if (value == max_value) {
      
      int32_t val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision, _end);
      int32_t n_bypass = val;

      if (n_bypass > 8) throw std::invalid_argument("Invalid escape length");
      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision, _end);
        n_bypass += val;
        if (n_bypass > 8) throw std::invalid_argument("Invalid escape length");
      }

      uint32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision, _end);
        raw_val |= static_cast<uint32_t>(val) << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    const int64_t decoded = static_cast<int64_t>(value) + offset;
    if (decoded < std::numeric_limits<int16_t>::min() || decoded > std::numeric_limits<int16_t>::max())
      throw std::invalid_argument("Decoded symbol is outside int16 range");
    outout_ptr[i] = static_cast<int16_t>(decoded);
  }
  return output;
}
