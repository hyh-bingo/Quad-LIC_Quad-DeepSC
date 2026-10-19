

#pragma once

#include <memory>
#include <vector>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#include <rans64.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass; 
};

class RansEncoderLib {
public:
  RansEncoderLib() = default;
  virtual ~RansEncoderLib() = default;

  RansEncoderLib(const RansEncoderLib &) = delete;
  RansEncoderLib(RansEncoderLib &&) = delete;
  RansEncoderLib &operator=(const RansEncoderLib &) = delete;
  RansEncoderLib &operator=(RansEncoderLib &&) = delete;

  virtual void encode_with_indexes(
      const std::shared_ptr<std::vector<int16_t>> symbols,
      const std::shared_ptr<std::vector<int16_t>> indexes,
      const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
      const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
      const std::shared_ptr<std::vector<int32_t>> offsets);
  virtual void flush();
  virtual std::vector<uint8_t> get_encoded_stream();
  virtual void reset();

private:
  std::vector<RansSymbol> _syms;
  std::vector<uint8_t> _stream;
};

class RansDecoderLib {
public:
  RansDecoderLib() = default;
  virtual ~RansDecoderLib() = default;

  RansDecoderLib(const RansDecoderLib &) = delete;
  RansDecoderLib(RansDecoderLib &&) = delete;
  RansDecoderLib &operator=(const RansDecoderLib &) = delete;
  RansDecoderLib &operator=(RansDecoderLib &&) = delete;

  void set_stream(const std::shared_ptr<std::vector<uint8_t>> encoded);

  virtual std::vector<int16_t>
  decode_stream(const std::shared_ptr<std::vector<int16_t>> indexes,
                const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
                const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
                const std::shared_ptr<std::vector<int32_t>> offsets);

private:
  Rans64State _rans;
  uint32_t *_ptr = nullptr;
  uint32_t *_end = nullptr;
  std::shared_ptr<std::vector<uint8_t>> _stream;
};
