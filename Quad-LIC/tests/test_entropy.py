import numpy as np
import pytest
from MLCodec_rans import RansEncoder, RansDecoder
from MLCodec_CXX import pmf_to_quantized_cdf


def test_native_rans_symbols_and_escape_roundtrip():
    cdf = np.array([pmf_to_quantized_cdf([0.2, 0.3, 0.4, 0.1], 16)], dtype=np.int32)
    lengths = np.array([5], dtype=np.int32)
    offsets = np.array([-1], dtype=np.int32)
    symbols = np.array([-30000, -17, -1, 0, 1, 2, 19, 30000], dtype=np.int16)
    indexes = np.zeros(symbols.size, dtype=np.int16)
    encoder = RansEncoder(False, 1)
    encoder.reset()
    encoder.encode_with_indexes(symbols, indexes, cdf, lengths, offsets)
    encoder.flush()
    stream = encoder.get_encoded_stream()
    assert stream.nbytes > 0
    decoder = RansDecoder(1)
    decoder.set_stream(stream)
    decoded = np.array(decoder.decode_stream(indexes, cdf, lengths, offsets))
    np.testing.assert_array_equal(decoded, symbols)


@pytest.mark.parametrize('symbols', [[], [0], [-30000], [30000], [-30000, 30000] * 2048])
def test_minimal_and_escape_stress(symbols):
    cdf = np.array([pmf_to_quantized_cdf([0.2, 0.3, 0.4, 0.1], 16)], dtype=np.int32)
    sizes = np.array([5], dtype=np.int32)
    offsets = np.array([-1], dtype=np.int32)
    values = np.array(symbols, dtype=np.int16)
    indexes = np.zeros(values.size, dtype=np.int16)
    encoder = RansEncoder(False, 1)
    encoder.reset()
    encoder.encode_with_indexes(values, indexes, cdf, sizes, offsets)
    encoder.flush()
    stream = encoder.get_encoded_stream()
    decoder = RansDecoder(1)
    decoder.set_stream(stream)
    np.testing.assert_array_equal(decoder.decode_stream(indexes, cdf, sizes, offsets), values)


def test_reject_truncation_invalid_cdf_and_indexes():
    cdf = np.array([pmf_to_quantized_cdf([0.2, 0.3, 0.4, 0.1], 16)], dtype=np.int32)
    sizes = np.array([5], dtype=np.int32)
    offsets = np.array([-1], dtype=np.int32)
    symbols = np.array([-30000, 30000] * 512, dtype=np.int16)
    indexes = np.zeros(symbols.size, dtype=np.int16)
    encoder = RansEncoder(False, 1)
    encoder.encode_with_indexes(symbols, indexes, cdf, sizes, offsets)
    encoder.flush()
    stream = encoder.get_encoded_stream()
    for cut in [0, 1, 8, 9, len(stream) - 1, len(stream) - 4]:
        with pytest.raises(ValueError):
            decoder = RansDecoder(1)
            decoder.set_stream(stream[:cut])
            decoder.decode_stream(indexes, cdf, sizes, offsets)
    for bad in [-1, 1]:
        with pytest.raises(ValueError):
            encoder.encode_with_indexes(np.array([0], dtype=np.int16), np.array([bad], dtype=np.int16), cdf, sizes, offsets)
    broken = cdf.copy()
    broken[0, 1] = 0
    with pytest.raises(ValueError):
        encoder.encode_with_indexes(symbols, indexes, broken, sizes, offsets)
    with pytest.raises(ValueError):
        RansEncoder(True, 1)
    with pytest.raises(ValueError):
        RansDecoder(2)


@pytest.mark.parametrize('pmf,precision', [([], 16), ([0, 0], 16), ([-0.1, 1.1], 16), ([float('nan'), 1], 16), ([0.5, 0.5], 0)])
def test_invalid_pmf_rejected(pmf, precision):
    with pytest.raises(ValueError):
        pmf_to_quantized_cdf(pmf, precision)
