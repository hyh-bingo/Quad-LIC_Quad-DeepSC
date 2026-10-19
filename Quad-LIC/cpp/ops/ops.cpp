

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <stdexcept>

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float> &pmf,
                                           int precision) {
  

  if (precision < 1 || precision > 16 || pmf.empty() || pmf.size() > (1u << precision))
    throw std::invalid_argument("Invalid PMF size or precision");
  for (float value : pmf)
    if (!std::isfinite(value) || value < 0 || value > 1)
      throw std::invalid_argument("PMF values must be finite probabilities");
  std::vector<uint32_t> cdf(pmf.size() + 1);
  cdf[0] = 0; 

  std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1, [=](float p) {
    return static_cast<uint32_t>(std::round(p * (1 << precision)) + 0.5);
  });

  const uint64_t total = std::accumulate(cdf.begin(), cdf.end(), uint64_t{0});
  if (total == 0) throw std::invalid_argument("PMF has zero quantized probability mass");

  std::transform(
      cdf.begin(), cdf.end(), cdf.begin(), [precision, total](uint32_t p) {
        return static_cast<uint32_t>((((1ull << precision) * p) / total));
      });

  std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
  cdf.back() = 1 << precision;

  for (int i = 0; i < static_cast<int>(cdf.size() - 1); ++i) {
    if (cdf[i] == cdf[i + 1]) {
      
      uint32_t best_freq = ~0u;
      int best_steal = -1;
      for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
        uint32_t freq = cdf[j + 1] - cdf[j];
        if (freq > 1 && freq < best_freq) {
          best_freq = freq;
          best_steal = j;
        }
      }

      assert(best_steal != -1);

      if (best_steal < i) {
        for (int j = best_steal + 1; j <= i; ++j) {
          cdf[j]--;
        }
      } else {
        assert(best_steal > i);
        for (int j = i + 1; j <= best_steal; ++j) {
          cdf[j]++;
        }
      }
    }
  }

  assert(cdf[0] == 0);
  assert(cdf.back() == (1u << precision));
  for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
    assert(cdf[i + 1] > cdf[i]);
  }

  return cdf;
}

PYBIND11_MODULE(MLCodec_CXX, m) {
  m.attr("__name__") = "MLCodec_CXX";


  m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf);
}
