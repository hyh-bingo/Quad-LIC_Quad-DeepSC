from typing import Any, Optional, Tuple
import math
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class EntropyCoder:

    def __init__(self, ec_thread=False, stream_part=1, method='rans_cpp'):
        super().__init__()
        self.name = method
        try:
            from MLCodec_rans import RansEncoder, RansDecoder
        except ImportError as error:
            raise RuntimeError('Entropy backend is missing; run bash scripts/setup_entropy.sh') from error
        self.encoder = RansEncoder(ec_thread, stream_part)
        self.decoder = RansDecoder(stream_part)

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from MLCodec_CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
        cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[:pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, :_cdf.size(0)] = _cdf
        return cdf

    def reset(self):
        self.encoder.reset()

    def flush(self):
        self.encoder.flush()

    def encode_with_indexes(self, symbols, indexes, cdf, cdf_length, offset):
        if not torch.isfinite(symbols).all() or torch.any(symbols < -30000) or torch.any(symbols > 30000):
            raise ValueError('Entropy symbols must be finite integers in [-30000, 30000]')
        self.encoder.encode_with_indexes(symbols.to(torch.int16).cpu().numpy(), indexes.to(torch.int16).cpu().numpy(), cdf, cdf_length, offset)

    def get_encoded_stream(self):
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream):
        self.decoder.set_stream(np.frombuffer(stream, dtype=np.uint8))

    def decode_stream(self, indexes, cdf, cdf_length, offset):
        rv = self.decoder.decode_stream(indexes.to(torch.int16).cpu().numpy(), cdf, cdf_length, offset)
        rv = torch.Tensor(rv)
        return rv

class EntropyModel(nn.Module):

    def __init__(self, entropy_coder_precision=16):
        super().__init__()
        self.entropy_coder = None
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None
        self.entropy_coder_precision = int(entropy_coder_precision)

    def set_entropy_coder(self, coder):
        self.entropy_coder = coder

    def set_cdf_info(self, quantized_cdf, cdf_length, offset):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()
        self._offset = offset.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return (self._quantized_cdf, self._cdf_length, self._offset)

class EntropyBottleneck(EntropyModel):
    _offset: Tensor

    def __init__(self, channels: int, *args: Any, tail_mass: float=1e-09, init_scale: float=10, filters: Tuple[int, ...]=(3, 3, 3, 3), **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.channels = int(channels)
        self.filters = tuple((int(f) for f in filters))
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f'_matrix{i:d}', nn.Parameter(matrix))
            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f'_bias{i:d}', nn.Parameter(bias))
            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f'_factor{i:d}', nn.Parameter(factor))
        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

    def _get_medians(self) -> Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force=False, entropy_coder=None):
        if entropy_coder is not None:
            self.entropy_coder = entropy_coder
        if self._offset is not None and (not force):
            return False
        with torch.no_grad():
            device = next(self.parameters()).device
            medians = self.quantiles[:, 0, 1]
            minima = medians - self.quantiles[:, 0, 0]
            minima = torch.ceil(minima).int()
            minima = torch.clamp(minima, min=0)
            maxima = self.quantiles[:, 0, 2] - medians
            maxima = torch.ceil(maxima).int()
            maxima = torch.clamp(maxima, min=0)
            offset = -minima
            pmf_start = medians - minima
            pmf_length = maxima + minima + 1
            max_length = pmf_length.max().item()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device)
            samples = samples[None, :] + pmf_start[:, None, None]
            half = float(0.5)
            lower = self._logits_cumulative(samples - half)
            upper = self._logits_cumulative(samples + half)
            sign = -torch.sign(lower + upper)
            pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
            pmf = pmf[:, 0, :]
            tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])
            quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.set_cdf_info(quantized_cdf, cdf_length, offset)
            return True

    def _logits_cumulative(self, inputs: Tensor) -> Tensor:
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f'_matrix{i:d}')
            logits = torch.matmul(F.softplus(matrix), logits)
            bias = getattr(self, f'_bias{i:d}')
            logits += bias
            if i < len(self.filters):
                factor = getattr(self, f'_factor{i:d}')
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @staticmethod
    def build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C, dtype=torch.int).view(1, -1, 1, 1)
        return indexes.repeat(N, 1, H, W)

    def encode(self, x):
        indexes = self.build_indexes(x.size())
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1), *self.get_cdf_info())

    def decode_stream(self, size, dtype, device):
        output_size = (1, self.channels, size[0], size[1])
        indexes = self.build_indexes(output_size)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1), *self.get_cdf_info())
        val = val.reshape(indexes.shape)
        return val.to(dtype).to(device)

class GaussianConditional(EntropyModel):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.tail_mass = float(1e-09)
        self.scale_min = 0.11
        self.scale_max = 64.0
        self.scale_level = 256
        self.scale_bound = self.scale_min
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)
        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)

    @staticmethod
    def get_scale_table(min_val, max_val, levels):
        return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-2 ** (-0.5))
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update(self, force=False, entropy_coder=None):
        if entropy_coder is not None:
            self.entropy_coder = entropy_coder
        if not force and self._offset is not None:
            return
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()
        device = pmf_center.device
        samples = torch.abs(torch.arange(max_length, device=device).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower
        tail_mass = 2 * lower[:, :1]
        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        offset = -pmf_center
        cdf_length = pmf_length + 2
        self.set_cdf_info(quantized_cdf, cdf_length, offset)

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-05)
        indexes = (torch.log(scales) - self.log_scale_min) / self.log_scale_step
        indexes = indexes.clamp_(0, self.scale_level - 1)
        return indexes.int()

    def encode(self, x, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1), *self.get_cdf_info())

    def decode_stream(self, scales, dtype, device):
        indexes = self.build_indexes(scales)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1), *self.get_cdf_info())
        val = val.reshape(scales.shape)
        return val.to(device).to(dtype)
