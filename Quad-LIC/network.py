import torch
import torch.nn as nn
from common_model import CompressionModel
from layers import DepthConvBlock, ResidualBottleneckBlock, UNet, conv, deconv, conv3x3, AttentionBlock

class QSC(CompressionModel):

    def __init__(self, N=192, M=320, inplace=False, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = int(N)
        self.M = int(M)
        self.g_a = nn.Sequential(conv(3, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), conv(N, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), AttentionBlock(N), conv(N, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), conv(N, M), AttentionBlock(M))
        self.g_s = nn.Sequential(AttentionBlock(M), deconv(M, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), deconv(N, N), AttentionBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), deconv(N, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), deconv(N, 16))
        self.h_a = nn.Sequential(conv3x3(M, N), nn.ReLU(inplace=True), conv(N, N), nn.ReLU(inplace=True), conv(N, N))
        self.h_s = nn.Sequential(deconv(N, N), nn.ReLU(inplace=True), deconv(N, N * 3 // 2), nn.ReLU(inplace=True), conv3x3(N * 3 // 2, 2 * M))
        self.y_spatial_prior_adaptor_1 = nn.Conv2d(M * 3, M * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(M * 3, M * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(M * 3, M * 3, 1)
        self.y_spatial_prior = nn.Sequential(DepthConvBlock(M * 3, M * 2, inplace=inplace), DepthConvBlock(M * 2, M * 2, inplace=inplace))
        self.refine = nn.Sequential(UNet(16, 16, inplace=inplace), conv3x3(16, 3))
        self.entropy_coder = None

    @torch.inference_mode()
    def compress(self, x):
        if self.entropy_coder is None:
            raise RuntimeError("Call model.update(force=True) before coding")
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3 or x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("compress expects a single RGB image padded to multiples of 16")
        y = self.g_a(x)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.h_a(y_pad)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = torch.round(z - z_offset)
        z_hat = z_tmp + z_offset
        params = self.h_s(z_hat)
        params = self.slice_to_y(params, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2, self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        self.entropy_coder.reset()
        self.entropy_bottleneck.encode(z_tmp)
        self.gaussian_conditional.encode(y_q_w_0, scales_w_0)
        self.gaussian_conditional.encode(y_q_w_1, scales_w_1)
        self.gaussian_conditional.encode(y_q_w_2, scales_w_2)
        self.gaussian_conditional.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()
        bit_stream = self.entropy_coder.get_encoded_stream()
        return {'bit_stream': bit_stream}

    @torch.inference_mode()
    def decompress(self, bit_stream, height, width):
        if self.entropy_coder is None:
            raise RuntimeError("Call model.update(force=True) before coding")
        if min(height, width) <= 0 or not bit_stream:
            raise ValueError("Invalid image dimensions or empty bitstream")
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(height, width, 64)
        y_height, y_width = self.get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        z_tmp = self.entropy_bottleneck.decode_stream(z_size, dtype, device)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = z_tmp + z_offset
        params = self.h_s(z_hat)
        params = self.slice_to_y(params, slice_shape)
        y_hat = self.decompress_four_part_prior(params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2, self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        x_hat = self.g_s(y_hat)
        x_hat = self.refine(x_hat).clamp_(0, 1)
        return {'x_hat': x_hat}
