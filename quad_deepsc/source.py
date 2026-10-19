import torch
import torch.nn as nn
from .layers import conv3x3, AttentionBlock, DepthConvBlock, ResidualBottleneckBlock, Quantizer, UNet, conv, deconv
import math

class QSC_for_JSCC(nn.Module):

    def __init__(self, N=192, M=320, inplace=False, **kwargs):
        super().__init__()
        self.N = int(N)
        self.M = int(M)
        self.masks = {}
        self.g_a = nn.Sequential(conv(3, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), conv(N, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), AttentionBlock(N), conv(N, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), conv(N, M), AttentionBlock(M))
        self.g_s = nn.Sequential(AttentionBlock(M), deconv(M, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), deconv(N, N), AttentionBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), deconv(N, N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), ResidualBottleneckBlock(N), deconv(N, 16))
        self.h_a = nn.Sequential(conv3x3(M, N), nn.ReLU(inplace=True), conv(N, N), nn.ReLU(inplace=True), conv(N, N))
        self.h_s = nn.Sequential(deconv(N, N), nn.ReLU(inplace=True), deconv(N, N * 3 // 2), nn.ReLU(inplace=True), conv3x3(N * 3 // 2, 2 * M))
        self.y_spatial_prior_adaptor_1 = nn.Conv2d(M * 3, M * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(M * 3, M * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(M * 3, M * 3, 1)
        self.y_spatial_prior = nn.Sequential(DepthConvBlock(M * 3, M * 2, inplace=inplace), DepthConvBlock(M * 2, M * 2, inplace=inplace))
        self.refine = nn.Sequential(UNet(16, 16, inplace=inplace), conv3x3(16, 3))
        self.quantizer = Quantizer()

    def get_mask_four_parts(self, height, width, dtype, device):
        curr_mask_str = f'{width}x{height}'
        if curr_mask_str not in self.masks:
            micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
            mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)
            micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
            mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
            mask_1 = mask_1[:height, :width]
            mask_1 = torch.unsqueeze(mask_1, 0)
            mask_1 = torch.unsqueeze(mask_1, 0)
            micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
            mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
            mask_2 = mask_2[:height, :width]
            mask_2 = torch.unsqueeze(mask_2, 0)
            mask_2 = torch.unsqueeze(mask_2, 0)
            micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
            mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
            mask_3 = mask_3[:height, :width]
            mask_3 = torch.unsqueeze(mask_3, 0)
            mask_3 = torch.unsqueeze(mask_3, 0)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def jscc_encoder(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat = self.quantizer.quantize(z, 'noise')
        latent_scales, latent_means = self.h_s(z_hat).chunk(2, 1)
        common_params = torch.cat((latent_scales, latent_means), dim=1)
        dtype = y.dtype
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)
        y_0, y_1, y_2, y_3 = y.chunk(4, 1)
        y_hat = self.quantizer.quantize(y, 'noise')
        y_hat_0, y_hat_1, y_hat_2, y_hat_3 = y_hat.chunk(4, 1)
        latent_scales_0, latent_scales_1, latent_scales_2, latent_scales_3 = latent_scales.chunk(4, 1)
        latent_means_0, latent_means_1, latent_means_2, latent_means_3 = latent_means.chunk(4, 1)
        y_hat_curr_step = torch.cat((y_hat_0 * mask_0, y_hat_1 * mask_1, y_hat_2 * mask_2, y_hat_3 * mask_3), dim=1)
        y_jscc_0 = y_0 * mask_0 + y_1 * mask_1 + y_2 * mask_2 + y_3 * mask_3
        scales_0 = latent_scales_0 * mask_0 + latent_scales_1 * mask_1 + latent_scales_2 * mask_2 + latent_scales_3 * mask_3
        means_0 = latent_means_0 * mask_0 + latent_means_1 * mask_1 + latent_means_2 * mask_2 + latent_means_3 * mask_3
        y_hat_so_far = y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        latent_scales_0, latent_scales_1, latent_scales_2, latent_scales_3, latent_means_0, latent_means_1, latent_means_2, latent_means_3 = self.y_spatial_prior(self.y_spatial_prior_adaptor_1(params)).chunk(8, 1)
        y_hat_curr_step = torch.cat((y_hat_0 * mask_3, y_hat_1 * mask_2, y_hat_2 * mask_1, y_hat_3 * mask_0), dim=1)
        y_jscc_1 = y_0 * mask_3 + y_1 * mask_2 + y_2 * mask_1 + y_3 * mask_0
        scales_1 = latent_scales_0 * mask_3 + latent_scales_1 * mask_2 + latent_scales_2 * mask_1 + latent_scales_3 * mask_0
        means_1 = latent_means_0 * mask_3 + latent_means_1 * mask_2 + latent_means_2 * mask_1 + latent_means_3 * mask_0
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        latent_scales_0, latent_scales_1, latent_scales_2, latent_scales_3, latent_means_0, latent_means_1, latent_means_2, latent_means_3 = self.y_spatial_prior(self.y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        y_hat_curr_step = torch.cat((y_hat_0 * mask_2, y_hat_1 * mask_3, y_hat_2 * mask_0, y_hat_3 * mask_1), dim=1)
        y_jscc_2 = y_0 * mask_2 + y_1 * mask_3 + y_2 * mask_0 + y_3 * mask_1
        scales_2 = latent_scales_0 * mask_2 + latent_scales_1 * mask_3 + latent_scales_2 * mask_0 + latent_scales_3 * mask_1
        means_2 = latent_means_0 * mask_2 + latent_means_1 * mask_3 + latent_means_2 * mask_0 + latent_means_3 * mask_1
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        latent_scales_0, latent_scales_1, latent_scales_2, latent_scales_3, latent_means_0, latent_means_1, latent_means_2, latent_means_3 = self.y_spatial_prior(self.y_spatial_prior_adaptor_3(params)).chunk(8, 1)
        y_jscc_3 = y_0 * mask_1 + y_1 * mask_0 + y_2 * mask_3 + y_3 * mask_2
        scales_3 = latent_scales_0 * mask_1 + latent_scales_1 * mask_0 + latent_scales_2 * mask_3 + latent_scales_3 * mask_2
        means_3 = latent_means_0 * mask_1 + latent_means_1 * mask_0 + latent_means_2 * mask_3 + latent_means_3 * mask_2
        y_quadtree_likelihoods = []
        y_quadtree_likelihood = gaussian_likelihood(y_jscc_0, scales_0, means_0)
        y_quadtree_likelihoods.append(y_quadtree_likelihood)
        y_quadtree_likelihood = gaussian_likelihood(y_jscc_1, scales_1, means_1)
        y_quadtree_likelihoods.append(y_quadtree_likelihood)
        y_quadtree_likelihood = gaussian_likelihood(y_jscc_2, scales_2, means_2)
        y_quadtree_likelihoods.append(y_quadtree_likelihood)
        y_quadtree_likelihood = gaussian_likelihood(y_jscc_3, scales_3, means_3)
        y_quadtree_likelihoods.append(y_quadtree_likelihood)
        y_likelihoods = torch.cat(y_quadtree_likelihoods, dim=1)
        return {'y_jscc': torch.cat([y_jscc_0, y_jscc_1, y_jscc_2, y_jscc_3], dim=0), 'likelihoods': {'y': y_likelihoods}}

    def jscc_decoder(self, y_hat):
        x_hat = self.g_s(y_hat)
        x_hat = self.refine(x_hat)
        return {'x_hat': x_hat}

def gaussian_likelihood(inputs, scales, means):
    values = inputs + torch.empty_like(inputs).uniform_(-0.5, 0.5)
    values = (values - means).abs()
    scales = scales.clamp_min(0.11)
    upper = 0.5 * torch.erfc(-float(2 ** -0.5) * ((0.5 - values) / scales))
    lower = 0.5 * torch.erfc(-float(2 ** -0.5) * ((-0.5 - values) / scales))
    return (upper - lower).clamp_min(1e-09)
