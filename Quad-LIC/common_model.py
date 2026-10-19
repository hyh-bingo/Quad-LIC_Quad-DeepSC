import math
import torch
import torch.nn as nn
from entropy_model import EntropyBottleneck, GaussianConditional, EntropyCoder

class CompressionModel(nn.Module):

    def __init__(self, entropy_bottleneck_channels, ec_thread=False, stream_part=1):
        super().__init__()
        self.z_channel = entropy_bottleneck_channels
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        self.gaussian_conditional = GaussianConditional()
        self.ec_thread = ec_thread
        self.stream_part = stream_part
        self.masks = {}

    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_conditional.update(force=force, entropy_coder=self.entropy_coder)
        self.entropy_bottleneck.update(force=force, entropy_coder=self.entropy_coder)

    def quant(self, x):
        return torch.round(x)

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = self.get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(y, (padding_l, padding_r, padding_t, padding_b), mode='replicate')
        return (y_pad, (-padding_l, -padding_r, -padding_t, -padding_b))

    def get_padding_size(self, height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_left = 0
        padding_right = new_w - width - padding_left
        padding_top = 0
        padding_bottom = new_h - height - padding_top
        return (padding_left, padding_right, padding_top, padding_bottom)

    def get_to_y_slice_shape(self, height, width):
        padding_l, padding_r, padding_t, padding_b = self.get_padding_size(height, width, 4)
        return (-padding_l, -padding_r, -padding_t, -padding_b)

    def slice_to_y(self, param, slice_shape):
        return torch.nn.functional.pad(param, slice_shape)

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask
        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat
        return (y_res, y_q, y_hat, scales_hat)

    def get_downsampled_shape(self, height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return (int(new_h / p + 0.5), int(new_w / p + 0.5))

    def get_mask_four_parts(self, height, width, dtype, device):
        curr_mask_str = (height, width, str(dtype), str(device))
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

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3, x_1_0, x_1_1, x_1_2, x_1_3, x_2_0, x_2_1, x_2_2, x_2_3, x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)

    def forward_four_part_prior(self, y, common_params, y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2, y_spatial_prior_adaptor_3, y_spatial_prior):
        scales, means = common_params.chunk(2, 1)
        dtype = y.dtype
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)
        y_0, y_1, y_2, y_3 = y.chunk(4, 1)
        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)
        y_res_0_0, y_q_0_0, y_hat_0_0, s_hat_0_0 = self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_res_1_1, y_q_1_1, y_hat_1_1, s_hat_1_1 = self.process_with_mask(y_1, scales_1, means_1, mask_1)
        y_res_2_2, y_q_2_2, y_hat_2_2, s_hat_2_2 = self.process_with_mask(y_2, scales_2, means_2, mask_2)
        y_res_3_3, y_q_3_3, y_hat_3_3, s_hat_3_3 = self.process_with_mask(y_3, scales_3, means_3, mask_3)
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)
        y_hat_so_far = y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)
        y_res_0_3, y_q_0_3, y_hat_0_3, s_hat_0_3 = self.process_with_mask(y_0, scales_0, means_0, mask_3)
        y_res_1_2, y_q_1_2, y_hat_1_2, s_hat_1_2 = self.process_with_mask(y_1, scales_1, means_1, mask_2)
        y_res_2_1, y_q_2_1, y_hat_2_1, s_hat_2_1 = self.process_with_mask(y_2, scales_2, means_2, mask_1)
        y_res_3_0, y_q_3_0, y_hat_3_0, s_hat_3_0 = self.process_with_mask(y_3, scales_3, means_3, mask_0)
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        y_res_0_2, y_q_0_2, y_hat_0_2, s_hat_0_2 = self.process_with_mask(y_0, scales_0, means_0, mask_2)
        y_res_1_3, y_q_1_3, y_hat_1_3, s_hat_1_3 = self.process_with_mask(y_1, scales_1, means_1, mask_3)
        y_res_2_0, y_q_2_0, y_hat_2_0, s_hat_2_0 = self.process_with_mask(y_2, scales_2, means_2, mask_0)
        y_res_3_1, y_q_3_1, y_hat_3_1, s_hat_3_1 = self.process_with_mask(y_3, scales_3, means_3, mask_1)
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)
        y_res_0_1, y_q_0_1, y_hat_0_1, s_hat_0_1 = self.process_with_mask(y_0, scales_0, means_0, mask_1)
        y_res_1_0, y_q_1_0, y_hat_1_0, s_hat_1_0 = self.process_with_mask(y_1, scales_1, means_1, mask_0)
        y_res_2_3, y_q_2_3, y_hat_2_3, s_hat_2_3 = self.process_with_mask(y_2, scales_2, means_2, mask_3)
        y_res_3_2, y_q_3_2, y_hat_3_2, s_hat_3_2 = self.process_with_mask(y_3, scales_3, means_3, mask_2)
        y_q_w_0 = y_q_0_0 + y_q_1_1 + y_q_2_2 + y_q_3_3
        y_q_w_1 = y_q_0_3 + y_q_1_2 + y_q_2_1 + y_q_3_0
        y_q_w_2 = y_q_0_2 + y_q_1_3 + y_q_2_0 + y_q_3_1
        y_q_w_3 = y_q_0_1 + y_q_1_0 + y_q_2_3 + y_q_3_2
        scales_w_0 = s_hat_0_0 + s_hat_1_1 + s_hat_2_2 + s_hat_3_3
        scales_w_1 = s_hat_0_3 + s_hat_1_2 + s_hat_2_1 + s_hat_3_0
        scales_w_2 = s_hat_0_2 + s_hat_1_3 + s_hat_2_0 + s_hat_3_1
        scales_w_3 = s_hat_0_1 + s_hat_1_0 + s_hat_2_3 + s_hat_3_2
        y_hat = self.combine_four_parts(y_hat_0_0, y_hat_0_1, y_hat_0_2, y_hat_0_3, y_hat_1_0, y_hat_1_1, y_hat_1_2, y_hat_1_3, y_hat_2_0, y_hat_2_1, y_hat_2_2, y_hat_2_3, y_hat_3_0, y_hat_3_1, y_hat_3_2, y_hat_3_3)
        return (y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat)

    def compress_four_part_prior(self, y, common_params, y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2, y_spatial_prior_adaptor_3, y_spatial_prior):
        return self.forward_four_part_prior(y, common_params, y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2, y_spatial_prior_adaptor_3, y_spatial_prior)

    def decompress_four_part_prior(self, common_params, y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2, y_spatial_prior_adaptor_3, y_spatial_prior):
        scales, means = common_params.chunk(2, 1)
        dtype = means.dtype
        device = means.device
        _, _, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)
        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)
        scales_r = scales_0 * mask_0 + scales_1 * mask_1 + scales_2 * mask_2 + scales_3 * mask_3
        y_q_r = self.gaussian_conditional.decode_stream(scales_r, dtype, device)
        y_hat_0_0 = (y_q_r + means_0) * mask_0
        y_hat_1_1 = (y_q_r + means_1) * mask_1
        y_hat_2_2 = (y_q_r + means_2) * mask_2
        y_hat_3_3 = (y_q_r + means_3) * mask_3
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)
        y_hat_so_far = y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)
        scales_r = scales_0 * mask_3 + scales_1 * mask_2 + scales_2 * mask_1 + scales_3 * mask_0
        y_q_r = self.gaussian_conditional.decode_stream(scales_r, dtype, device)
        y_hat_0_3 = (y_q_r + means_0) * mask_3
        y_hat_1_2 = (y_q_r + means_1) * mask_2
        y_hat_2_1 = (y_q_r + means_2) * mask_1
        y_hat_3_0 = (y_q_r + means_3) * mask_0
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        scales_r = scales_0 * mask_2 + scales_1 * mask_3 + scales_2 * mask_0 + scales_3 * mask_1
        y_q_r = self.gaussian_conditional.decode_stream(scales_r, dtype, device)
        y_hat_0_2 = (y_q_r + means_0) * mask_2
        y_hat_1_3 = (y_q_r + means_1) * mask_3
        y_hat_2_0 = (y_q_r + means_2) * mask_0
        y_hat_3_1 = (y_q_r + means_3) * mask_1
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)
        scales_r = scales_0 * mask_1 + scales_1 * mask_0 + scales_2 * mask_3 + scales_3 * mask_2
        y_q_r = self.gaussian_conditional.decode_stream(scales_r, dtype, device)
        y_hat_0_1 = (y_q_r + means_0) * mask_1
        y_hat_1_0 = (y_q_r + means_1) * mask_0
        y_hat_2_3 = (y_q_r + means_2) * mask_3
        y_hat_3_2 = (y_q_r + means_3) * mask_2
        y_hat_curr_step = torch.cat((y_hat_0_1, y_hat_1_0, y_hat_2_3, y_hat_3_2), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        return y_hat_so_far

from checkpoints import load_checkpoint, strict_load_original, export_checkpoint, normalize_state_dict
