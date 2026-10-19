import math
import time
import numpy as np
import torch
import torch.nn as nn
from .channel import AWGN_channel
from .layers import SwinBasicLayer, CrossAttentionLayer, RateAdaptionEncoder, RateAdaptionDecoder
from .source import QSC_for_JSCC

class QJSCC(nn.Module):

    def __init__(self, N=192, M=320, rate_type='low', distortion_type='mse', rate_shift=0, eta=0.2, channel_type='awgn', SNR=10, inplace=False, input_resolution=(16, 16), depths=[4, 4], num_heads=[5, 5], window_size=4, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.eta = eta
        self.rate_type = rate_type
        self.channel_type = channel_type
        self.distortion_type = distortion_type
        self.source_model = QSC_for_JSCC(N=N, M=M, inplace=inplace)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if rate_type == 'low':
            self.rate_choice_new = [4, 8, 16, 24, 32, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 132, 140, 148, 156, 164, 180, 200, 220, 240, 260, 280, 320]
        elif rate_type == 'high':
            self.rate_choice_new = [4, 8, 16, 24, 32, 40, 44, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 270, 272, 278, 288, 304, 320]
        elif rate_type == 'ultra_low':
            self.rate_choice_new = [0, 2, 4, 6, 8, 12, 16, 24, 32, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 132, 140, 148, 156, 164, 180, 200, 220, 240, 260, 280, 320]
        else:
            raise ValueError(rate_type)
        self.rate_shift = rate_shift
        self.rate_num = len(self.rate_choice_new)
        self.register_buffer('rate_choice_tensor_new', torch.tensor(np.asarray(self.rate_choice_new)))
        self.channel_sep_enc = SwinBasicLayer(dim=M // 4, input_resolution=input_resolution, depth=depths[0], num_heads=num_heads[0], window_size=window_size, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:0]):sum(depths[:1])])
        self.channel_sep_dec = SwinBasicLayer(dim=M // 4, input_resolution=input_resolution, depth=depths[0], num_heads=num_heads[0], window_size=window_size, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:0]):sum(depths[:1])])
        self.channel_ctx_encs = nn.ModuleList([CrossAttentionLayer(dim=M // 4, input_resolution=input_resolution, depth=depths[0], num_heads=num_heads[0], window_size=window_size, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:0]):sum(depths[:1])]) for i in range(3)])
        self.channel_ctx_decs = nn.ModuleList([CrossAttentionLayer(dim=M // 4, input_resolution=input_resolution, depth=depths[0], num_heads=num_heads[0], window_size=window_size, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:0]):sum(depths[:1])]) for i in range(3)])
        self.channel_token_enc_0 = nn.Parameter(torch.zeros(self.rate_num, M // 4))
        self.channel_token_enc_1 = nn.Parameter(torch.zeros(self.rate_num, M // 4))
        self.channel_token_enc_2 = nn.Parameter(torch.zeros(self.rate_num, M // 4))
        self.channel_token_enc_3 = nn.Parameter(torch.zeros(self.rate_num, M // 4))
        self.modulation = SwinBasicLayer(dim=M, input_resolution=input_resolution, depth=depths[1], num_heads=num_heads[1] * 4, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:0]):sum(depths[:1])], norm_layer=norm_layer)
        self.demodulation = SwinBasicLayer(dim=M, input_resolution=input_resolution, depth=depths[1], num_heads=num_heads[1] * 4, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:0]):sum(depths[:1])], norm_layer=norm_layer)
        self.rate_adaption_enc = RateAdaptionEncoder(M, self.rate_choice_new)
        self.rate_adaption_dec = RateAdaptionDecoder(M, self.rate_choice_new)
        self.channel_token_dec_0 = nn.Parameter(torch.zeros(self.rate_num, M // 4))
        self.channel_token_dec_1 = nn.Parameter(torch.zeros(self.rate_num, M // 4))
        self.channel_token_dec_2 = nn.Parameter(torch.zeros(self.rate_num, M // 4))
        self.channel_token_dec_3 = nn.Parameter(torch.zeros(self.rate_num, M // 4))
        self.rate_token_enc = nn.Parameter(torch.zeros(self.rate_num, M))
        self.rate_token_dec = nn.Parameter(torch.zeros(self.rate_num, M))
        self.channel = AWGN_channel(channel_type=channel_type, SNR=SNR)

    def MERGE(self, y_0, y_1, y_2, y_3):
        B, C, H, W = y_0.shape
        dtype = y_0.dtype
        device = y_0.device
        mask_0, mask_1, mask_2, mask_3 = self.source_model.get_mask_four_parts(H, W, dtype, device)
        x_0 = y_0 * mask_0 + y_1 * mask_3 + y_2 * mask_2 + y_3 * mask_1
        x_1 = y_0 * mask_1 + y_1 * mask_2 + y_2 * mask_3 + y_3 * mask_0
        x_2 = y_0 * mask_2 + y_1 * mask_1 + y_2 * mask_0 + y_3 * mask_3
        x_3 = y_0 * mask_3 + y_1 * mask_0 + y_2 * mask_1 + y_3 * mask_2
        x = torch.cat([x_0, x_1, x_2, x_3], dim=1)
        return x

    def get_indexes(self, likelihoods, eta, rate_choice=None, rate_num=None):
        hx = torch.clamp_min(-torch.log(likelihoods) / math.log(2), 0)
        symbol_num = torch.sum(hx, dim=1).flatten(0) * eta
        indexes = torch.searchsorted(rate_choice, symbol_num).clamp(0, rate_num - 1)
        if self.rate_shift == 5:
            left = rate_choice[indexes - 1]
            right = rate_choice[indexes]
            indexes = torch.where(symbol_num - left >= right - symbol_num, indexes, indexes - 1)
        indexes = indexes.clamp(0, rate_num - 1)
        return indexes

    def c_tx(self, x, indexes):
        y_0, y_1, y_2, y_3 = x.chunk(4, 0)
        B, C, H, W = y_0.size()
        y_0 = y_0 + torch.index_select(self.channel_token_enc_0, 0, indexes).reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_1 = y_1 + torch.index_select(self.channel_token_enc_1, 0, indexes).reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_2 = y_2 + torch.index_select(self.channel_token_enc_2, 0, indexes).reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_3 = y_3 + torch.index_select(self.channel_token_enc_3, 0, indexes).reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_0, y_1, y_2, y_3 = self.channel_sep_enc(torch.cat([y_0, y_1, y_2, y_3], dim=0)).chunk(4, 0)
        y_1, y_2, y_3 = self.channel_ctx_encs[0](torch.cat([y_1, y_2, y_3], dim=0), torch.cat([y_0, y_0, y_0], dim=0)).chunk(3, 0)
        y_2, y_3 = self.channel_ctx_encs[1](torch.cat([y_2, y_3], dim=0), torch.cat([y_1, y_1], dim=0)).chunk(2, 0)
        y_3 = self.channel_ctx_encs[2](y_3, y_2)
        return torch.cat([y_0, y_1, y_2, y_3], dim=1)

    def c_rx(self, x, indexes):
        y_0, y_1, y_2, y_3 = x.chunk(4, 1)
        B, C, H, W = y_0.size()
        y_0 = y_0 + torch.index_select(self.channel_token_dec_0, 0, indexes).reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_1 = y_1 + torch.index_select(self.channel_token_dec_1, 0, indexes).reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_2 = y_2 + torch.index_select(self.channel_token_dec_2, 0, indexes).reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_3 = y_3 + torch.index_select(self.channel_token_dec_3, 0, indexes).reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_0 = self.channel_sep_dec(y_0)
        y_1 = self.channel_ctx_decs[0](y_1, y_0)
        y_1 = self.channel_sep_dec(y_1)
        y_2 = self.channel_ctx_decs[1](y_2, y_1)
        y_2 = self.channel_ctx_decs[0](y_2, y_0)
        y_2 = self.channel_sep_dec(y_2)
        y_3 = self.channel_ctx_decs[2](y_3, y_2)
        y_3 = self.channel_ctx_decs[1](y_3, y_1)
        y_3 = self.channel_ctx_decs[0](y_3, y_0)
        y_3 = self.channel_sep_dec(y_3)
        return self.MERGE(y_0, y_1, y_2, y_3)

    def m_tx(self, x, indexes):
        B, C, H, W = x.size()
        rate_token = torch.index_select(self.rate_token_enc, 0, indexes)
        rate_token = rate_token.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.modulation(x + rate_token)
        x_masked, mask = self.rate_adaption_enc(x, indexes)
        return (x_masked, mask)

    def m_rx(self, x, indexes):
        B, C, H, W = x.size()
        x = self.rate_adaption_dec(x, indexes)
        rate_token = torch.index_select(self.rate_token_dec, 0, indexes)
        rate_token = rate_token.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.demodulation(x + rate_token)
        return x

    def symbol_pass_channel(self, s_masked, mask=None):
        if mask is None:
            mask = torch.ones_like(s_masked)
        if mask.sum().item() == 0:
            return torch.zeros_like(s_masked), mask.sum() / 2, s_masked
        avg_pwr = torch.sum(s_masked ** 2) / mask.sum()
        s_hat, _ = self.channel.forward(s_masked, avg_pwr)
        raw_data = s_masked
        s_hat = s_hat * mask
        channel_usage = mask.sum() / 2
        return (s_hat, channel_usage, raw_data)

    def compress(self, x):
        tx_source = self.source_model.jscc_encoder(x)
        y_s_tx = tx_source['y_jscc']
        y_likelihoods = tx_source['likelihoods']['y']
        indexes = self.get_indexes(y_likelihoods, self.eta, self.rate_choice_tensor_new, self.rate_num)
        if self.rate_shift == 0:
            indexes[indexes >= self.rate_num // 3 * 2] -= 1
            indexes[indexes == 5] += 1
        elif self.rate_shift == 2:
            indexes[indexes >= self.rate_num // 3 * 2] -= 1
            indexes[indexes >= self.rate_num // 2] -= 1
            indexes[indexes == 5] += 1
        elif self.rate_shift == 3:
            indexes[indexes >= self.rate_num // 3 * 2] -= 2
            indexes[indexes >= self.rate_num // 2] -= 1
            indexes[indexes == 5] += 1
        y_c_tx = self.c_tx(y_s_tx, indexes)
        y_m_tx, mask = self.m_tx(y_c_tx, indexes)
        return (y_m_tx, mask, indexes)

    def decompress(self, y_m_rx, indexes):
        y_c_rx = self.m_rx(y_m_rx, indexes)
        y_s_rx = self.c_rx(y_c_rx, indexes)
        return self.source_model.jscc_decoder(y_s_rx)

    @torch.inference_mode()
    def forward(self, x):
        y, mask, indexes = self.compress(x)
        received, usage, _ = self.symbol_pass_channel(y, mask)
        out = self.decompress(received, indexes)
        return {'x_hat': out['x_hat'], 'indexes': indexes, 'k': usage}
