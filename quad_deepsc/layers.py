from typing import Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2)

def conv1x1(in_ch: int, out_ch: int, stride: int=1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3(in_ch: int, out_ch: int, stride: int=1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

class Quantizer:

    def quantize(self, inputs, mode=None):
        return inputs + torch.empty_like(inputs).uniform_(-0.5, 0.5)

class AttentionBlock(nn.Module):

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(conv1x1(N, N // 2), nn.ReLU(inplace=True), conv3x3(N // 2, N // 2), nn.ReLU(inplace=True), conv1x1(N // 2, N))
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out
        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())
        self.conv_b = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit(), conv1x1(N, N))

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class ResidualBottleneckBlock(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch // 2, in_ch // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch // 2, in_ch)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = out + identity
        return out

def subpel_conv1x1(in_ch, out_ch, r=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r))

class UNet(nn.Module):

    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = DepthConvBlock(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock(64, 128, inplace=inplace)
        self.context_refine = nn.Sequential(DepthConvBlock(128, 128, inplace=inplace), DepthConvBlock(128, 128, inplace=inplace), DepthConvBlock(128, 128, inplace=inplace), DepthConvBlock(128, 128, inplace=inplace))
        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock(128, 64, inplace=inplace)
        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock(64, out_ch, inplace=inplace)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)
        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)
        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RateAdaptionEncoder(nn.Module):

    def __init__(self, channel_num, rate_choice, mode='CHW'):
        super(RateAdaptionEncoder, self).__init__()
        self.C, self.H, self.W = (channel_num, 16, 16)
        self.rate_num = len(rate_choice)
        self.rate_choice = rate_choice
        self.register_buffer('rate_choice_tensor_new', torch.tensor(np.asarray(rate_choice)))
        self.weight = nn.Parameter(torch.zeros(self.rate_num, self.C, max(self.rate_choice)))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice)))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.zeros_(self.bias)
        mask = torch.arange(0, max(self.rate_choice)).repeat(self.H * self.W, 1)
        self.register_buffer('mask', mask)

    def forward(self, x, indexes):
        B, C, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        if H != self.H or W != self.W:
            self.update_resolution(H, W, x.device)
        w = torch.index_select(self.weight, 0, indexes).reshape(B, H * W, self.C, -1)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, -1)
        mask = self.mask.repeat(B, 1, 1)
        rate_constraint = self.rate_choice_tensor_new[indexes].reshape(B, H * W, 1).repeat(1, 1, max(self.rate_choice))
        mask[mask < rate_constraint] = 1.0
        mask[mask >= rate_constraint] = 0.0
        x_BLC_masked = (torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b) * mask
        x_masked = x_BLC_masked.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        mask_BCHW = mask.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return (x_masked, mask_BCHW)

    def update_resolution(self, H, W, device):
        self.H = H
        self.W = W
        self.num_patches = H * W
        self.mask = torch.arange(0, max(self.rate_choice), device=device).repeat(self.num_patches, 1)
        self.mask = self.mask.to(device)

class RateAdaptionDecoder(nn.Module):

    def __init__(self, channel_num, rate_choice, mode='CHW'):
        super(RateAdaptionDecoder, self).__init__()
        self.C = channel_num
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.weight = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice), self.C))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, self.C))
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, indexes):
        B, _, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        w = torch.index_select(self.weight, 0, indexes).reshape(B, H * W, max(self.rate_choice), self.C)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, self.C)
        x_BLC = torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b
        out = x_BLC.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class WindowCrossAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, relative_pos_embedding=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.relative_pos_embedding = relative_pos_embedding
        if relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, mask=None):
        B_, N, C = x1.shape
        q = self.q(x1).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(x2).reshape(B_, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (q[0], kv[0], kv[1])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        B_, N, C = x_windows.shape
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        if self.shift_size > 0:
            H, W = self.input_resolution
            H = H + (self.window_size - H % self.window_size) % self.window_size
            W = W + (self.window_size - W % self.window_size) % self.window_size
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.to(self.norm1.weight.device)
        else:
            pass

class CrossAttnBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=4, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, relative_pos_embedding=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = WindowCrossAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, relative_pos_embedding=relative_pos_embedding, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x1, x2):
        H, W = self.input_resolution
        B, L1, C = x1.shape
        if x2.dim() == 4:
            B, T, L2, C = x2.shape
        else:
            B, L2, C = x2.shape
            T = 1
        shortcut = x1
        x1 = self.norm1(x1)
        x2 = self.norm1(x2)
        x1 = x1.view(B, H, W, C)
        x2 = x2.view(B * T, H, W, C)
        x1_windows = window_partition(x1, self.window_size)
        x2_windows = window_partition(x2, self.window_size)
        x1_windows = x1_windows.view(-1, self.window_size * self.window_size, C)
        x2_windows = x2_windows.view(-1, T * self.window_size * self.window_size, C)
        B_, N, C = x1_windows.shape
        attn_windows = self.attn(x1_windows, x2_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        if self.shift_size > 0:
            H, W = self.input_resolution
            H = H + (self.window_size - H % self.window_size) % self.window_size
            W = W + (self.window_size - W % self.window_size) % self.window_size
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.to(self.norm1.weight.device)
        else:
            pass

class SwinBasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=self.input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])

    def forward(self, x):
        B, C, H, W = x.size()
        if self.input_resolution != (H, W):
            self.update_resolution(H, W)
        x = x.flatten(2).permute(0, 2, 1)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()

class CrossAttentionLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, relative_pos_embedding=True, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.crs_blocks = nn.ModuleList([CrossAttnBlock(dim=dim, input_resolution=self.input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0, relative_pos_embedding=relative_pos_embedding, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        self.slf_blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=self.input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])

    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        if self.input_resolution != (H, W):
            self.update_resolution(H, W)
        x1 = x1.flatten(2).permute(0, 2, 1)
        x2 = x2.flatten(2).permute(0, 2, 1)
        for i, blk in enumerate(self.slf_blocks):
            x1 = self.slf_blocks[i](x1)
            x1 = self.crs_blocks[i](x1, x2)
        x1 = x1.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x1

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i, blk in enumerate(self.slf_blocks):
            self.slf_blocks[i].input_resolution = (H, W)
            self.crs_blocks[i].input_resolution = (H, W)
            self.slf_blocks[i].update_mask()
            self.crs_blocks[i].update_mask()

class DepthConv(nn.Module):

    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01, inplace=False):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, dw_ch, 1, stride=stride), nn.LeakyReLU(negative_slope=slope, inplace=inplace))
        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2, groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)
        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)
        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)
        return out + identity

class ConvFFN(nn.Module):

    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        internal_ch = max(min(in_ch * 4, 1024), in_ch * 2)
        self.conv = nn.Sequential(nn.Conv2d(in_ch, internal_ch, 1), nn.LeakyReLU(negative_slope=slope, inplace=inplace), nn.Conv2d(internal_ch, in_ch, 1), nn.LeakyReLU(negative_slope=slope, inplace=inplace))

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        return identity + self.conv(x)

class DepthConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.conv1 = DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv, inplace=inplace)
        self.conv2 = ConvFFN(out_ch, slope=slope_ffn, inplace=inplace)

    def forward(self, x, identity=None):
        x = self.conv1(x)
        x = self.conv2(x, identity)
        return x
