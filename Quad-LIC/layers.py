import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2)

def conv1x1(in_ch: int, out_ch: int, stride: int=1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3(in_ch: int, out_ch: int, stride: int=1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

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
