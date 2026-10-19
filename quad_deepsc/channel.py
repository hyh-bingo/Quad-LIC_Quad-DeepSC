import math
import torch
from torch import nn

class AWGN_channel(nn.Module):

    def __init__(self, channel_type='awgn', SNR=10):
        super().__init__()
        if channel_type != 'awgn':
            raise ValueError('Only the released AWGN operating points are supported')
        self.chan_param = SNR

    def forward(self, inputs, avg_pwr=None):
        power = inputs.square().mean() if avg_pwr is None else avg_pwr
        if power.item() == 0:
            return torch.zeros_like(inputs), inputs.numel() // 2
        normalized = inputs / torch.sqrt(power * 2)
        flat = normalized.reshape(inputs.shape[0], -1)
        symbols = torch.complex(flat[:, ::2], flat[:, 1::2])
        std = math.sqrt(1 / (2 * 10 ** (self.chan_param / 10)))
        real = torch.normal(0.0, std, size=symbols.shape, device=inputs.device)
        imag = torch.normal(0.0, std, size=symbols.shape, device=inputs.device)
        received = symbols + torch.complex(real, imag)
        result = torch.empty_like(flat)
        result[:, ::2] = received.real
        result[:, 1::2] = received.imag
        return (result.reshape_as(inputs) * torch.sqrt(power * 2), symbols.numel())
