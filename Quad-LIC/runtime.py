from pathlib import Path
import json
import math
import struct
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from checkpoints import load_checkpoint


HEADER = struct.Struct('>III')


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start


def pack_stream(height, width, payload):
    return HEADER.pack(height, width, len(payload)) + payload


def unpack_stream(stream):
    if len(stream) < HEADER.size:
        raise ValueError('Truncated image bitstream header')
    height, width, length = HEADER.unpack_from(stream)
    if min(height, width, length) <= 0 or len(stream) != HEADER.size + length:
        raise ValueError('Invalid image bitstream dimensions or payload length')
    return height, width, stream[HEADER.size:]


def read_image(path, device):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def save_image(x, path):
    array = x.squeeze(0).clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def pad_image(x):
    height, width = x.shape[-2:]
    return F.pad(x, (0, (-width) % 16, 0, (-height) % 16), mode='replicate')


def prepare_model(checkpoint, device, acceleration='none', *, require_cuda=False):
    model, metadata = load_checkpoint(checkpoint)
    model.update(force=True)
    model.to(device)
    if acceleration != 'none':
        from quad_acceleration import configure
        configured = configure(model, cuda=acceleration in ('cuda', 'all'), sdpa=acceleration in ('sdpa', 'all'))
        if require_cuda and configured.get('dc_ffn_patched', 0) <= 0:
            raise RuntimeError('CUDA backend is not enabled for this model')
    return model, metadata, {'mode': acceleration}


@torch.inference_mode()
def roundtrip(model, x):
    height, width = x.shape[-2:]
    padded = pad_image(x)
    compressed, encoding_seconds = timed(x.device, lambda: model.compress(padded))
    stream = pack_stream(height, width, compressed['bit_stream'])
    stream_height, stream_width, payload = unpack_stream(stream)
    decoded, decoding_seconds = timed(x.device, lambda: model.decompress(payload, stream_height, stream_width))
    reconstruction = decoded['x_hat'][..., :height, :width]
    return reconstruction, stream, {'encoding_seconds': encoding_seconds, 'decoding_seconds': decoding_seconds, 'codec_seconds': encoding_seconds + decoding_seconds}


def psnr(x, reconstruction):
    mse = torch.mean((x - reconstruction) ** 2).item()
    return None if mse == 0 else -10.0 * math.log10(mse)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
