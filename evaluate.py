import argparse
import io
import json
import math
import random
import tempfile
import time
from pathlib import Path
import numpy as np
import png
from PIL import Image
import torch
import torch.nn.functional as F
from quad_deepsc.checkpoints import load, profiles

def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_image(path, device):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255

@torch.inference_mode()
def run_frame(model, path, device, side_path, save_path=None):
    sync(device)
    start = time.perf_counter()
    x = read_image(path, device)
    h, w = x.shape[-2:]
    padded = F.pad(x, (0, -w % 64, 0, -h % 64), mode='replicate')
    sync(device)
    enc_start = time.perf_counter()
    symbols, mask, indexes = model.compress(padded)
    grid = indexes.reshape(padded.shape[-2] // 16, padded.shape[-1] // 16).cpu().numpy().astype(np.uint8)
    png.from_array(grid, 'L').save(str(side_path))
    sync(device)
    enc_end = time.perf_counter()
    received, usage, _ = model.symbol_pass_channel(symbols, mask)
    sync(device)
    channel_end = time.perf_counter()
    _, _, pixels, _ = png.Reader(filename=str(side_path)).read()
    decoded = torch.from_numpy(np.vstack(list(pixels)).astype(np.int64)).flatten().to(device)
    if not torch.equal(indexes, decoded):
        raise RuntimeError('PNG rate map roundtrip mismatch')
    x_hat = model.decompress(received, decoded)['x_hat'][..., :h, :w].clamp(0, 1)
    if not torch.isfinite(x_hat).all():
        raise RuntimeError('Nonfinite reconstruction')
    sync(device)
    dec_end = time.perf_counter()
    if save_path:
        array = (x_hat[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
        Image.fromarray(array).save(save_path)
    sync(device)
    end = time.perf_counter()
    side_bits = side_path.stat().st_size * 8
    cbr_main = usage.item() / (h * w * 3)
    snr = model.channel.chan_param
    if snr == 10:
        side_factor = 1024 / 658 / 4
    elif snr == 6:
        side_factor = 1024 / 679 / 2
    else:
        raise ValueError('Side-link conversion is defined only for released 6/10 dB profiles')
    cbr_side = side_bits / (h * w * 3) * side_factor
    mse = F.mse_loss(x_hat, x).item()
    return x, x_hat, {'psnr_db': -10 * math.log10(max(mse, 1e-12)), 'cbr':cbr_main+cbr_side, 'cbr_main':cbr_main, 'cbr_side':cbr_side, 'side_bits':side_bits, 'encoding_ms':(enc_end-enc_start)*1000, 'channel_ms':(channel_end-enc_end)*1000, 'decoding_ms':(dec_end-channel_end)*1000, 'end_to_end_ms':(end-start)*1000}

def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', choices=sorted(profiles()), required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint-root', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1926)
    p.add_argument('--limit', type=int)
    p.add_argument('--save-recon', action='store_true')
    p.add_argument('--metrics', nargs='+', choices=['psnr','msssim','lpips'], default=['psnr','msssim','lpips'])
    p.add_argument('--lpips-net', choices=['vgg','alex'], default='vgg')
    p.add_argument('--allow-tf32', action='store_true')
    p.add_argument('--sdpa', action='store_true')
    p.add_argument('--cuda-kernels', action='store_true')
    return p

def configure(model, args):
    if args.sdpa or args.cuda_kernels:
        from quad_acceleration import configure
        return configure(model, sdpa=args.sdpa, cuda=args.cuda_kernels)
    return {}

def main():
    args = parser().parse_args()
    seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    files = sorted(p for p in args.dataset.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.ppm','.bmp','.tif','.tiff'})
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('limit must be positive')
        files = files[:args.limit]
    if not files:
        raise ValueError('Dataset contains no supported images')
    model, cfg = load(args.profile, args.checkpoint_root, args.device)
    configure(model, args)
    lpips_model = None
    if 'lpips' in args.metrics:
        import lpips
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(args.device).eval().requires_grad_(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_recon:
        recon = args.output.parent / (args.output.stem + '_recon')
        recon.mkdir(exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        for i, path in enumerate(files):
            x, x_hat, row = run_frame(model, path, args.device, Path(temp)/'side.png', recon/path.name if args.save_recon else None)
            with torch.inference_mode(), torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, benchmark=False, deterministic=True, allow_tf32=False):
                if 'msssim' in args.metrics:
                    from pytorch_msssim import ms_ssim
                    score = ms_ssim(x_hat,x,data_range=1.).item()
                    row.update(ms_ssim=score, ms_ssim_db=-10*math.log10(max(1-score,1e-12)))
                if lpips_model:
                    row['lpips_'+args.lpips_net] = lpips_model(x_hat*2-1,x*2-1).item()
            row['image'] = path.name
            rows.append(row)
            print(json.dumps(row), flush=True)
    means = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != 'image'}
    if 'ms_ssim' in means:
        means['mean_image_ms_ssim_db'] = means['ms_ssim_db']
        means['ms_ssim_db'] = -10 * math.log10(max(1-means['ms_ssim'],1e-12))
    report = {'profile':args.profile,'lpips_backbone':args.lpips_net,'configuration':cfg,'seed':args.seed,'count':len(rows),'device':str(args.device),'quality_metric_tf32':False,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None,'cuda_kernels':args.cuda_kernels,'sdpa':args.sdpa,'timing':'Synchronized wall time; includes rate-map PNG disk write/read. End-to-end includes input read, padding and optional reconstruction PNG write; excludes quality metrics. No warmup: use benchmark.py for warmed latency.','side_link':'Error-free PNG rate map; 10 dB: nominal code rate658/1024 and16-QAM; 6 dB:679/1024 andQPSK; no coded-channel simulation.','mean':means,'images':rows}
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
