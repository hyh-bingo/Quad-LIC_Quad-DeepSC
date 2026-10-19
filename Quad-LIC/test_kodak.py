import argparse
import platform
from pathlib import Path
import statistics
import time
import torch
from pytorch_msssim import ms_ssim
from runtime import prepare_model, read_image, roundtrip, save_image, psnr, write_json, synchronize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', nargs='+', type=Path, required=True)
    parser.add_argument('--allow-tf32', action='store_true', help='Allow cuDNN convolution TF32 as in the research source; matmul TF32 remains disabled')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, default=Path('results/kodak.json'))
    parser.add_argument('--streams', type=Path)
    parser.add_argument('--reconstructions', type=Path)
    parser.add_argument('--acceleration', choices=['none', 'cuda', 'sdpa', 'all'], default='none')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.threads <= 0 or args.warmup < 0 or (args.limit is not None and args.limit <= 0):
        parser.error('threads and limit must be positive, warmup must be nonnegative')
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    paths = sorted(path for path in args.dataset.iterdir() if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'})
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error('No images found')
    outputs = {'dataset': str(args.dataset), 'image_count': len(paths), 'device': args.device, 'precision_profile': 'source_cudnn_tf32' if args.allow_tf32 else 'strict_fp32', 'numerics': {'dtype': 'float32', 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}, 'torch': torch.__version__, 'python': platform.python_version(), 'cuda_runtime': torch.version.cuda, 'device_name': torch.cuda.get_device_name(torch.device(args.device)) if torch.device(args.device).type == 'cuda' else platform.machine(), 'bpp_definition': '8 * (rANS payload bytes + 12-byte height/width/length header) / original RGB pixel count', 'latency_definition': 'Synchronized encode/decode wall time, includes neural transforms, CPU rANS and CPU/GPU transfers; excludes PNG and bitstream disk I/O, padding, model/CDF setup, and metrics', 'quality_metric_numerics': {'ms_ssim_cudnn_allow_tf32': False, 'psnr_dtype': 'float32'}, 'psnr_infinite_representation': None, 'checkpoints': []}
    for checkpoint in args.checkpoint:
        model, metadata, acceleration = prepare_model(checkpoint, args.device, args.acceleration)
        warmup_image = read_image(paths[0], args.device)
        for _ in range(args.warmup):
            roundtrip(model, warmup_image)
        records = []
        for path in paths:
            synchronize(args.device)
            started = time.perf_counter()
            x = read_image(path, args.device)
            synchronize(args.device)
            png_read_seconds = time.perf_counter() - started
            reconstruction, stream, timing = roundtrip(model, x)
            height, width = x.shape[-2:]
            with torch.inference_mode(), torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled, benchmark=False,
                deterministic=True, allow_tf32=False,
            ):
                ssim_value = float(ms_ssim(x, reconstruction, data_range=1.0).item())
            record = {'image': path.name, 'height': height, 'width': width, 'pixels': height * width, 'bytes': len(stream), 'payload_bytes': len(stream) - 12, 'bpp': len(stream) * 8 / (height * width), 'psnr_db': psnr(x, reconstruction), 'ms_ssim': ssim_value, 'png_read_seconds': png_read_seconds, **timing}
            if args.streams:
                target = args.streams / checkpoint.stem / f'{path.stem}.bin'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream)
            if args.reconstructions:
                synchronize(args.device)
                started = time.perf_counter()
                save_image(reconstruction, args.reconstructions / checkpoint.stem / path.name)
                synchronize(args.device)
                record['png_write_seconds'] = time.perf_counter() - started
            records.append(record)
            print(f'{checkpoint.name} {path.name}: {record["bpp"]:.4f} bpp, PSNR={record["psnr_db"]}, MS-SSIM={ssim_value:.6f}', flush=True)
        keys = ['bpp', 'ms_ssim', 'encoding_seconds', 'decoding_seconds', 'codec_seconds']
        averages = {key: statistics.mean(record[key] for record in records) for key in keys}
        averages['psnr_db'] = None if any(record['psnr_db'] is None for record in records) else statistics.mean(record['psnr_db'] for record in records)
        averages['pixel_weighted_bpp'] = sum(record['bytes'] for record in records) * 8 / sum(record['pixels'] for record in records)
        outputs['checkpoints'].append({'metadata': metadata, 'acceleration': acceleration, 'average': averages, 'images': records})
        write_json(args.output, outputs)
    print(args.output)


if __name__ == '__main__':
    main()
