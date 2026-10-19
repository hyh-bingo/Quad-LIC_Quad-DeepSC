import argparse
import platform
import statistics
from pathlib import Path
import torch
from runtime import prepare_model, roundtrip, write_json, read_image


def require_cuda_backend(device):
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA latency requires a CUDA device; use --acceleration none for reference or CPU inference')
    torch.cuda.get_device_properties(device)
    try:
        from quad_acceleration import backend_status
    except (ImportError, OSError) as error:
        raise RuntimeError('CUDA backend unavailable; run bash scripts/setup_cuda.sh') from error
    status = backend_status()
    if not status.get('cuda_extension_loaded'):
        raise RuntimeError('CUDA extension unavailable; run bash scripts/setup_cuda.sh')
    if 'channels_last_4d' not in status.get('cuda_supported_layouts', []):
        raise RuntimeError('CUDA extension is incompatible; rebuild with bash scripts/setup_cuda.sh')


class MacCounter:
    def __init__(self, model):
        self.macs = 0
        self.handles = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
                self.handles.append(module.register_forward_hook(self.count))

    def count(self, module, inputs, output):
        if isinstance(module, torch.nn.ConvTranspose2d):
            self.macs += inputs[0].numel() * (module.out_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, torch.nn.Conv2d):
            self.macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        else:
            self.macs += output.numel() * module.in_features

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--allow-tf32', action='store_true', help='Allow cuDNN convolution TF32 as in the research source; matmul TF32 remains disabled')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--acceleration', choices=['none', 'cuda', 'sdpa', 'all'], default='cuda')
    parser.add_argument('--output', type=Path, default=Path('results/benchmark.json'))
    args = parser.parse_args()
    if min(args.height, args.width, args.iterations, args.threads) <= 0 or args.warmup < 0:
        parser.error('dimensions, iterations and threads must be positive; warmup must be nonnegative')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    cuda_requested = args.acceleration in ('cuda', 'all')
    if cuda_requested:
        require_cuda_backend(args.device)
    model, metadata, acceleration = prepare_model(args.checkpoint, args.device, args.acceleration, require_cuda=cuda_requested)
    x = read_image(args.image, args.device) if args.image else torch.rand(1, 3, args.height, args.width, device=args.device)
    args.height, args.width = x.shape[-2:]
    for _ in range(args.warmup):
        roundtrip(model, x)
    records = [roundtrip(model, x)[2] for _ in range(args.iterations)]
    del model
    baseline, _metadata, _report = prepare_model(args.checkpoint, args.device, 'none')
    counter = MacCounter(baseline)
    roundtrip(baseline, x)
    counter.close()
    output = {'metadata': metadata, 'device': args.device, 'precision_profile': 'source_cudnn_tf32' if args.allow_tf32 else 'strict_fp32', 'numerics': {'dtype': 'float32', 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}, 'torch': torch.__version__, 'python': platform.python_version(), 'cuda_runtime': torch.version.cuda, 'device_name': torch.cuda.get_device_name(torch.device(args.device)) if torch.device(args.device).type == 'cuda' else platform.machine(), 'acceleration': acceleration, 'input_image': args.image.name if args.image else None, 'height': args.height, 'width': args.width, 'iterations': args.iterations, 'warmup': args.warmup, 'parameter_count': sum(p.numel() for p in baseline.parameters()), 'macs': counter.macs, 'flops_2x_macs': 2 * counter.macs, 'complexity_definition': 'Logical Conv2d, ConvTranspose2d and Linear MACs for one complete actual compression plus decompression; h_s and spatial priors execute in both directions. ConvTranspose2d counts input elements * output channels per group * kernel area, including boundary products. FLOPs=2*MACs. Excludes biases, activations, attention gating, interpolation, entropy/CDF, tensor operations, PNG I/O. Counted on the reference model.', 'latency_definition': 'Synchronized wall time, includes neural transforms, CPU rANS and CPU/GPU transfers; excludes PNG I/O, padding, model/CDF setup and metrics', 'timing_seconds': {key: {'mean': statistics.mean(r[key] for r in records), 'median': statistics.median(r[key] for r in records), 'min': min(r[key] for r in records), 'max': max(r[key] for r in records)} for key in records[0]}, 'samples': records}
    write_json(args.output, output)
    print(args.output)


if __name__ == '__main__':
    main()
