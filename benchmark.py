import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from evaluate import parser, seed_all, read_image, run_frame, configure
from quad_deepsc.checkpoints import load

def main():
    p = parser()
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--no-cuda-kernels', dest='cuda_kernels', action='store_false')
    p.add_argument('--no-sdpa', dest='sdpa', action='store_false')
    p.set_defaults(cuda_kernels=True, sdpa=True)
    args = p.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError('warmup >= 0 and repeats >= 1 required')
    if args.cuda_kernels or args.sdpa:
        if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
            p.error('CUDA latency requires a CUDA device; use --no-cuda-kernels --no-sdpa for reference profiling')
        torch.cuda.get_device_properties(torch.device(args.device))
        try:
            from quad_acceleration import backend_status
        except ImportError:
            p.error('CUDA operators are unavailable; run bash scripts/setup_cuda.sh')
        status = backend_status()
        if args.cuda_kernels and not status['cuda_extension_loaded']:
            p.error('CUDA operators are unavailable; run bash scripts/setup_cuda.sh')
        if args.cuda_kernels and 'channels_last_4d' not in status.get('cuda_supported_layouts', []):
            p.error('CUDA operators are incompatible; rebuild with bash scripts/setup_cuda.sh')
        if args.sdpa and not status['sdpa_available']:
            p.error('SDPA is unavailable in the installed PyTorch')
    seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    model, cfg = load(args.profile,args.checkpoint_root,args.device)
    x = read_image(args.dataset,args.device)
    h,w = x.shape[-2:]
    x = F.pad(x,(0,-w%64,0,-h%64),mode='replicate')
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        model(x)
    flops = counter.get_total_flops()
    configured = configure(model,args)
    if args.cuda_kernels and not configured.get('dc_ffn_patched'):
        p.error('The model has no applicable CUDA operators')
    if args.sdpa and not (configured.get('window_attention_patched') or configured.get('cross_attention_patched')):
        p.error('The model has no applicable SDPA modules')
    timings = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.warmup+args.repeats):
            seed_all(args.seed+i)
            _,_,row = run_frame(model,args.dataset,args.device,Path(tmp)/'map.png')
            if i >= args.warmup:
                timings.append(row)
    keys = ['encoding_ms','channel_ms','decoding_ms','end_to_end_ms']
    stats = {k:{'mean':float(np.mean([r[k] for r in timings])),'p50':float(np.median([r[k] for r in timings])),'p95':float(np.percentile([r[k] for r in timings],95))} for k in keys}
    report = {'profile':args.profile,'input_hw':[h,w],'padded_hw':list(x.shape[-2:]),'parameters':sum(p.numel() for p in model.parameters()),'flops':flops,'macs':flops/2,'complexity_scope':'PyTorch FlopCounterMode counted convolution, transposed convolution and matrix products (multiply-add = 2 FLOPs). Counts include dynamic rate-adapter matrix products and attention. Excludes elementwise, indexing, entropy/PNG and host I/O; counted on the reference implementation.','op_flops':{str(k):v for k,v in counter.get_flop_counts()['Global'].items()},'warmup':args.warmup,'repeats':args.repeats,'latency_ms':stats,'device':args.device,'gpu':torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_kernels':args.cuda_kernels,'sdpa':args.sdpa,'timing_scope':'Synchronized wall time; input PNG read, rate-map PNG write/read, channel, reconstruction included. No reconstruction file write; quality metrics excluded.','samples':timings}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
