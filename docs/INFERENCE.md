# Quad-DeepSC inference details

This guide documents the semantic communication component of the [joint Quad-DeepSC / Quad-LIC release](../README.md). See the main README for installation, model downloads and commands.

## Operating points

Rate sets are explicit and never guessed from a checkpoint filename. PSNR 0040 uses low/shift0; 0120, 0240 and 0450 use high/shift1,2,3. MS-SSIM 0050/0240 use low/shift1; 0800/2000 use high/shift3. LPIPS 0010/0040 use low/shift0; 0120/0240 use high/shift1,2; 10 dB 0005 uses ultra_low/shift5 with nearest-rate selection. INDEX migration executes in the original order after ceiling selection. The three full rate lists are in [model.py](../quad_deepsc/model.py) and stored in checkpoint tensors. Every published profile is in the [manifest](../checkpoints/manifest.json).

The original evaluation path samples bounded uniform latent proxies even in evaluation mode. This release preserves those samples and the CompressAI Gaussian-likelihood arithmetic because the existing rate selection uses them. The default seed is 1926, set once per checkpoint before model loading and the ordered image loop. Replacing this with deterministic rounding changes operating points. CPU inference is available with `--device cpu`.

## Rates, metrics and timing

The evaluator reports RGB PSNR, MS-SSIM and VGG LPIPS as arithmetic means per image. The aggregate MS-SSIM dB score is computed from mean raw MS-SSIM. LPIPS may download its pretrained VGG network on first use. Use `--metrics psnr` to avoid metric-network downloads in a smoke test. Quality networks and metric filters run with TF32 disabled even when the codec enables cuDNN TF32.

PNG rate-map disk write/read is performed and verified on every frame. Side information is delivered without errors. Nominal channel usage uses code rate 658/1024 and 16-QAM at 10 dB, and 679/1024 with QPSK at 6 dB. No LDPC or digital-channel simulation is included. Padding is right/bottom replication to a multiple of 64; metrics and rates use original image dimensions. Record the input dataset and ordering with reproduction runs.

The benchmark reports synchronized encoding/channel/decoding and end-to-end latency after warmup. End-to-end timing includes source PNG read, padding and rate-map PNG I/O, and excludes quality metrics and reconstruction file writes. Parameters and supported arithmetic FLOPs/MACs use PyTorch FlopCounterMode. A multiply-add is two FLOPs; elementwise operations and PNG/host operations are excluded from arithmetic counts. The JSON records dimensions, environment, per-operator counts, samples and percentiles.

## Maintainer checkpoint export

From the repository root:

```bash
python scripts/export_checkpoint.py --profile psnr_awgn10_0040 \
  --input original.pth.tar --output checkpoints/QJSCC_awgn_10dB_0040_psnr.pt
```

Conversion removes unused source entropy-bottleneck/conditional state, regenerates resolution caches, then strictly checks all inference tensors and rate tables. Published files contain tensors only and load with `weights_only=True`. Original research checkpoints are not modified.
