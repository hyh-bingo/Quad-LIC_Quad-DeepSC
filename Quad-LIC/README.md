# Quad-LIC

Quad-LIC is the learned image compression component of the [Quad-DeepSC work](../README.md). It provides image transforms, a four-pass spatial prior, and CPU rANS encoding and decoding. Five PSNR and four MS-SSIM checkpoints are available through the shared release.

## Installation

Use the common environment from the repository root:

```bash
bash scripts/setup.sh
source .venv/bin/activate
cd Quad-LIC
```

The environment is `../.venv` relative to this directory. See the [parent README](../README.md) for platform and installation options. A C++17 compiler is required for the bundled entropy backend. To rebuild that backend from this directory:

```bash
bash scripts/setup_entropy.sh
```

Run the remaining commands from `Quad-LIC`.

## Checkpoints

Filenames, objectives, rate points and file sizes are listed in [checkpoints/manifest.json](checkpoints/manifest.json). Complete checkpoint files are available on [Google Drive](https://drive.google.com/drive/folders/1ZiTYI1vJozInBVBJP_QS19_cVoiphswZ). Download one checkpoint with:

```bash
python scripts/download_checkpoints.py --file QSC_0006_psnr.pt
```

Omit `--file` to download all nine. The downloader uses the links in [checkpoints/downloads.json](checkpoints/downloads.json), checks file sizes, and skips existing files of the expected size. Tensor-only `.pt` checkpoints can also be downloaded in a browser and copied into `checkpoints/` for offline inference.

## Encode and decode Kodak

Use a directory containing the original Kodak RGB images:

```bash
python test_kodak.py --dataset /path/to/Kodak \
  --checkpoint checkpoints/QSC_0006_psnr.pt \
  --device cuda --output results/kodak_psnr_0006.json \
  --streams results/streams --reconstructions results/reconstructions
```

Each image is compressed and decompressed with rANS. Saved streams include a 12-byte height/width/payload-length header. Bpp counts the complete stream over the original pixel count. The encoder replicates the bottom/right edges to multiples of 16; the decoder crops to the original dimensions and clamps to `[0,1]`. JSON reports per-image values and arithmetic means for RGB PSNR, MS-SSIM, bpp and encoding/decoding time. Quality metrics use float reconstructions before PNG rounding.

Pass multiple paths after `--checkpoint` to evaluate several rate points. Use `--device cpu --limit 1 --warmup 0` for a small CPU run. Omit `--streams` or `--reconstructions` when those files are not needed.

Strict float32 is the default. `--allow-tf32` permits cuDNN convolution TF32 while keeping matmul TF32 disabled; metric filters always use full float32. Keep the checkpoint, numerical settings and backend consistent between encoder and decoder. Settings are recorded in the output JSON.

## Complexity and latency

```bash
python benchmark.py --checkpoint checkpoints/QSC_0006_psnr.pt \
  --image /path/to/Kodak/kodim01.png --device cuda --acceleration cuda \
  --warmup 3 --iterations 10 --output results/complexity_latency.json
```

Latency measurement defaults to the CUDA backend and requires a CUDA device and the compiled common extension. Run `bash scripts/setup_cuda.sh` if the extension has not been built. The requested CUDA backend must be available before measurement. Use `--acceleration none` for reference timing; combine it with `--device cpu` for CPU timing.

Timing uses synchronized encode/decode wall time, including neural transforms, CPU rANS and CPU/GPU transfers. It excludes model/CDF setup, padding, image/stream disk I/O and quality metrics. Complexity counts Conv2d, ConvTranspose2d and Linear MACs across the complete roundtrip, with two FLOPs per MAC. The JSON records the counting convention and exclusions.

## Tests and source layout

```bash
python -m pytest -q tests
```

`network.py` defines the codec, `layers.py` its image layers, `common_model.py` the four-pass coding logic, `entropy_model.py` the entropy models, and `checkpoints.py` checkpoint loading. The native backend is in `cpp/`; `setup_entropy.py` builds the top-level `MLCodec_rans` and `MLCodec_CXX` modules. Checkpoints load with `weights_only=True` and strict parameter checks.

Third-party notices are in [third_party/NOTICE.md](third_party/NOTICE.md). The shared release uses the [MIT license](../LICENSE).
