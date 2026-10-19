# Inference operators

Install into the active Python environment:

```bash
python -m pip install --no-build-isolation ./acceleration
```

Build the native extension with `bash scripts/setup_cuda.sh` from the repository root, using a compatible CUDA toolkit and `TORCH_CUDA_ARCH_LIST` for the target GPU.
