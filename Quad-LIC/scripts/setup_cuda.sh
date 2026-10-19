set -euo pipefail
cd "$(dirname "$0")/.."
QUAD_CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TORCH_CUDA_ARCH_LIST="$QUAD_CUDA_ARCH"
bash ../scripts/setup_cuda.sh
