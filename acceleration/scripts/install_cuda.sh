set -euo pipefail
acceleration_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export QUAD_BUILD_CUDA=1
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; assert torch.cuda.is_available(), "CUDA GPU unavailable"; assert CUDA_HOME, "CUDA toolkit/nvcc unavailable"; print({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_home": CUDA_HOME, "device": torch.cuda.get_device_name(0)})'
python -m pip install 'ninja>=1.11' 'pytest>=7' 'setuptools>=68' wheel
python -m pip install --no-build-isolation "$acceleration_root"
python -c 'from quad_acceleration import backend_status; status = backend_status(); print(status); assert status["cuda_extension_loaded"], status'
python -m pytest -q "$acceleration_root/tests" --require-cuda
