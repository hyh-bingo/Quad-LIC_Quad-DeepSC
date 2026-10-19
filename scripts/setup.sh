set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python3}"
"$python_bin" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), "Use Python 3.11–3.13 (recommended 3.12)"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
torch_wheel_flavor="${TORCH_WHEEL_FLAVOR:-cu118}"
if [ "${CPU_ONLY:-0}" = "1" ]; then
    torch_wheel_flavor=cpu
fi
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$torch_wheel_flavor"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch, png, lpips, pytorch_msssim; print(torch.__version__, torch.cuda.is_available())'

.venv/bin/python -m pip install --no-build-isolation ./acceleration
bash scripts/setup_entropy.sh
.venv/bin/python -m pytest -q tests
(
    cd Quad-LIC
    ../.venv/bin/python -m pytest -q tests
)
