set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup.sh
if [ "${CPU_ONLY:-0}" != "1" ]; then
    bash scripts/setup_cuda.sh
fi
