set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
bash acceleration/scripts/install_cuda.sh
