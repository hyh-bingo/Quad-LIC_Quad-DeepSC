set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHON="${PYTHON:-${PYTHON_BIN:-python3}}"
bash scripts/setup_all.sh
