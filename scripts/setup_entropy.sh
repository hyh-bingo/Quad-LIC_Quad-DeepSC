set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x "$PWD/.venv/bin/python" ]; then
    export PATH="$PWD/.venv/bin:$PATH"
fi
bash Quad-LIC/scripts/setup_entropy.sh
