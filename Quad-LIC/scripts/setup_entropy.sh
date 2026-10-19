set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install pybind11==3.0.1 setuptools==80.9.0 wheel
python setup_entropy.py build_ext --inplace
python -m pytest -q tests/test_entropy.py
