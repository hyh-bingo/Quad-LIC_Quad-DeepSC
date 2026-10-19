from pathlib import Path
from setuptools import Extension, setup
import pybind11


ROOT = Path(__file__).resolve().parent
common = [str(ROOT / 'cpp/rans'), str(ROOT / 'cpp/py_rans'), str(ROOT / 'third_party/ryg_rans'), pybind11.get_include()]
setup(name='quad-lic-entropy', version='1.0.0', packages=[], ext_modules=[
    Extension('MLCodec_rans', ['cpp/py_rans/py_rans.cpp', 'cpp/rans/rans.cpp'], include_dirs=common, language='c++', extra_compile_args=['-O3', '-std=c++17', '-pthread'], extra_link_args=['-pthread']),
    Extension('MLCodec_CXX', ['cpp/ops/ops.cpp'], include_dirs=common, language='c++', extra_compile_args=['-O3', '-std=c++17']),
])
