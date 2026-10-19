import os
from setuptools import find_packages, setup

extensions = []
commands = {}
if os.environ.get("QUAD_BUILD_CUDA") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    extensions = [
        CUDAExtension(
            "quad_acceleration._cuda",
            ["csrc/binding.cpp", "csrc/residual.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--fmad=false"]},
        )
    ]
    commands = {"build_ext": BuildExtension}

setup(
    name="quad-acceleration",
    version="0.1.2",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch>=2.1"],
    ext_modules=extensions,
    cmdclass=commands,
)
