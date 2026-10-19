import pytest
import torch

from quad_acceleration import __version__, backend_status


def pytest_addoption(parser):
    parser.addoption("--require-cuda", action="store_true")


@pytest.fixture(scope="session", autouse=True)
def require_cuda(request):
    if request.config.getoption("--require-cuda"):
        status = backend_status()
        if not torch.cuda.is_available() or not status["cuda_extension_loaded"]:
            pytest.fail(f"CUDA verification requested but backend unavailable: {status}")
        if status["cuda_extension_version"] != __version__ or "channels_last_4d" not in status["cuda_supported_layouts"]:
            pytest.fail(f"CUDA verification requires the current {__version__} extension: {status}")
