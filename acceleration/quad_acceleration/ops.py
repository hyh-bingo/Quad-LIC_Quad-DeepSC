import importlib

import torch
import torch.nn.functional as F

_diagnostics_enabled = False
_runtime_stats = {
    "cuda_operator_calls": 0,
    "pytorch_fallback_calls": 0,
    "cuda_layout_calls": {"default_contiguous": 0, "channels_last_4d": 0},
    "sdpa_api_calls": 0,
    "original_attention_forward_calls": 0,
    "relative_bias_cache_hits": 0,
    "relative_bias_cache_misses": 0,
    "relative_bias_cache_bypasses": 0,
}

try:
    _extension = importlib.import_module("quad_acceleration._cuda")
    _load_error = None
except (ImportError, OSError) as error:
    _extension = None
    _load_error = str(error)


def backend_status():
    layouts = ["default_contiguous"] if _extension is not None else []
    if getattr(_extension, "supports_channels_last", False):
        layouts.append("channels_last_4d")
    collected = any(value > 0 for value in _runtime_stats.values() if isinstance(value, int))
    return {
        "package_version": "0.1.2",
        "cuda_extension_version": getattr(_extension, "version", "0.1.0") if _extension is not None else None,
        "cuda_extension_loaded": _extension is not None,
        "cuda_extension_error": _load_error,
        "cuda_device_available": torch.cuda.is_available(),
        "sdpa_available": hasattr(F, "scaled_dot_product_attention"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_supported_layouts": layouts,
        "cuda_stride_requirement": "Input and residual must have identical strides",
        "sdpa_bias_layout": "Last dimension contiguous",
        "sdpa_cache_policy": "One relative-position bias per module; version/device/dtype/shape/stride/stream keyed; no full shifted-mask cache",
        "runtime_diagnostics": {
            "enabled": _diagnostics_enabled,
            "collected": collected,
            "scope": "Process-global calls and relative-cache events; SDPA API counts do not identify the selected backend; null counts mean no diagnostic observation",
            "cuda_operator_calls": _runtime_stats["cuda_operator_calls"] if collected else None,
            "pytorch_fallback_calls": _runtime_stats["pytorch_fallback_calls"] if collected else None,
            "cuda_layout_calls": dict(_runtime_stats["cuda_layout_calls"]) if collected else None,
            **{key: value if collected else None for key, value in _runtime_stats.items() if key not in ("cuda_operator_calls", "pytorch_fallback_calls", "cuda_layout_calls")},
        },
    }


def set_diagnostics(enabled=True, reset=True):
    global _diagnostics_enabled
    _diagnostics_enabled = bool(enabled)
    if reset:
        for key, value in _runtime_stats.items():
            if isinstance(value, dict):
                for nested in value:
                    value[nested] = 0
            else:
                _runtime_stats[key] = 0


def _record_attention(event):
    if _diagnostics_enabled:
        _runtime_stats[event] += 1


def _matching_layout(x, identity):
    if x.stride() != identity.stride():
        return False
    if x.is_contiguous() and identity.is_contiguous():
        return True
    return (
        getattr(_extension, "supports_channels_last", False)
        and x.ndim == 4
        and x.is_contiguous(memory_format=torch.channels_last)
        and identity.is_contiguous(memory_format=torch.channels_last)
    )


def leaky_relu_add(x, identity, negative_slope=0.1):
    eligible = (
        _extension is not None
        and x.is_cuda
        and identity.is_cuda
        and not torch.is_grad_enabled()
        and x.device == identity.device
        and x.shape == identity.shape
        and x.dtype == identity.dtype
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        and _matching_layout(x, identity)
    )
    if eligible:
        output = _extension.leaky_relu_add(x, identity, float(negative_slope))
        if _diagnostics_enabled:
            _runtime_stats["cuda_operator_calls"] += 1
            layout = "default_contiguous" if x.is_contiguous() else "channels_last_4d"
            _runtime_stats["cuda_layout_calls"][layout] += 1
        return output
    if _diagnostics_enabled:
        _runtime_stats["pytorch_fallback_calls"] += 1
    return F.leaky_relu(x, negative_slope=float(negative_slope)) + identity
