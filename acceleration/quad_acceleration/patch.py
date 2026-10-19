from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F

from .ops import _record_attention, backend_status, leaky_relu_add


def _inference(module):
    return not module.training and not torch.is_grad_enabled()


def _stamp(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        return None
    return (id(tensor), version, tuple(tensor.shape), tuple(tensor.stride()), tensor.device, tensor.dtype)


def _relative_bias(module, q, k):
    query_length, key_length, heads = q.shape[-2], k.shape[-2], q.shape[1]
    table, index = module.relative_position_bias_table, module.relative_position_index
    if index.numel() != query_length * key_length:
        raise ValueError("Relative position index does not match attention token lengths")
    table_stamp, index_stamp = _stamp(table), _stamp(index)
    key = None
    if table_stamp is not None and index_stamp is not None:
        stream = torch.cuda.current_stream(q.device).cuda_stream if q.is_cuda else None
        key = (table_stamp, index_stamp, q.device, q.dtype, query_length, key_length, heads, stream)
        cache = module._quad_relative_bias_cache
        if cache.get("key") == key:
            _record_attention("relative_bias_cache_hits")
            return cache["bias"]
        _record_attention("relative_bias_cache_misses")
    else:
        _record_attention("relative_bias_cache_bypasses")
        module._quad_relative_bias_cache = {}
    bias = table[index.reshape(-1)].view(query_length, key_length, heads).permute(2, 0, 1).unsqueeze(0)
    bias = bias.to(device=q.device, dtype=q.dtype).contiguous()
    if key is not None:
        module._quad_relative_bias_cache = {"key": key, "bias": bias, "sources": (table, index)}
    return bias


def _bias(module, q, k, mask):
    batch, heads, query_length, _ = q.shape
    key_length = k.shape[-2]
    bias = None
    if getattr(module, "relative_pos_embedding", True) and hasattr(module, "relative_position_bias_table"):
        bias = _relative_bias(module, q, k)
    if mask is not None:
        windows = mask.shape[0]
        if not windows or batch % windows or tuple(mask.shape[1:]) != (query_length, key_length):
            raise ValueError("Shifted-window mask does not match attention token lengths")
        shifted = mask.to(device=q.device, dtype=q.dtype)
        shifted = shifted.unsqueeze(0).expand(batch // windows, -1, -1, -1)
        shifted = shifted.reshape(batch, 1, query_length, key_length)
        bias = shifted if bias is None else bias + shifted
    return bias.contiguous() if bias is not None else None


def _window_forward(self, x, mask=None):
    if not _inference(self):
        _record_attention("original_attention_forward_calls")
        return self._quad_original_forward(x, mask=mask)
    batch, length, channels = x.shape
    qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, channels // self.num_heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=_bias(self, q, k, mask), dropout_p=0.0, scale=float(self.scale)
    )
    _record_attention("sdpa_api_calls")
    output = output.transpose(1, 2).reshape(batch, length, channels)
    return self.proj_drop(self.proj(output))


def _cross_forward(self, x1, x2, mask=None):
    if not _inference(self):
        _record_attention("original_attention_forward_calls")
        return self._quad_original_forward(x1, x2, mask=mask)
    batch, length, channels = x1.shape
    q = self.q(x1).reshape(batch, length, self.num_heads, channels // self.num_heads)
    q = q.permute(0, 2, 1, 3)
    kv = self.kv(x2).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)
    k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=_bias(self, q, k, mask), dropout_p=0.0, scale=float(self.scale)
    )
    _record_attention("sdpa_api_calls")
    output = output.transpose(1, 2).reshape(batch, length, channels)
    return self.proj_drop(self.proj(output))


def _ffn_forward(self, x, identity=None):
    if not _inference(self):
        return self._quad_original_forward(x, identity)
    if identity is None:
        identity = x
    output = self.conv[0](x)
    output = self.conv[1](output)
    output = self.conv[2](output)
    return leaky_relu_add(output, identity, self.conv[3].negative_slope)


def _replace(module, forward):
    if hasattr(module, "_quad_original_forward"):
        return False
    module._quad_original_forward = module.forward
    module.forward = MethodType(forward, module)
    return True


def configure(model, cuda=True, sdpa=True):
    status = backend_status()
    counts = {"dc_ffn_patched": 0, "window_attention_patched": 0, "cross_attention_patched": 0}
    for module in model.modules():
        name = type(module).__name__
        if cuda and name == "ConvFFN" and isinstance(getattr(module, "conv", None), nn.Sequential):
            conv = module.conv
            if len(conv) == 4 and isinstance(conv[0], nn.Conv2d) and isinstance(conv[1], nn.LeakyReLU):
                if isinstance(conv[2], nn.Conv2d) and isinstance(conv[3], nn.LeakyReLU):
                    counts["dc_ffn_patched"] += int(_replace(module, _ffn_forward))
        if not sdpa or not status["sdpa_available"]:
            continue
        common = ("num_heads", "scale", "proj", "proj_drop")
        if not all(hasattr(module, attr) for attr in common):
            continue
        if name == "WindowAttention" and hasattr(module, "qkv"):
            if _replace(module, _window_forward):
                module._quad_relative_bias_cache = {}
                counts["window_attention_patched"] += 1
        elif name == "WindowCrossAttention" and hasattr(module, "q") and hasattr(module, "kv"):
            if _replace(module, _cross_forward):
                module._quad_relative_bias_cache = {}
                counts["cross_attention_patched"] += 1
    return {
        **status, **counts, "cuda_requested": cuda, "sdpa_requested": sdpa,
        "patch_count_definition": "Installed forward-method wrappers, not observed CUDA operator calls",
    }


def attention_cache_status(model):
    modules = []
    for name, module in model.named_modules():
        if not hasattr(module, "_quad_relative_bias_cache"):
            continue
        cache = module._quad_relative_bias_cache
        bias = cache.get("bias")
        modules.append({
            "name": name,
            "cached": bias is not None,
            "bytes": bias.numel() * bias.element_size() if bias is not None else 0,
            "shape": list(bias.shape) if bias is not None else None,
        })
    return {
        "policy": "Relative-position bias only; one entry per patched attention module",
        "module_count": len(modules),
        "cached_modules": sum(item["cached"] for item in modules),
        "resident_bytes": sum(item["bytes"] for item in modules),
        "modules": modules,
    }


def restore(model):
    count = 0
    for module in model.modules():
        if hasattr(module, "_quad_original_forward"):
            module.forward = module._quad_original_forward
            delattr(module, "_quad_original_forward")
            if hasattr(module, "_quad_relative_bias_cache"):
                delattr(module, "_quad_relative_bias_cache")
            count += 1
    return count
