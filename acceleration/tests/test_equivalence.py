import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from quad_acceleration import attention_cache_status, backend_status, configure, leaky_relu_add, restore, set_diagnostics
from quad_acceleration import ops


class WindowAttention(nn.Module):
    def __init__(self, cross=False, relative=True):
        super().__init__()
        self.num_heads = 2
        self.scale = 0.37
        self.relative_pos_embedding = relative
        self.proj = nn.Linear(16, 16)
        self.proj_drop = nn.Dropout(0.0)
        self.attn_drop = nn.Dropout(0.0)
        if relative:
            self.relative_position_bias_table = nn.Parameter(torch.randn(49, 2) * 0.1)
            coords = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij"))
            positions = coords.flatten(1)
            offsets = positions[:, :, None] - positions[:, None, :]
            self.register_buffer("relative_position_index", (offsets[0] + 3) * 7 + offsets[1] + 3)
        if cross:
            self.q = nn.Linear(16, 16)
            self.kv = nn.Linear(16, 32)
        else:
            self.qkv = nn.Linear(16, 48)

    def _attention(self, q, k, v, mask):
        batch, heads, length, _ = q.shape
        keys = k.shape[-2]
        scores = (q * self.scale) @ k.transpose(-2, -1)
        if self.relative_pos_embedding:
            bias = self.relative_position_bias_table[self.relative_position_index.flatten()]
            scores = scores + bias.view(length, keys, heads).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            scores = scores.view(batch // windows, windows, heads, length, keys)
            scores = (scores + mask[None, :, None]).view(batch, heads, length, keys)
        result = self.attn_drop(scores.softmax(-1)) @ v
        return self.proj_drop(self.proj(result.transpose(1, 2).reshape(batch, length, 16)))

    def forward(self, x, mask=None):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, 2, channels // 2)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return self._attention(q, k, v, mask)


class WindowCrossAttention(WindowAttention):
    def __init__(self, relative=True):
        super().__init__(cross=True, relative=relative)

    def forward(self, x1, x2, mask=None):
        batch, length, channels = x1.shape
        q = self.q(x1).reshape(batch, length, 2, channels // 2).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(batch, -1, 2, 2, channels // 2)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        return self._attention(q, k, v, mask)


class ConvFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, 1), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 8, 1), nn.LeakyReLU(0.1),
        )

    def forward(self, x, identity=None):
        return (x if identity is None else identity) + self.conv(x)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["self", "cross", "cross_multiple"])
@pytest.mark.parametrize("shifted", [False, True])
def test_sdpa_equivalence(device, kind, shifted):
    torch.manual_seed(34)
    module = WindowAttention() if kind == "self" else WindowCrossAttention(relative=kind != "cross_multiple")
    module = module.to(device).eval()
    original = copy.deepcopy(module)
    keys_before = tuple(module.state_dict())
    x = torch.randn(6, 16, 16, device=device)
    key_length = 32 if kind == "cross_multiple" else 16
    y = torch.randn(6, key_length, 16, device=device)
    mask = None
    if shifted:
        mask = torch.zeros(3, 16, key_length, device=device)
        mask[:, :8, key_length // 2:] = -100.0
        mask[:, 8:, :key_length // 2] = -100.0
    inputs = (x,) if kind == "self" else (x, y)
    report = configure(module, cuda=False)
    assert report["window_attention_patched"] + report["cross_attention_patched"] == 1
    assert tuple(module.state_dict()) == keys_before
    with torch.inference_mode():
        expected = original(*inputs, mask=mask)
        actual = module(*inputs, mask=mask)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert restore(module) == 1
    assert restore(module) == 0
    with torch.inference_mode():
        torch.testing.assert_close(module(*inputs, mask=mask), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("external_identity", [False, True])
def test_ffn_preserves_residual_order(device, external_identity):
    torch.manual_seed(35)
    module = ConvFFN().to(device).eval()
    original = copy.deepcopy(module)
    x = torch.randn(2, 8, 11, 13, device=device)
    identity = torch.randn_like(x) if external_identity else None
    before = tuple(module.state_dict())
    report = configure(module, sdpa=False)
    assert report["dc_ffn_patched"] == 1
    assert tuple(module.state_dict()) == before
    with torch.inference_mode():
        expected = original(x, identity)
        actual = module(x, identity)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_grad_enabled_falls_back():
    module = ConvFFN().eval()
    original = copy.deepcopy(module)
    configure(module)
    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    calls = []
    forward = module._quad_original_forward

    def fallback(*args, **kwargs):
        calls.append(True)
        return forward(*args, **kwargs)

    module._quad_original_forward = fallback
    actual = module(x)
    assert calls == [True]
    assert actual.requires_grad
    torch.testing.assert_close(actual, original(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("channels_last", [False, True])
def test_cuda_kernel_uses_current_stream(dtype, channels_last, monkeypatch):
    if not torch.cuda.is_available() or not backend_status()["cuda_extension_loaded"]:
        pytest.skip("CUDA extension unavailable")
    stream = torch.cuda.Stream()
    native = ops._extension.leaky_relu_add
    calls = []

    def observed(*args):
        calls.append(True)
        return native(*args)

    monkeypatch.setattr(ops._extension, "leaky_relu_add", observed)
    with torch.cuda.stream(stream), torch.inference_mode():
        x = torch.randn(2, 32, 31, 29, device="cuda", dtype=dtype)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        identity = torch.randn_like(x)
        expected = F.leaky_relu(x, negative_slope=0.1) + identity
        actual = leaky_relu_add(x, identity, 0.1)
        empty = leaky_relu_add(x[:0], identity[:0])
    stream.synchronize()
    assert calls == [True, True]
    assert actual.stride() == x.stride()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert empty.shape == x[:0].shape


def test_cuda_mismatched_strides_fall_back_and_native_rejects(monkeypatch):
    if not torch.cuda.is_available() or not backend_status()["cuda_extension_loaded"]:
        pytest.skip("CUDA extension unavailable")
    x = torch.randn(2, 8, 7, 5, device="cuda").contiguous(memory_format=torch.channels_last)
    identity = torch.randn(x.shape, device="cuda")
    assert x.stride() != identity.stride()
    native = ops._extension.leaky_relu_add

    def unexpected(*args):
        raise AssertionError("Mismatched strides must not call the native operator")

    monkeypatch.setattr(ops._extension, "leaky_relu_add", unexpected)
    with torch.inference_mode():
        actual = leaky_relu_add(x, identity)
        expected = F.leaky_relu(x, 0.1) + identity
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="strides"):
            native(x, identity, 0.1)


def test_cuda_unsupported_dense_layout_is_rejected():
    if not torch.cuda.is_available() or not backend_status()["cuda_extension_loaded"]:
        pytest.skip("CUDA extension unavailable")
    x = torch.randn(2, 8, 7, 5, device="cuda").transpose(2, 3)
    identity = torch.randn_like(x)
    assert x.stride() == identity.stride()
    with torch.inference_mode():
        torch.testing.assert_close(leaky_relu_add(x, identity), F.leaky_relu(x, 0.1) + identity, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="contiguous"):
            ops._extension.leaky_relu_add(x, identity, 0.1)


@pytest.mark.parametrize("device", DEVICES)
def test_opt_in_runtime_diagnostics(device):
    x = torch.randn(2, 8, 7, 5, device=device).contiguous(memory_format=torch.channels_last)
    identity = torch.randn_like(x)
    set_diagnostics(False)
    status = backend_status()["runtime_diagnostics"]
    assert status["enabled"] is False and status["collected"] is False
    assert status["cuda_operator_calls"] is None
    try:
        set_diagnostics(True)
        with torch.inference_mode():
            leaky_relu_add(x, identity)
        status = backend_status()["runtime_diagnostics"]
        assert status["collected"] is True
        if device == "cuda" and "channels_last_4d" in backend_status()["cuda_supported_layouts"]:
            assert status["cuda_operator_calls"] == 1
            assert status["pytorch_fallback_calls"] == 0
            assert status["cuda_layout_calls"]["channels_last_4d"] == 1
        else:
            assert status["cuda_operator_calls"] == 0
            assert status["pytorch_fallback_calls"] == 1
    finally:
        set_diagnostics(False)


def test_noncontiguous_fallback():
    x = torch.randn(3, 5).transpose(0, 1)
    identity = torch.randn_like(x)
    actual = leaky_relu_add(x, identity)
    torch.testing.assert_close(actual, F.leaky_relu(x, negative_slope=0.1) + identity)


def test_attention_grad_enabled_falls_back():
    module = WindowAttention().eval()
    original = copy.deepcopy(module)
    configure(module, cuda=False)
    x = torch.randn(1, 16, 16, requires_grad=True)
    calls = []
    forward = module._quad_original_forward

    def fallback(*args, **kwargs):
        calls.append(True)
        return forward(*args, **kwargs)

    module._quad_original_forward = fallback
    actual = module(x)
    assert calls == [True]
    assert actual.requires_grad
    torch.testing.assert_close(actual, original(x), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cross", [False, True])
def test_relative_cache_invalidation_diagnostics_and_restore(device, cross):
    torch.manual_seed(43)
    module = (WindowCrossAttention() if cross else WindowAttention()).to(device).eval()
    original = copy.deepcopy(module)
    x, y = torch.randn(6, 16, 16, device=device), torch.randn(6, 16, 16, device=device)
    mask = torch.zeros(3, 16, 16, device=device)
    mask[:, :8, 8:] = -100
    inputs = (x, y) if cross else (x,)
    keys = tuple(module.state_dict())
    configure(module, cuda=False)
    try:
        set_diagnostics(True)
        with torch.inference_mode():
            expected = original(*inputs, mask=mask)
            module(*inputs, mask=mask)
            torch.testing.assert_close(module(*inputs, mask=mask), expected, rtol=3e-5, atol=3e-6)
            diagnostics = backend_status()["runtime_diagnostics"]
            assert diagnostics["sdpa_api_calls"] == 2
            assert diagnostics["relative_bias_cache_misses"] == 1
            assert diagnostics["relative_bias_cache_hits"] == 1
            assert diagnostics["relative_bias_cache_bypasses"] == 0
            info = attention_cache_status(module)
            assert info["cached_modules"] == 1 and info["resident_bytes"] == 2048
            module.relative_position_bias_table[:, 0].add_(torch.linspace(-0.5, 0.5, 49, device=device))
            original.load_state_dict(module.state_dict(), strict=True)
            torch.testing.assert_close(module(*inputs, mask=mask), original(*inputs, mask=mask), rtol=3e-5, atol=3e-6)
            module.relative_position_index.copy_(module.relative_position_index.flip(0))
            original.load_state_dict(module.state_dict(), strict=True)
            torch.testing.assert_close(module(*inputs, mask=mask), original(*inputs, mask=mask), rtol=3e-5, atol=3e-6)
            mask[:, 8:, :8] = -100
            torch.testing.assert_close(module(*inputs, mask=mask), original(*inputs, mask=mask), rtol=3e-5, atol=3e-6)
            original.relative_position_bias_table.mul_(-0.7)
            module.load_state_dict(original.state_dict(), strict=True)
            torch.testing.assert_close(module(*inputs, mask=mask), original(*inputs, mask=mask), rtol=3e-5, atol=3e-6)
            assert tuple(module.state_dict()) == keys
            assert restore(module) == 1
            assert attention_cache_status(module)["resident_bytes"] == 0
            torch.testing.assert_close(module(*inputs, mask=mask), original(*inputs, mask=mask), rtol=0, atol=0)
    finally:
        set_diagnostics(False)


def test_relative_cache_does_not_reuse_unversioned_sources():
    with torch.inference_mode():
        module = WindowAttention().eval()
        original = copy.deepcopy(module)
        x = torch.randn(6, 16, 16)
        configure(module, cuda=False)
        try:
            set_diagnostics(True)
            module(x)
            module(x)
            module.relative_position_bias_table[:, 0].add_(torch.linspace(-0.5, 0.5, 49))
            original.load_state_dict(module.state_dict(), strict=True)
            torch.testing.assert_close(module(x), original(x), rtol=3e-5, atol=3e-6)
            diagnostics = backend_status()["runtime_diagnostics"]
            assert diagnostics["relative_bias_cache_bypasses"] == 3
            assert diagnostics["relative_bias_cache_hits"] == 0
            assert attention_cache_status(module)["resident_bytes"] == 0
        finally:
            set_diagnostics(False)


def test_relative_cache_tracks_dtype_changes():
    module = WindowAttention().eval()
    original = copy.deepcopy(module)
    configure(module, cuda=False)
    x = torch.randn(6, 16, 16)
    with torch.inference_mode():
        module(x)
        module.double()
        original.double()
        torch.testing.assert_close(module(x.double()), original(x.double()), rtol=3e-5, atol=3e-6)
        assert attention_cache_status(module)["resident_bytes"] == 4096


def test_relative_cache_is_scoped_to_current_cuda_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = WindowAttention().cuda().eval()
    original = copy.deepcopy(module)
    x = torch.randn(6, 16, 16, device="cuda")
    configure(module, cuda=False)
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    try:
        set_diagnostics(True)
        with torch.inference_mode():
            expected = original(x)
            first.wait_stream(torch.cuda.current_stream())
            second.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(first):
                a = module(x)
            with torch.cuda.stream(second):
                b = module(x)
        first.synchronize()
        second.synchronize()
        torch.testing.assert_close(a, expected, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(b, expected, rtol=3e-5, atol=3e-6)
        diagnostics = backend_status()["runtime_diagnostics"]
        assert diagnostics["relative_bias_cache_misses"] == 2
        assert diagnostics["relative_bias_cache_hits"] == 0
    finally:
        set_diagnostics(False)
