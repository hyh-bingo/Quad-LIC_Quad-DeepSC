import pytest
import torch
from runtime import pack_stream, unpack_stream, roundtrip
from network import QSC
from checkpoints import load_checkpoint, export_checkpoint


def test_container_validation():
    assert unpack_stream(pack_stream(19, 31, b'abcd')) == (19, 31, b'abcd')
    for value in (b'', b'123', pack_stream(19, 31, b'abcd')[:-1]):
        with pytest.raises(ValueError):
            unpack_stream(value)


def test_checkpoint_compatibility_and_native_image_roundtrip(tmp_path):
    torch.manual_seed(9)
    torch.set_num_threads(1)
    model = QSC(N=8, M=16).eval().requires_grad_(False)
    state = {'module.' + key: value for key, value in model.state_dict().items()}
    original = tmp_path / 'wrapped.pt'
    torch.save({'state_dict': state, 'epoch': 12, 'optimizer': {}}, original)
    clean = tmp_path / 'inference.pt'
    export_checkpoint(original, clean)
    restored, metadata = load_checkpoint(clean)
    assert metadata['N'] == 8 and metadata['M'] == 16
    assert set(restored.state_dict()) == set(model.state_dict())
    restored.update(force=True)
    x = torch.rand(1, 3, 65, 79)
    reconstruction, stream, timing = roundtrip(restored, x)
    assert reconstruction.shape == x.shape
    assert torch.isfinite(reconstruction).all()
    assert reconstruction.min() >= 0 and reconstruction.max() <= 1
    height, width, payload = unpack_stream(stream)
    decoded_again = restored.decompress(payload, height, width)['x_hat'][..., :height, :width]
    torch.testing.assert_close(decoded_again, reconstruction, rtol=0, atol=0)
    assert len(stream) > 12 and all(seconds >= 0 for seconds in timing.values())
    with pytest.raises(RuntimeError):
        broken = tmp_path / 'broken.pt'
        torch.save({'g_a.0.weight': state['module.g_a.0.weight'], 'g_a.13.weight': state['module.g_a.13.weight']}, broken)
        load_checkpoint(broken)
