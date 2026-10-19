from collections.abc import Mapping
from pathlib import Path
import re
import torch


def normalize_state_dict(checkpoint):
    state = checkpoint
    for _ in range(6):
        if not isinstance(state, Mapping):
            raise TypeError('Checkpoint must contain a tensor state dictionary')
        wrapper = next((key for key in ('state_dict', 'model_state_dict', 'model', 'network', 'net', 'params') if key in state and isinstance(state[key], Mapping)), None)
        if wrapper is None:
            break
        state = state[wrapper]
    cleaned = {}
    ignored = []
    regenerated = {'entropy_bottleneck._quantized_cdf', 'entropy_bottleneck._offset', 'entropy_bottleneck._cdf_length', 'gaussian_conditional._quantized_cdf', 'gaussian_conditional._offset', 'gaussian_conditional._cdf_length', 'gaussian_conditional.scale_table', 'gaussian_conditional.scale_bound', 'entropy_bottleneck.target', 'entropy_bottleneck.likelihood_lower_bound.bound', 'gaussian_conditional.likelihood_lower_bound.bound', 'gaussian_conditional.lower_bound_scale.bound'}
    for key, value in state.items():
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        match = re.fullmatch(r'entropy_bottleneck\._(biases|matrices|factors)\.(\d+)', key)
        if match:
            key = f'entropy_bottleneck._{dict(biases="bias", matrices="matrix", factors="factor")[match[1]]}{match[2]}'
        if key in regenerated:
            ignored.append(key)
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Non-tensor checkpoint entry: {key}')
        if key in cleaned:
            raise ValueError(f'Duplicate normalized key: {key}')
        cleaned[key] = value.detach().cpu()
    if not cleaned:
        raise ValueError('Empty model state dictionary')
    return cleaned, sorted(ignored)


def load_checkpoint(path, device='cpu'):
    from network import QSC
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state, ignored = normalize_state_dict(checkpoint)
    n = int(state['g_a.0.weight'].shape[0])
    m = int(state['g_a.13.weight'].shape[0])
    model = QSC(N=n, M=m)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, {'checkpoint': str(path), 'N': n, 'M': m, 'regenerated_or_removed_keys': ignored}


def strict_load_original(path):
    return load_checkpoint(path, device='cpu')


def export_checkpoint(source, destination):
    model, metadata = strict_load_original(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    reloaded, _ = load_checkpoint(destination)
    for key, value in model.state_dict().items():
        if not torch.equal(value, reloaded.state_dict()[key]):
            raise RuntimeError(f'Export roundtrip mismatch: {key}')
    return metadata
