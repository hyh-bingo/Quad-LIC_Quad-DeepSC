import json
from pathlib import Path
import torch
from .model import QJSCC

MANIFEST = Path(__file__).resolve().parents[1] / 'checkpoints/manifest.json'

def profiles():
    return json.loads(MANIFEST.read_text())['profiles']

def construct(profile):
    cfg = profiles()[profile]
    model = QJSCC(rate_type=cfg['rate_type'], rate_shift=cfg['rate_shift'], eta=cfg['eta'], SNR=cfg['snr_db'], distortion_type=cfg['metric'])
    return model.eval().requires_grad_(False)

def load_original(path, profile):
    model = construct(profile)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    loaded = loaded.get('state_dict', loaded)
    state = {k.removeprefix('module.'): v for k, v in loaded.items()}
    regenerated = {k for k in model.state_dict() if k.endswith('attn_mask') or k == 'rate_adaption_enc.mask'}
    omitted = []
    for key in list(state):
        if key.startswith(('source_model.entropy_bottleneck.', 'source_model.gaussian_conditional.')) or key in regenerated:
            omitted.append(key)
            del state[key]
    for key in regenerated:
        state[key] = model.state_dict()[key]
    expected = model.rate_choice_tensor_new.clone()
    for key in ['rate_choice_tensor_new', 'rate_adaption_enc.rate_choice_tensor_new']:
        if not torch.equal(state[key], expected):
            raise ValueError(f'Rate table mismatch: {key}')
    model.load_state_dict(state, strict=True)
    return model.eval(), {'omitted_keys': sorted(omitted)}

def load(profile, root=None, device='cuda'):
    cfg = profiles()[profile]
    root = Path(root) if root else MANIFEST.parent
    path = root / cfg['checkpoint']
    if not path.is_file():
        raise FileNotFoundError(f'{path}: obtain this checkpoint from the Google Drive release in README.md')
    model = construct(profile)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.rate_choice_tensor_new.cpu(), torch.tensor(model.rate_choice_new)):
        raise ValueError('Checkpoint rate table does not match its profile')
    return model.to(device).eval(), cfg
