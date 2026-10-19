import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from quad_deepsc.model import QJSCC
from quad_deepsc.source import gaussian_likelihood
from quad_deepsc.checkpoints import profiles

def test_cpu_resolution_changes_and_png_roundtrip(tmp_path):
    import numpy as np
    from PIL import Image
    from evaluate import run_frame
    torch.set_num_threads(1)
    model = QJSCC(N=8,M=320,depths=[2,2]).eval()
    source = tmp_path/'image.png'
    Image.fromarray(np.full((65,97,3),128,dtype=np.uint8)).save(source)
    with torch.inference_mode():
        x, y, result = run_frame(model,source,'cpu',tmp_path/'side.png')
    assert x.shape == y.shape == (1,3,65,97)
    assert torch.isfinite(y).all()
    assert result['side_bits'] > 0
    assert result['cbr'] >= result['cbr_main'] > 0

def test_release_profile_scope():
    p = profiles()
    assert len(p) == 17
    assert {c['snr_db'] for c in p.values() if c['metric'] != 'lpips'} == {10}
    assert 'lpips_awgn6_0005' not in p
    assert p['lpips_awgn10_0005']['rate_shift'] == 5

def test_zero_symbol_operating_point_is_finite():
    from quad_deepsc.channel import AWGN_channel
    channel=AWGN_channel(SNR=10)
    output,_=channel(torch.zeros(1,4,4,4))
    assert torch.equal(output,torch.zeros_like(output))
    output,usage,_=QJSCC.symbol_pass_channel(None,torch.zeros(1,4,4,4),torch.zeros(1,4,4,4))
    assert usage.item()==0
    assert torch.isfinite(output).all()

def test_gaussian_negative_tail_is_finite_and_reproducible():
    x=torch.tensor([0.,1.,50.,-50.])
    scales=torch.tensor([0.,1.,.11,.11])
    torch.manual_seed(123)
    first=gaussian_likelihood(x,scales,torch.zeros_like(x))
    torch.manual_seed(123)
    second=gaussian_likelihood(x,scales,torch.zeros_like(x))
    assert torch.equal(first,second)
    assert torch.isfinite(first).all()
    assert torch.all(first>=1e-9)
    assert first[2] == first[3] == 1e-9
