import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from quad_deepsc.checkpoints import load_original, construct, profiles

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=sorted(profiles()),required=True)
    args = p.parse_args()
    model, metadata = load_original(args.input,args.profile)
    state = {k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(state,args.output)
    construct(args.profile).load_state_dict(torch.load(args.output,map_location='cpu',weights_only=True),strict=True)
    metadata.update(profile=args.profile,file=args.output.name,bytes=args.output.stat().st_size,tensors=len(state))
    args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)

if __name__ == '__main__':
    main()
