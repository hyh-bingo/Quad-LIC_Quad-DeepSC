import argparse
import subprocess
import sys
from pathlib import Path
from quad_deepsc.checkpoints import profiles

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/Kodak'))
    parser.add_argument('--profiles',nargs='+',choices=sorted(profiles()))
    args,extra=parser.parse_known_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for profile in args.profiles or profiles():
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'--dataset',str(args.dataset),'--profile',profile,'--output',str(args.output/(profile+'.json')),*extra],check=True)

if __name__=='__main__':
    main()
