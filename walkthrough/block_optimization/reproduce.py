"""Build and check one full block; optionally run the existing RTL simulator.

Example (fresh result directory required):
  model2MLIR/.venv/bin/python reproduce.py --name block_repeat --rtl
Use --compiler compiler for the frozen baseline, or --groups 0 for host-only.
RTL takes roughly five minutes on one core for the default captured block.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--compiler',choices=('compiler','compiler_optimized'),default='compiler_optimized')
    parser.add_argument('--capture',type=Path,default=REPO/'walkthrough/transformer_layer/capture')
    parser.add_argument('--groups',choices=('0','1'))
    parser.add_argument('--rtl',action='store_true')
    args=parser.parse_args()
    args.capture=args.capture.resolve(strict=True)
    builder=REPO/'walkthrough/attention_optimization/prepare_variant.py'
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(builder),'--name',args.name,'--compiler',args.compiler,'--opt','3','--capture',str(args.capture.resolve())]
    if args.groups is not None:command+=['--groups',args.groups]
    subprocess.run(command,cwd=REPO,env=env,check=True,timeout=600)
    if args.rtl:
        runner=builder.parent/'variants'/args.name/'run_verilator.py'
        subprocess.run([sys.executable,str(runner),'--seconds','1800'],cwd=REPO,env=env,check=True,timeout=1860)


if __name__=='__main__':main()
