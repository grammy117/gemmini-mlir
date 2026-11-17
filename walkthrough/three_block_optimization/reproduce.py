"""Build a bounded decoder chain and optionally measure it on existing RTL.

--checkpoint reconstructs the completed one-block compiler for a fair baseline.
Without it, the current compiler_optimized is used. RTL is pinned to one core,
with a 30-minute wall limit; optimized three-block runs take about 9--12 minutes
on this machine, while the checkpoint can take 12--20 minutes under host load.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
CHECKPOINT='969ce4377f7e5a8f06ef5cd5b774f867570b1a97'


def checkpoint_compiler():
    destination=ROOT/'baseline'
    compiler=destination/'compiler_optimized'
    if not compiler.exists():
        destination.mkdir(exist_ok=True)
        child=subprocess.Popen(['git','archive',CHECKPOINT,'compiler_optimized'],cwd=REPO,stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=child.stdout,mode='r|') as archive:
                archive.extractall(destination,filter='data')
        finally:
            child.stdout.close()
        if child.wait():raise RuntimeError('checkpoint archive failed')
    identity=json.loads((ROOT/'baseline_identity.json').read_text())
    if identity['commit']!=CHECKPOINT:raise ValueError('baseline checkpoint changed')
    for row in identity['files']:
        if hashlib.sha256((compiler/row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('baseline compiler changed: '+row['path'])
    return compiler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--blocks',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--checkpoint',action='store_true')
    parser.add_argument('--rtl',action='store_true')
    args=parser.parse_args()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    capture=(REPO/'walkthrough/transformer_layer/capture' if args.blocks==1 else
             ROOT/('capture' if args.blocks==3 else 'capture_two'))
    if not capture.exists():
        subprocess.run([sys.executable,str(ROOT/'capture.py'),'--blocks',str(args.blocks),'--output',str(capture)],
                       env=env,cwd=REPO,check=True,timeout=180)
    compiler=checkpoint_compiler() if args.checkpoint else REPO/'compiler_optimized'
    subprocess.run([sys.executable,str(REPO/'walkthrough/attention_optimization/prepare_variant.py'),
        '--name',args.name,'--compiler',str(compiler),'--opt','3','--capture',str(capture),
        '--output-root',str(ROOT/'variants')],env=env,cwd=REPO,check=True,timeout=600)
    if args.blocks==3 and not args.checkpoint:
        subprocess.run([sys.executable,str(ROOT/'export_stages.py'),'--variant',args.name],
                       cwd=REPO,env=env,check=True,timeout=90)
    if args.rtl:
        subprocess.run([sys.executable,str(ROOT/'variants'/args.name/'run_verilator.py'),
            '--seconds','1800','--max-cycles','6000000'],env=env,cwd=REPO,check=True,timeout=1860)


if __name__=='__main__':main()
