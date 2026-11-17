"""Build an isolated upstream stack-promotion experiment with bounded inputs.

This probe checks the selected workload's total static allocation bound first.
It does not enable stack promotion in the shipped compiler.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(REPO/'compiler_optimized'),str(REPO/'compiler_optimized/compiler_library')]


def main():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from xdsl.dialects.builtin import IntegerType,Float32Type,Float64Type,Float16Type,IndexType,NoneAttr
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--name',required=True)
    a=p.parse_args()
    source=next(a.source.glob('fresh10-authored-compiler/*/mixed/host/model.bufferized.mlir'))
    module=parse_mlir_text(source.read_text());total=0;count=0
    for op in module.walk():
        if op.name not in {'memref.alloc','memref.alloca'}:continue
        t=op.results[0].type
        if not isinstance(t.layout,NoneAttr) or any(d<0 for d in t.get_shape()):raise ValueError('unbounded or strided allocation')
        e=t.element_type
        bits=e.width.data if isinstance(e,IntegerType) else {Float32Type:32,Float64Type:64,Float16Type:16,IndexType:64}.get(type(e))
        if bits is None:raise ValueError('unknown element storage')
        total+=math.prod(t.get_shape())*((bits+7)//8)+64;count+=1
    if total>65536:raise ValueError('probe total static frame bound exceeds 64 KiB')
    dst=ROOT/'exploration'/(a.name+'_compiler')
    shutil.copytree(REPO/'compiler_optimized',dst,ignore=shutil.ignore_patterns('__pycache__'))
    path=dst/'compiler_library/merlin/llvmlower/stock_cli.py';text=path.read_text()
    marker="    if os.environ.get('MERLIN_STOCK_ALIGNED_ALLOC','0') == '1':"
    if text.count(marker)!=1:raise ValueError('unexpected pipeline construction')
    text=text.replace(marker,"    pipeline=pipeline.replace('ownership-based-buffer-deallocation','func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=4096 max-rank-of-allocated-memref=0}),ownership-based-buffer-deallocation')\n"+marker)
    path.write_text(text)
    done=subprocess.run(['/scratch/grammy/blast/model2MLIR/.venv/bin/python',str(REPO/'walkthrough/attention_optimization/prepare_variant.py'),
        '--name',a.name,'--compiler',str(dst),'--opt','3','--capture',str(a.source/'capture'),'--output-root',str(ROOT/'exploration')],timeout=300)
    (ROOT/'exploration'/a.name/'stack_probe.json').write_text(json.dumps({'allocation_count':count,'static_allocation_upper_bound_with_padding':total,'budget':65536,
        'builder_returncode':done.returncode,'status':'functional_pass' if done.returncode==0 else 'rejected_functional_failure',
        'scope':'experimental fixed-input guard; upstream promotion before ownership deallocation'},indent=2))


if __name__=='__main__':main()
