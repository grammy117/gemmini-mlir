"""Capture a true chain of complete decoder blocks with distinct weights."""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import m2m
import numpy as np
import torch
from torch import nn
from m2m.coverage import opaque_report

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'walkthrough/transformer_layer'))
from capture import replace_linears


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'capture')
    args=parser.parse_args()
    if not 1<=args.blocks<=3:raise ValueError('this bounded experiment supports one to three blocks')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(117)
    spec=importlib.util.spec_from_file_location('small_llama',Path('/scratch/grammy/blast/model2MLIR/workloads/small_llama/loader.py'))
    llama=importlib.util.module_from_spec(spec);spec.loader.exec_module(llama)
    class Chain(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=nn.ModuleList([llama.Block(d=32,h=2,hidden=64) for _ in range(args.blocks)])
        def forward(self,x):
            pos=torch.arange(x.shape[1])
            for block in self.blocks:x=block(x,pos)
            return x
    floating=Chain().eval();model=copy.deepcopy(floating);replace_linears(model)
    baseline=REPO/'walkthrough/transformer_layer/capture'
    x=torch.from_numpy(np.load(baseline/'inputs.npz',allow_pickle=False)['x'].copy())
    with torch.no_grad():
        intermediate=x
        for i,block in enumerate(model.blocks):
            intermediate=block(intermediate,torch.arange(x.shape[1]))
            np.save(out/f'block_{i}_golden.npy',intermediate.numpy())
            if i==0 and not np.array_equal(intermediate.numpy().view(np.uint32),np.load(baseline/'golden.npy').view(np.uint32)):
                raise ValueError('first block differs from the completed one-block experiment')
        golden=model(x)
        assert torch.equal(golden,intermediate)
    result=m2m.convert(model,(x,),backend='fx_importer',weights_path=str(out/'weights.safetensors'),capture_trace=True)
    if not result.ok:raise RuntimeError(result.diagnostics)
    opaque=opaque_report(result.mlir_text)
    if opaque:raise ValueError('capture contains opaque operations: '+str(opaque))
    (out/'model.mlir').write_text(result.mlir_text)
    (out/'exported_graph.txt').write_text(str(result.exported_program.graph_module.graph))
    np.savez(out/'inputs.npz',x=x.numpy());np.save(out/'golden.npy',golden.numpy())
    (out/'input_order.json').write_text(json.dumps(['x']))
    report={'architecture':f'{args.blocks} complete Llama-style decoder layers in sequence',
            'blocks':args.blocks,'shape':{'batch':1,'tokens':8,'model_width':32,'heads':2,'head_width':16,'mlp_width':64},
            'weights':'seed-117 distinct per-block weights; first block matches the previous experiment',
            'input_sha256':hashlib.sha256((out/'inputs.npz').read_bytes()).hexdigest(),
            'first_block_bit_exact_to_previous':True,'capture_path':result.path_taken,'opaque':opaque,
            'quantization':{'activation_scale':1/32,'weight_scale':1/128,'dequant_scale':1/4096,
                            'rounding':'ties-even','clamp':[-128,127],'accumulator':'int32','linear_count':7*args.blocks},
            'operation_counts':dict(Counter(op.name for op in result.module.walk())),
            'diagnostics':result.diagnostics}
    (out/'capture_summary.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps({k:report[k] for k in ['architecture','first_block_bit_exact_to_previous','opaque','quantization']},indent=2))


if __name__=='__main__':main()
