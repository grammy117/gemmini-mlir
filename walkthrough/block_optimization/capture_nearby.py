"""Capture a second full block to check generalization of compiler decisions."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import m2m
import numpy as np
import torch
from torch import nn
from m2m.coverage import opaque_report

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'transformer_layer'))
from capture import replace_linears


def main():
    torch.set_num_threads(2)
    torch.manual_seed(119)
    spec=importlib.util.spec_from_file_location('small_llama',Path('/scratch/grammy/blast/model2MLIR/workloads/small_llama/loader.py'))
    llama=importlib.util.module_from_spec(spec);spec.loader.exec_module(llama)
    class OneLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.block=llama.Block(d=48,h=3,hidden=80)
        def forward(self,x):return self.block(x,torch.arange(x.shape[1]))
    model=copy.deepcopy(OneLayer().eval());replace_linears(model)
    x=torch.randn(1,5,48)
    with torch.no_grad():golden=model(x)
    out=ROOT/'nearby/capture';out.mkdir(parents=True,exist_ok=False)
    result=m2m.convert(model,(x,),backend='fx_importer',weights_path=str(out/'weights.safetensors'),capture_trace=True)
    if not result.ok:raise RuntimeError(result.diagnostics)
    (out/'model.mlir').write_text(result.mlir_text)
    np.savez(out/'inputs.npz',x=x.numpy());np.save(out/'golden.npy',golden.numpy())
    (out/'input_order.json').write_text(json.dumps(['x']))
    report={'architecture':'one complete Llama-style decoder layer','seed':119,
            'shape':{'batch':1,'tokens':5,'model_width':48,'heads':3,'head_width':16,'mlp_width':80},
            'capture_path':result.path_taken,'opaque':opaque_report(result.mlir_text)}
    (out/'capture_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
