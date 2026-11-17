"""Capture one complete, small Llama decoder block with explicit W8A8 linears.

Uses the existing model2MLIR Llama implementation; weights are deterministic
random demonstration weights, not a downloaded or trained checkpoint.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
import m2m
from m2m.coverage import opaque_report

ROOT = Path(__file__).resolve().parent
M2M = Path('/scratch/grammy/blast/model2MLIR')


class W8A8Linear(nn.Module):
    """Symmetric fixed-scale quantization and an actual i8*i8->i32 product."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.register_buffer('weight_t',
            torch.clamp(torch.round(linear.weight.detach() * 128.0), -128, 127)
            .to(torch.int8).t().contiguous())

    def forward(self, x):
        quantized = torch.clamp(torch.round(x * 32.0), -128, 127).to(torch.int8)
        flat = quantized.reshape(-1, quantized.shape[-1])
        accumulated = torch._int_mm(flat, self.weight_t)
        result = accumulated.to(torch.float32) * (1.0 / 4096.0)
        return result.reshape(*x.shape[:-1], self.weight_t.shape[-1])


def replace_linears(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, W8A8Linear(child))
        else:
            replace_linears(child)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(117)
    ROOT.joinpath('capture').mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location('m2m_small_llama', M2M / 'workloads/small_llama/loader.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)

    class OneLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = source.Block(d=32, h=2, hidden=64)

        def forward(self, x):
            return self.block(x, torch.arange(x.shape[1]))

    floating = OneLayer().eval()
    model = copy.deepcopy(floating)
    replace_linears(model)
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        golden = model(x)
        float_golden = floating(x)
    destination = ROOT / 'capture'
    result = m2m.convert(model, (x,), backend='fx_importer',
                         weights_path=str(destination / 'weights.safetensors'),
                         capture_trace=True)
    if not result.ok:
        raise RuntimeError(result.diagnostics)
    (destination / 'model.mlir').write_text(result.mlir_text)
    (destination / 'frontend_trace.json').write_text(json.dumps(result.capture_trace, indent=2, default=str))
    ep = result.exported_program
    if ep is not None:
        (destination / 'exported_graph.txt').write_text(str(ep.graph_module.graph))
        (destination / 'graph_signature.txt').write_text(str(ep.graph_signature))
        torch.export.save(ep, destination / 'exported_program.pt2')
    np.savez(destination / 'inputs.npz', x=x.numpy())
    np.save(destination / 'golden.npy', golden.numpy())
    np.save(destination / 'float_golden.npy', float_golden.numpy())
    (destination / 'input_order.json').write_text(json.dumps(['x']))
    report = {'architecture':'one complete Llama-style decoder layer',
              'shape':{'batch':1,'tokens':8,'model_width':32,'heads':2,'head_width':16,'mlp_width':64},
              'weights':'seed-117 random initialization; demonstration only',
              'quantization':{'activation_scale':1/32,'weight_scale':1/128,
                              'dequant_scale':1/4096,'rounding':'ties-even',
                              'clamp':[-128,127],'accumulator':'int32',
                              'linear_count':7},
              'capture_path':result.path_taken,'opaque':opaque_report(result.mlir_text),
              'quantized_vs_float_max_abs':float((golden-float_golden).abs().max()),
              'operation_counts':dict(Counter(op.name for op in result.module.walk())) if result.module else {},
              'diagnostics':result.diagnostics}
    (destination / 'capture_summary.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
