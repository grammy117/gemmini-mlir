"""Validate complete RTL receipts and record progress beyond the checkpoint."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def result(root):
    files=[p for p in root.glob('verilator_*/result.json') if read(p).get('passed')]
    if len(files)!=1:raise ValueError('expected one successful RTL receipt: '+str(root))
    p=files[0];d=read(p)
    if not d['comparison']['exact_bits'] or d['comparison']['observed_elements']!=256:
        raise ValueError('incomplete or inexact RTL outputs')
    if d['elf']['sha256']!=sha(Path(d['elf']['path'])):raise ValueError('timed ELF changed')
    return {'path':str(p),'cycles':d['block_cycles'],'wall_seconds':d['wall_seconds'],
            'exact_bits':True,'elements':256,'simulator_sha256':d['simulator']['sha256'],'elf':d['elf']}


def main():
    baseline=read(ROOT/'baseline_identity.json')
    before_one=result(REPO/'walkthrough/attention_optimization/variants/block_host_only_o3')
    before_three=result(ROOT/'variants/three_checkpoint_o3')
    after_one=result(ROOT/'variants/one_rne_final')
    after_three=result(ROOT/'variants/three_rne_final')
    arms=[before_one,before_three,after_one,after_three]
    if len({x['simulator_sha256'] for x in arms})!=1:raise ValueError('RTL simulator differs')
    for name in ['model.mlir','inputs.npz','weights.safetensors','golden.npy']:
        if sha(ROOT/'variants/three_checkpoint_o3/capture'/name)!=sha(ROOT/'variants/three_rne_final/capture'/name):
            raise ValueError('three-block comparison inputs differ: '+name)
        if sha(REPO/'walkthrough/transformer_layer/capture'/name)!=sha(ROOT/'variants/one_rne_final/capture'/name):
            raise ValueError('one-block comparison inputs differ: '+name)
    measurements={}
    for blocks,a,b in [(1,before_one,after_one),(3,before_three,after_three)]:
        measurements[str(blocks)]={'blocks':blocks,'device_groups':blocks*7,'before':a,'after':b,
            'cycles_saved':a['cycles']-b['cycles'],'cycle_reduction':1-b['cycles']/a['cycles'],
            'speedup':a['cycles']/b['cycles']}
    functional=[]
    for name in ['one_rne_final','two_rne_final','three_rne_final','nearby_rne_final']:
        root=ROOT/'variants'/name;b=read(root/'build/build_summary.json')
        if not b['numerical_execution']['exact_bits']:raise ValueError('functional check failed: '+name)
        recipe=next(root.glob('fresh10-authored-compiler/*/mixed/combined/scalar-quant-rne.json'))
        r=read(recipe)
        if r['implementation_sha256']!=sha(REPO/'compiler_optimized/backend/host_quant_rne.py'):
            raise ValueError('build used a different legalizer')
        if r['output_sha256']!=sha(root/'build/combined.ll'):raise ValueError('selected LLVM changed')
        functional.append({'name':name,'exact_bits':True,'elements':b['numerical_execution']['observed_elements'],
                           'device_groups':b['device_calls'],'rewritten_scalar_sites':len(r['routes']),
                           'elf_bytes':b['elf_bytes'],'receipt':str(recipe)})
    edge=read(ROOT/'edge_checks/result.json')
    if not edge['passed']:raise ValueError('RV64GC boundary checks failed')
    changes=[];original={row['path']:row['sha256'] for row in baseline['files']}
    for p in sorted((REPO/'compiler_optimized').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            rel=p.relative_to(REPO/'compiler_optimized').as_posix()
            if original.get(rel)!=sha(p):changes.append({'path':rel,'sha256':sha(p)})
    cache=next((ROOT/'variants/three_rne_final/fresh10-authored-compiler').iterdir())
    report={'schema':'gemmini.three_decoder_block_optimization.v1',
            'branch':'work/transformer-intergroup-3block','checkpoint_commit':baseline['commit'],
            'scope':'one, two and three complete sequential decoder blocks; distinct per-block weights',
            'shape_per_block':read(ROOT/'capture/capture_summary.json')['shape'],
            'chain_identity':read(ROOT/'chain_identity.json'),
            'settings':read(REPO/'compiler_optimized/optimization.json'),
            'compiler_changes_since_checkpoint':changes,'measurements':measurements,
            'functional_checks':functional,'source_cse_three_blocks':read(cache/'source-cse.json'),
            'ablation_identity':read(ROOT/'ablation_identity.json'),
            'boundary_execution':{k:v for k,v in edge.items() if k!='routes'},
            'semantic_checks':{'bounded_rne':19,'source_cse':8,'source_clustering':8,'transfers_and_readout':7},
            'stages_index':str(ROOT/'stages_index.json'),
            'transformation':'closed f32 round-to-even, complete integral ordered clamps and signed conversion -> bounded RV64GC fmax/fmin/fcvt with explicit RNE',
            'refusals':['strict FP/fenv','extra intermediate observers','unknown round callee','source-owned runtime symbol',
                        'unsupported CPU or float/integer type','unproved clamp bounds','unrecognized flags or scalar syntax'],
            'measurement_scope':'mcycle around complete kernel_entry; includes host and Gemmini; boot, reset and output excluded; single cold RTL run per arm',
            'simulator':'existing unchanged GemminiRocketConfig Verilator binary, magic memory, one host CPU core, 30-minute wall limit per run',
            'toolchain':'same emitted-LLVM -> Clang -O3 RV64GC/lp64d path in all arms; contraction and vectorization disabled; local opt remains absent',
            'numerics':'bit-exact to seeded quantized PyTorch; W8A8 linears, host FP32 attention/norms/residuals/SiLU; no trained checkpoint',
            'reproduce':f'/scratch/grammy/blast/model2MLIR/.venv/bin/python {ROOT}/reproduce.py --name three_repeat --blocks 3 --rtl'}
    (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{field:v[field] for field in ['blocks','device_groups','cycles_saved','cycle_reduction','speedup']}
                      for k,v in measurements.items()},indent=2))


if __name__=='__main__':main()
