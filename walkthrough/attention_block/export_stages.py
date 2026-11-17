"""Make the actual products easy to navigate; replay recorded MLIR pass prefixes."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT/'build'
STAGES = ROOT/'stages'

def main():
    STAGES.mkdir(exist_ok=True)
    mixed = next((ROOT/'fresh10-authored-compiler').glob('*/mixed'))
    q = mixed/'merlin_device_group_0'
    files = {
        '01_model.mlir':ROOT/'capture/model.mlir',
        '02_routed.mlir':mixed/'routed.mlir',
        '03_q_source.mlir':q/'source.mlir',
        '04_q_logical_commands.json':q/'commands.json',
        '06_q_llvm.mlir':q/'device.mlir',
        '09_host_llvm.mlir':mixed/'host/model.llvm.mlir',
        '10_host.ll':mixed/'host/model.ll',
        '11_device_shim.c':mixed/'device_shim.c',
        '12_entry_wrapper.c':mixed/'combined/entry.c',
        '13_combined.ll':BUILD/'combined.ll',
        '14_model.s':BUILD/'model.s',
    }
    for name, source in files.items():
        shutil.copyfile(source,STAGES/name)
    schedules=json.loads((BUILD/'schedules.json').read_text())
    first=schedules[0]['lowering']['dense_schedules'][0]
    (STAGES/'05_q_schedule.mlir').write_text(first['schedule_ir']['text'])
    recipe=json.loads((BUILD/'lowering_recipe.json').read_text())
    invocation=next(c['argv'] for c in recipe['commands'] if any(a.startswith('--pass-pipeline=') for a in c['argv']))
    full=next(a for a in invocation if a.startswith('--pass-pipeline='))
    body=full.removeprefix('--pass-pipeline=builtin.module(')[:-1]
    buffer_end=body.index('}',body.index('one-shot-bufferize{'))+1
    loop_token='func.func(convert-linalg-to-loops)'
    loops_end=body.index(loop_token)+len(loop_token)
    replays=[]
    for name,end in [('07_host_bufferized.mlir',buffer_end),('08_host_loops.mlir',loops_end)]:
        argv=[invocation[0],str(mixed/'host/model.mlir'),
              '--pass-pipeline=builtin.module('+body[:end]+')','-o',str(STAGES/name)]
        result=subprocess.run(argv,capture_output=True,text=True,timeout=30)
        replays.append({'argv':argv,'returncode':result.returncode,'stderr':result.stderr})
        if result.returncode:raise RuntimeError(result.stderr)
    short=[]
    for row in schedules:
        s=row['lowering']['dense_schedules'][0]
        p=s['rhs_block_transfers']['projection']
        short.append({'kernel':Path(row['llvm']).parent.name,
                      'block_tiles_M_K_N':s['panel_geometry']['selected_block_tiles'],
                      'selected_pipeline':s['selected_pipeline'],
                      'retained_operand':s['operand_residency']['selection'],
                      'physical_load_commands':p['physical_load_commands'],
                      'logical_load_bands':p['expanded_logical_tile_bands'],
                      'compute_pairs':sum(e['compute_pairs'] for e in s['epochs']),
                      'stores':sum(e['readout_commands'] for e in s['epochs']),
                      'ready_readout':s['ready_readout_selection']['applied'],
                      'identical_device_reemission':row['identical_reemission']})
    summary={'kernels':short,
             'tensor_preparation_cse':json.loads((mixed/'host/tensor-preparation-cse.json').read_text()),
             'host_static_arena':json.loads((mixed/'host/adapter-result.json').read_text())['stats']['static_arena'],
             'stage_prefix_replays':replays,
             'stages':[{'file':str(p),'bytes':p.stat().st_size,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(STAGES.iterdir()) if p.is_file()]}
    (BUILD/'trace_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'kernels':short,'stage_files':len(summary['stages'])},indent=2))

if __name__=='__main__':
    main()
