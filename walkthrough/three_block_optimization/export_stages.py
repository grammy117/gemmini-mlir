"""Make the actual products easy to navigate; replay recorded MLIR pass prefixes."""
import hashlib
import argparse
import json
import shutil
import subprocess
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',default='three_rne_final')
    parser.add_argument('--variant-root',type=Path,help='explicit built variant, including exploration results')
    args=parser.parse_args()
    ROOT=args.variant_root.resolve() if args.variant_root else EXPERIMENT/'variants'/args.variant
    BUILD=ROOT/'build'
    STAGES=EXPERIMENT/'stages'
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
        '13_before_scalar_quant_rne.ll':mixed/'combined/before-scalar-quant-rne.ll',
        '13_scalar_quant_rne.json':mixed/'combined/scalar-quant-rne.json',
        '00_shared_source.mlir':mixed.parent/'source-optimized.mlir',
        '14_model.s':BUILD/'model.s',
    }
    for name, source in files.items():
        shutil.copyfile(source,STAGES/name)
    for name,source in {
        '07a_host_static_copies.mlir':mixed/'host/model.static-copies.mlir',
        '07_static_host_copies.json':mixed/'host/static-host-copies.json',
        '13_host_contracts.json':mixed/'combined/host-contracts.json',
        '13_native_helpers.json':mixed/'combined/native-helpers.json',
    }.items():
        if source.is_file():shutil.copyfile(source,STAGES/name)
        else:(STAGES/name).unlink(missing_ok=True)
    schedules=json.loads((BUILD/'schedules.json').read_text())
    first=schedules[0]['lowering']
    if 'dense_schedules' in first:
        (STAGES/'05_q_schedule.mlir').write_text(first['dense_schedules'][0]['schedule_ir']['text'])
        (STAGES/'05_q_shared_schedule.json').unlink(missing_ok=True)
    else:
        (STAGES/'05_q_shared_schedule.json').write_text(json.dumps(first['shared_products'],indent=2))
        (STAGES/'05_q_schedule.mlir').unlink(missing_ok=True)
    recipe=json.loads((BUILD/'lowering_recipe.json').read_text())
    invocations=[c['argv'] for c in recipe['commands'] if any(a.startswith('--pass-pipeline=') for a in c['argv'])]
    invocation=next(c for c in invocations if any('convert-linalg-to-loops' in a for a in c))
    full=next(a for a in invocation if a.startswith('--pass-pipeline='))
    body=full.removeprefix('--pass-pipeline=builtin.module(')[:-1]
    loop_token='func.func(convert-linalg-to-loops)'
    loops_end=body.index(loop_token)+len(loop_token)
    replays=[]
    if (mixed/'host/model.bufferized.mlir').is_file():
        shutil.copyfile(mixed/'host/model.bufferized.mlir',STAGES/'07_host_bufferized.mlir')
        prefixes=[('08_host_loops.mlir',loops_end)]
    else:
        buffer_end=body.index('}',body.index('one-shot-bufferize{'))+1
        prefixes=[('07_host_bufferized.mlir',buffer_end),('08_host_loops.mlir',loops_end)]
    for name,end in prefixes:
        argv=[invocation[0],invocation[1],
              '--pass-pipeline=builtin.module('+body[:end]+')','-o',str(STAGES/name)]
        result=subprocess.run(argv,capture_output=True,text=True,timeout=30)
        replays.append({'argv':argv,'returncode':result.returncode,'stderr':result.stderr})
        if result.returncode:raise RuntimeError(result.stderr)
    short=[]
    for row in schedules:
        if 'shared_products' in row['lowering']:
            s=row['lowering']['shared_products'][0]
            wide=s['completion_epochs']['wide_rhs_transfers']
            short.append({'kernel':Path(row['llvm']).parent.name,'kind':'shared_products',
                'members':s['members'],'lhs_loads':s['lhs_loads'],
                'physical_rhs_loads':wide.get('physical_rhs_commands',s['rhs_loads']),
                'logical_rhs_bands':wide.get('logical_rhs_bands',s['rhs_loads']),
                'output_tiles':s['output_tiles'],'completion':s['completion_epochs']['completion'],
                'identical_device_reemission':row['identical_reemission']})
            continue
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
    summary={'variant':str(ROOT),'kernels':short,
             'tensor_preparation_cse':json.loads((mixed/'host/tensor-preparation-cse.json').read_text()),
             'host_static_arena':json.loads((mixed/'host/adapter-result.json').read_text())['stats']['static_arena'],
             'stage_prefix_replays':replays,
             'stages':[{'file':str(p),'bytes':p.stat().st_size,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(STAGES.iterdir()) if p.is_file()]}
    (EXPERIMENT/'stages_index.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'kernels':short,'stage_files':len(summary['stages'])},indent=2))

if __name__=='__main__':
    main()
