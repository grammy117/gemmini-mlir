"""Validate numerical/timing receipts and assemble the one-block comparison."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
VARIANTS=REPO/'walkthrough/attention_optimization/variants'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def measured(name):
    root=VARIANTS/name
    builds=read(root/'build/build_summary.json')
    passed=[p for p in root.glob('verilator_*/result.json') if read(p).get('passed')]
    if len(passed)!=1:raise ValueError(f'{name}: expected one complete successful RTL receipt')
    path=passed[0];r=read(path)
    if not r['comparison'].get('exact_bits') or r['comparison']['observed_elements']!=256:
        raise ValueError('a timing receipt lacks the complete bit-exact output')
    if r['elf']['sha256']!=digest(Path(r['elf']['path'])):
        raise ValueError('measured ELF changed')
    return {'name':name,'device_groups':builds['device_calls'],'elf_bytes':builds['elf_bytes'],
            'block_cycles':r['block_cycles'],'wall_seconds':r['wall_seconds'],
            'exact_bits':True,'elements':256,'result':str(path),'simulator_sha256':r['simulator']['sha256'],
            'source_sha256':builds['source_mlir_sha256'],'elf':r['elf']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected',default='block_host_only_o3')
    args=parser.parse_args()
    names=['block_baseline_o3','block_host_only_o3','block_wide_o3','block_readout_o3']
    if args.selected not in names:names.append(args.selected)
    rows=[measured(name) for name in names]
    by_name={r['name']:r for r in rows}
    baseline=by_name['block_baseline_o3'];host=by_name['block_host_only_o3'];selected=by_name[args.selected]
    if len({r['source_sha256'] for r in rows})!=1 or len({r['simulator_sha256'] for r in rows})!=1:
        raise ValueError('source or simulator differs between comparison arms')
    for file in ['weights.safetensors','inputs.npz','golden.npy']:
        if len({digest(VARIANTS/n/'capture'/file) for n in names})!=1:
            raise ValueError(file+' differs between arms')
    source_passes=read(ROOT/'source-pass-report.json')
    selected_root=VARIANTS/args.selected
    schedules=read(VARIANTS/'block_readout_o3/build/schedules.json')
    groups=[]
    for s in schedules:
        for r in s['lowering'].get('shared_products',[]):
            e=r['completion_epochs'];w=e.get('wide_rhs_transfers',{})
            groups.append({'members':r['members'],'requested_activation_bytes':r['requested_lhs_bytes'],
                           'weight_loads_before_coalescing':w.get('original_physical_rhs_commands'),
                           'weight_loads_after_coalescing':w.get('physical_rhs_commands'),
                           'uniform_readout':e.get('readout_epochs',False),
                           'bounded_loop_specialization':e.get('literal_epoch',{}).get('applied',False)})
    nearby=read(VARIANTS/'block_nearby_selected_o3/build/build_summary.json')
    if not nearby['numerical_execution']['exact_bits']:raise ValueError('nearby full block differs')
    files=[]
    for path in sorted((REPO/'compiler_optimized').rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:continue
        relative=path.relative_to(REPO/'compiler_optimized');original=REPO/'compiler'/relative
        if not original.exists() or original.read_bytes()!=path.read_bytes():
            files.append({'path':str(relative),'sha256':digest(path)})
    old=read(REPO/'walkthrough/transformer_layer/verilator_20261117T080655Z/result.json')
    report={'schema':'gemmini.one_transformer_block_optimization.v1',
            'scope':'one complete decoder block; seven original device products and intervening host operations',
            'shape':read(REPO/'walkthrough/transformer_layer/capture/capture_summary.json')['shape'],
            'source_sha256':baseline['source_sha256'],'selected_variant':args.selected,
            'settings':read(REPO/'compiler_optimized/optimization.json'),
            'compiler_changes':files,'source_cse':source_passes['cse'],
            'optional_source_grouping_analysis':source_passes['clusters'],
            'selected_group_structure':(['Q','K','V','output projection','gate','up','down projection']
                if selected['device_groups']==7 else ['Q,K,V','output projection','gate,up','down projection']),
            'optional_grouped_structure':['Q,K,V','output projection','gate,up','down projection'],
            'clustering_default':'new product-motion pass disabled after whole-block RTL A/B; host sharing is enabled across the original seven groups',
            'optional_grouped_schedules':groups,'runs':rows,
            'selected_build_equivalence':read(ROOT/'selected-build-equivalence.json'),
            'improvement':{'cycles_saved_vs_same_O3':baseline['block_cycles']-selected['block_cycles'],
                           'cycle_reduction_vs_same_O3':1-selected['block_cycles']/baseline['block_cycles'],
                           'speedup_vs_same_O3':baseline['block_cycles']/selected['block_cycles'],
                           'cycles_saved_by_group_path_vs_host_only':host['block_cycles']-selected['block_cycles'],
                           'cycle_reduction_vs_host_only':1-selected['block_cycles']/host['block_cycles'],
                           'optional_grouped_cycle_increase_vs_selected':by_name['block_readout_o3']['block_cycles']/selected['block_cycles']-1,
                           'historical_O1_cycles':old['block_cycles'],
                           'cycle_reduction_vs_historical_O1':1-selected['block_cycles']/old['block_cycles']},
            'host_static_allocation_sites':{'original':33,'optimized':27},
            'activation_quantization_chains':{'original':7,'optimized':4},
            'requested_device_activation_bytes':{'original':2048,'selected':2048 if selected['device_groups']==7 else 1280,
                                                 'optional_grouped':1280,'scope':'logical requested bytes; not measured memory bus traffic'},
            'nearby_functional':{'shape':read(ROOT/'nearby/capture/capture_summary.json')['shape'],
                                 'engine':'Spike plus Gemmini; no RTL timing claim',
                                 'exact_bits':True,'elements':nearby['numerical_execution']['observed_elements'],
                                 'device_groups':nearby['device_calls']},
            'checks':{'source_cse':8,'clustering':8,'transfer_and_readout':7,'frozen_source_inventory_files':1251},
            'measurement':'mcycle around complete kernel_entry including host and accelerator; boot/reset/output excluded; one RTL run per arm',
            'simulator':'unchanged GemminiRocketConfig Verilator binary; magic memory; one host CPU core',
            'toolchain_adaptation':'both arms compile emitted combined LLVM with Clang -O3, contraction off and vectorization disabled; local opt executable absent',
            'limits':['W8A8 projection linears with FP32 host attention, norms, residuals and SiLU',
                      'tiny seeded demonstration block; no trained checkpoint or multi-block validation',
                      'source motion crosses only pure operations with dominating identical SSA inputs; all original product outputs remain',
                      'no cross-call scratchpad residency assumption; unsupported schedules keep existing lowering'],
            'reproduce':f'/scratch/grammy/blast/model2MLIR/.venv/bin/python {ROOT}/reproduce.py --name block_repeat --rtl'}
    (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'selected':args.selected,'runs':[{k:r[k] for k in ('name','block_cycles','device_groups','exact_bits')} for r in rows],
                      'improvement':report['improvement']},indent=2))


if __name__=='__main__':main()
