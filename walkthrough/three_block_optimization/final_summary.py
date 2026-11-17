"""Recheck final RTL evidence, binary equality and functional generalization."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
SIM='a37f42b6d2f3413881faf154a94eaffaedf8d65b800e667200f523570d30d0d7'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def measured(root):
    root=root.resolve()
    candidates=[p for p in root.glob('verilator_*/result.json') if read(p).get('passed')]
    if len(candidates)!=1:raise ValueError('need exactly one accepted result: '+str(root))
    path=candidates[0];d=read(path)
    discovery=read(path.parent/'discovery.json')
    if not (d['returncode']==0 and not d['timed_out'] and d['harness_finish'] and
            d['comparison']['exact_bits'] and d['comparison']['complete_output'] and
            d['comparison']['observed_elements']==256 and d['simulator']['sha256']==SIM):
        raise ValueError('incomplete or changed RTL evidence: '+str(path))
    if sha(Path(d['elf']['path']))!=d['elf']['sha256']:raise ValueError('timed ELF changed')
    if sha(Path(d['log']['path']))!=d['log']['sha256']:raise ValueError('RTL log changed')
    if sha(root/'build/model.o')!=discovery['compiler_model_object']['sha256']:raise ValueError('measured model object changed')
    if sha(root/'build/attention_block.elf')!=discovery['original_elf']['sha256']:raise ValueError('original checked ELF changed')
    if sha(root/'capture/golden.npy')!=d['reference']['sha256']:raise ValueError('reference changed')
    return {'receipt':str(path.relative_to(REPO)),'cycles':d['block_cycles'],
            'wall_seconds':d['wall_seconds'],'timed_elf_sha256':d['elf']['sha256'],
            'exact_bits':True,'elements':256}


def profile(root,clean):
    result=measured(root);desc=read(root/'profile_description.json')
    raw=Path(REPO/result['receipt']).parent/'verilator.log'
    segments=[];categories=[]
    for line in raw.read_text().splitlines():
        v=line.split()
        if v[:1]==['SEG']:
            i,n=map(int,v[1:]);segments.append({'id':i,'cycles':n,'kind':'device' if i%2==0 else 'host'})
        if v[:1]==['CAT']:
            i,c,n=map(int,v[1:]);categories.append({'category':desc['categories'][i],'calls':c,'cycles':n})
    if len(segments)!=desc['groups']*2+1 or len(categories)!=len(desc['categories']):raise ValueError('incomplete profile counters')
    host=sum(s['cycles'] for s in segments if s['kind']=='host');device=sum(s['cycles'] for s in segments if s['kind']=='device')
    return {**result,'instrumentation_increase':result['cycles']/clean-1,'host_segment_cycles':host,
            'device_segment_cycles':device,'device_segment_fraction':device/(host+device),
            'categories':categories,'segments':segments,'scope':'intrusive counters; categories are included in host segments, not additive'}


def main():
    specifications={
        'separate_all':('exploration/one_inline_all','variants/three_optimized',7,'inline_all'),
        'grouped_all':('variants/one_cluster_optimized','variants/three_cluster_optimized',4,'inline_all'),
        'separate_context':('exploration/one_inline_adapters','variants/three_context_separate',7,'inline_adapters'),
        'grouped_context':('exploration/one_cluster_inline_adapters','variants/three_context_grouped',4,'inline_adapters'),
    }
    sweep={name:{'one':measured(ROOT/a),'three':measured(ROOT/b),'groups_per_block':g,'policy':p}
           for name,(a,b,g,p) in specifications.items()}
    winner=min(sweep,key=lambda n:sweep[n]['three']['cycles'])
    settings=read(REPO/'compiler_optimized/optimization.json')
    selected=sweep[winner]
    if settings['cluster_shared_products']!=(selected['groups_per_block']==4) or settings['native_helper_policy']!=selected['policy']:
        raise ValueError('defaults do not match the measured three-block winner: '+winner)
    if subprocess.run(['git','diff','--quiet','HEAD','--','compiler_optimized'],cwd=REPO).returncode:
        raise ValueError('commit compiler changes before publishing its inventory')
    implementation_commit=subprocess.check_output(['git','log','-1','--format=%H','--','compiler_optimized'],cwd=REPO,text=True).strip()
    equivalence=read(ROOT/'final_binary_equivalence.json')
    for row in equivalence['checks']:
        a=Path(row['measured_variant']);b=Path(row['final_variant'])
        for f in row['identical_build_products']:
            if sha(a/'build'/f['file'])!=f['sha256'] or sha(b/'build'/f['file'])!=f['sha256']:raise ValueError('binary equivalence no longer holds')
        for f in ['model.mlir','inputs.npz','weights.safetensors','golden.npy']:
            if sha(a/'capture'/f)!=sha(b/'capture'/f):raise ValueError('capture changed')
    one=selected['one'];three=selected['three']
    before_one=measured(REPO/'walkthrough/attention_optimization/variants/block_host_only_o3')
    before_three=measured(ROOT/'variants/three_checkpoint_o3')
    functional=[]
    for n,blocks,elements in [('one_release',1,256),('two_release',2,256),('three_release',3,256),('nearby_release',1,240)]:
        root=ROOT/'variants'/n;b=read(root/'build/build_summary.json');d=b['numerical_execution']
        if not d['exact_bits'] or d['observed_elements']!=elements or not d['passed']:raise ValueError('functional validation failed: '+n)
        if sha(Path(b['elf']))!=b['elf_sha256']:raise ValueError('functional ELF changed')
        schedules=read(root/'build/schedules.json')
        if not all(s['identical_reemission'] for s in schedules):raise ValueError('device re-emission differs')
        functional.append({'variant':n,'blocks':blocks,'elements':elements,'device_groups':b['device_calls'],
            'distinct_device_kernels':b['distinct_device_kernels'],'elf_bytes':b['elf_bytes'],'elf_sha256':b['elf_sha256'],'bit_exact':True,
            'source_mlir_sha256':sha(root/'capture/model.mlir'),'weights_sha256':sha(root/'capture/weights.safetensors'),
            'reference_sha256':sha(root/'capture/golden.npy'),'capture_description':read(root/'capture/capture_summary.json')['architecture']})
    measurements={}
    for blocks,a,b in [(1,before_one,one),(3,before_three,three)]:
        measurements[str(blocks)]={'blocks':blocks,'before_groups':blocks*7,'after_groups':blocks*selected['groups_per_block'],
            'logical_products':blocks*7,'before':a,'after':b,
            'cycle_reduction':1-b['cycles']/a['cycles'],'speedup':a['cycles']/b['cycles']}
    compiler=REPO/'compiler_optimized';inventory=[];changes=[]
    for p in sorted(compiler.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        rel=p.relative_to(compiler);row={'path':str(rel),'sha256':sha(p),'bytes':p.stat().st_size};inventory.append(row)
        old=REPO/'compiler'/rel
        if not old.is_file() or sha(old)!=row['sha256']:changes.append(row)
    (ROOT/'final_compiler_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    prof=profile(ROOT/'exploration/profile_one_optimized',sweep['separate_all']['one']['cycles'])
    prof['policy']='inline_all; seven groups; post-host-pass attribution'
    profile_variant={('inline_all',4):'profile_full_grouped',('inline_all',7):'profile_one_optimized',
        ('inline_adapters',4):'profile_context_grouped',('inline_adapters',7):'profile_context_separate'}
    functional_profile_root=ROOT/'exploration'/profile_variant[(selected['policy'],selected['groups_per_block'])]
    counter_description=read(functional_profile_root/'profile_description.json')
    counts=[]
    for line in (functional_profile_root/'build/spike.stdout.log').read_text().splitlines():
        v=line.split()
        if v[:1]==['CAT']:counts.append({'category':counter_description['categories'][int(v[1])],'calls':int(v[2])})
    if len(counts)!=len(counter_description['categories']):raise ValueError('incomplete functional call counters')
    functional_profile={'policy':selected['policy'],'groups':counter_description['groups'],'source':str(functional_profile_root.relative_to(REPO)),
        'counts':counts,'engine':'Spike functional call counts only; no timing claim',
        'bit_exact':read(functional_profile_root/'build/execution_summary.json')['exact_bits']}
    if not functional_profile['bit_exact']:raise ValueError('functional profile outputs differ')
    edge=read(ROOT/'edge_checks/result.json')
    if not edge['passed'] or edge['implementation_sha256']!=sha(compiler/'backend/host_quant_rne.py'):raise ValueError('quantization proof is stale')
    report={'schema':'gemmini.final_transformer_optimization.v1','branch':'work/transformer-intergroup-3block',
        'original_snapshot_commit':'7cd030e13360fd0961bd83fa288bd9009ba813df','checkpoint_commit':'969ce4377f7e5a8f06ef5cd5b774f867570b1a97',
        'compiler_implementation_commit':implementation_commit,'settings':settings,'selected_configuration':winner,
        'scope':'one and three complete sequential W8A8/FP32 decoder blocks; all final outputs compared; functional two-block and nearby-shape checks',
        'measurements':measurements,'functional_checks':functional,'post_host_pass_profile':prof,
        'functional_call_profile':functional_profile,'grouping_inlining_sweep':sweep,
        'native_codegen':read(ROOT/'native_codegen_results.json'),
        'rejected_stack_promotion':read(ROOT/'exploration/one_stack_probe/stack_probe.json'),
        'component_execution':{k:v for k,v in edge.items() if k!='routes'},
        'semantic_checks':{'source_sharing':8,'source_clustering':8,'shared_transfers_readout':7,'bounded_quantization':19,'host_contracts':7,'static_copies':8,'native_helpers':10,'total':67},
        'changed_compiler_files_from_frozen':changes,'inventory':'walkthrough/three_block_optimization/final_compiler_inventory.json',
        'binary_equivalence':equivalence,'simulator_sha256':SIM,
        'measurement_scope':'complete kernel_entry mcycle; cold first invocation; boot/reset/output excluded; one accepted run per arm; magic memory',
        'toolchain_adaptation':'standalone opt is absent; compile emitted combined LLVM with Clang O3, scalar RV64GC/lp64d, FP contraction and vectorization disabled',
        'archive_scope':'original compiler and paper metrics preserved; original full construction suite not rerun'}
    (ROOT/'final_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'selected':winner,'measurements':measurements,'profile':{'host':prof['host_segment_cycles'],'device':prof['device_segment_cycles'],'categories':prof['categories']},'sweep':sweep},indent=2))


if __name__=='__main__':main()
