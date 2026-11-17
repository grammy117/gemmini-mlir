"""Time the complete self-attention block on an existing GemminiRocket RTL simulator.

Reuses the already checked model.o and runtime objects. Only the harness adds
mcycle reads around kernel_entry, outside allocation initialization and output.
Results are new products; historical TVM records and the prior ELF are retained.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BUILD = ROOT/'build'
HISTORY = Path('/scratch/grammy/blast/merlin-tvm-tuning/out/artifacts/verification/gemmini-tvm-rtl-a26/v1/verification_gemmini-tvm-rtl-a26_v1_20261026T184010Z_431485b/runtime-qualification.json')


def identity(path):
    path = Path(path).resolve(strict=True)
    return {'path':str(path),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def limited_child():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_STACK,(64<<20,64<<20))
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(32<<20,32<<20))


def execute(argv, env, log, seconds):
    start=time.monotonic()
    timed_out=False
    with log.open('wb') as stream:
        child=subprocess.Popen(list(map(str,argv)),stdout=stream,stderr=subprocess.STDOUT,
                               env=env,preexec_fn=limited_child,start_new_session=True)
        print(json.dumps({'event':'started','pid':child.pid,'log':str(log),
                          'wall_limit_seconds':seconds}),flush=True)
        try:
            code=child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                code=child.wait()
        except BaseException:
            # The child owns a separate session; clean it up on interruption.
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            raise
    return {'argv':list(map(str,argv)),'returncode':code,'timed_out':timed_out,
            'wall_seconds':time.monotonic()-start,'log':identity(log)}


def compare(text, golden):
    matches=re.findall(r'^OUT (\d+) (0x[0-9a-fA-F]+)$',text,re.MULTILINE)
    complete=[int(i) for i,_ in matches]==list(range(golden.size)) and 'TRANSFORMER_END' in text
    report={'complete_output':complete,'observed_elements':len(matches)}
    if complete:
        actual=np.array([int(v,16) for _,v in matches],dtype=np.uint32).view(np.float32).reshape(golden.shape)
        report.update(max_abs_error=float(np.max(np.abs(actual-golden))),
                      exact_bits=bool(np.array_equal(actual.view(np.uint32),golden.view(np.uint32))),
                      within_tolerance=bool(np.allclose(actual,golden,atol=1e-5,rtol=1e-5)))
        return report,actual
    return report,None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=180)
    parser.add_argument('--max-cycles',type=int,default=2_000_000)
    args=parser.parse_args()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/('verilator_'+stamp)
    out.mkdir(exist_ok=False)
    history=json.loads(HISTORY.read_text())
    simulator=identity(history['simulator']['path'])
    if simulator['sha256']!=history['simulator']['sha256']:
        raise RuntimeError('simulator changed from the selected a26 qualification record')
    old_build=json.loads((BUILD/'build_summary.json').read_text())
    if identity(BUILD/'attention_block.elf')['sha256']!=old_build['elf_sha256']:
        raise RuntimeError('previously checked ELF changed')
    source=(BUILD/'main.c').read_text()
    old_call=next(line for line in source.splitlines() if line.startswith('  kernel_entry('))
    source=source.replace(old_call,
        '  uint64_t begin_cycle,end_cycle;\n'
        '  __asm__ volatile("fence; csrr %0, mcycle" : "=r"(begin_cycle) :: "memory");\n'
        +old_call+'\n'
        '  __asm__ volatile("fence; csrr %0, mcycle" : "=r"(end_cycle) :: "memory");\n'
        '  htif_puts("BLOCK_CYCLES "); htif_putd((long)(end_cycle-begin_cycle)); htif_putc(\'\\n\');\n'
        '  htif_line_flush(0);')
    (out/'main.c').write_text(source)
    commands=json.loads((BUILD/'commands.json').read_text())
    compile_argv=next(row['argv'] for row in commands if str(BUILD/'main.c') in row['argv'] and '-c' in row['argv'])
    link_argv=next(row['argv'] for row in commands if '-nostdlib' in row['argv'])
    replacements={str(BUILD/'main.c'):str(out/'main.c'),str(BUILD/'main.o'):str(out/'main.o'),
                  str(BUILD/'attention_block.elf'):str(out/'attention_block.elf'),
                  '-Wl,-Map='+str(BUILD/'attention_block.map'):'-Wl,-Map='+str(out/'attention_block.map')}
    build_records=[]
    for original in [compile_argv,link_argv]:
        argv=[replacements.get(x,x) for x in original]
        result=subprocess.run(argv,capture_output=True,text=True,timeout=30)
        build_records.append({'argv':argv,'returncode':result.returncode,'stderr':result.stderr})
        if result.returncode:raise RuntimeError(result.stderr)
    elf=out/'attention_block.elf'
    env=dict(os.environ)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               LD_LIBRARY_PATH=history['runtime_environment']['LD_LIBRARY_PATH'])
    golden=np.load(ROOT/'capture/golden.npy',allow_pickle=False)
    prior=json.loads((BUILD/'execution_summary.json').read_text())
    spike_argv=prior['command'][:-1]+[str(elf)]
    spike=execute(spike_argv,env,out/'spike.log',30)
    spike['comparison'],_=compare((out/'spike.log').read_text(errors='replace'),golden)
    if spike['returncode']!=0 or not spike['comparison'].get('exact_bits'):
        (out/'spike_failure.json').write_text(json.dumps(spike,indent=2))
        raise RuntimeError('timed ELF failed its Spike functional control')
    discovery={'historical_receipt':identity(HISTORY),'simulator':simulator,
               'historical_controls':[{k:r[k] for k in ['case','wall_seconds','returncode','timed_out']} for r in history['accepted_controls']],
               'historical_attention_wall_estimate_seconds':[180,360],
               'workload':json.loads((ROOT/'capture/capture_summary.json').read_text()).get('architecture','self-attention block'),
               'capture_source':identity(ROOT/'capture/model.mlir'),
               'estimate_basis':'historical attention estimate; use recorded target cycles and current machine load for the selected capture',
               'prior_attempt':'same block exceeded the original 180-second wall limit',
               'approved_wall_limit_seconds':args.seconds,
               'configuration':'GemminiRocketConfig; Rocket + 16x16 INT8 Gemmini; 256MiB magic memory',
               'numerical_reference':identity(ROOT/'capture/golden.npy'),
               'compiler_model_object':identity(BUILD/'model.o'),
               'instrumented_elf':identity(elf),
               'timing_boundary':'mcycle around complete kernel_entry; excludes boot, allocator reset and output printing',
               'original_elf':identity(BUILD/'attention_block.elf'),
               'build_commands':build_records,'spike_control':spike,
               'limits':{'cpu_affinity':[min(os.sched_getaffinity(0))],'address_space_bytes':2<<30,
                         'stack_bytes':64<<20,'wall_seconds':args.seconds,'max_cycles':args.max_cycles}}
    # This prebuilt --main defaults VerilatedContext to hardware_concurrency()
    # even though VTestDriver::threads() returns 1. Interpose only that host CPU
    # count query, avoiding hundreds of idle thread stacks on the shared VM.
    # The simulator executable, generated RTL objects and device model stay exact.
    shim_source=out/'one_host_thread.c'
    shim_source.write_text(
        '/* Host resource policy only; the compiled RTL model uses one thread. */\n'
        'unsigned merlin_host_threads(void) __asm__("_ZNSt6thread20hardware_concurrencyEv");\n'
        'unsigned merlin_host_threads(void) { return 1; }\n')
    shim=out/'one_host_thread.so'
    shim_argv=['/scratch/grammy/blast/merlin/third_party/llvm-build/bin/clang-23',
               '-O2','-fPIC','-shared','-nostdlib',str(shim_source),'-o',str(shim)]
    compiled=subprocess.run(shim_argv,capture_output=True,text=True,timeout=30)
    if compiled.returncode:raise RuntimeError(compiled.stderr)
    env['LD_PRELOAD']=str(shim)
    discovery['host_thread_limit']={'source':identity(shim_source),'library':identity(shim),
                                  'compile_argv':shim_argv,'LD_PRELOAD':str(shim),
                                  'scope':'std::thread::hardware_concurrency returns 1; unchanged serial RTL model'}
    (out/'discovery.json').write_text(json.dumps(discovery,indent=2))
    argv=[simulator['path'],'+permissive','+max-cycles='+str(args.max_cycles),
          '+loadmem='+str(elf),'+permissive-off',str(elf)]
    result=execute(argv,env,out/'verilator.log',args.seconds)
    text=(out/'verilator.log').read_text(errors='replace')
    result['comparison'],actual=compare(text,golden)
    result['harness_finish']='Verilog $finish' in text
    cycles=re.findall(r'^BLOCK_CYCLES (\d+)$',text,re.MULTILINE)
    result['block_cycles']=int(cycles[0]) if len(cycles)==1 else None
    result['passed']=(result['returncode']==0 and not result['timed_out'] and result['harness_finish']
                      and result['block_cycles'] is not None and result['comparison'].get('exact_bits',False))
    result['status']='passed' if result['passed'] else ('wall_timeout' if result['timed_out'] else 'failed_or_incomplete')
    result['simulator']=simulator
    result['elf']=identity(elf)
    result['reference']=identity(ROOT/'capture/golden.npy')
    result['measurement_scope']=discovery['timing_boundary']
    if actual is not None:np.save(out/'actual.npy',actual)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(out),'status':result['status'],'wall_seconds':result['wall_seconds'],
                      'block_cycles':result['block_cycles'],'comparison':result['comparison']},indent=2),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
