"""Prepare and build a fresh compiler variant against an unchanged capture.

Uses the established walkthrough builder and numerical runner, with explicit
compiler selection, scalar codegen level and source-pass ablation settings.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'walkthrough/attention_block'
PYTHON=Path('/scratch/grammy/blast/model2MLIR/.venv/bin/python')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--compiler',default='compiler_optimized')
    parser.add_argument('--capture',type=Path,default=BASE/'capture')
    parser.add_argument('--opt',type=int,choices=(1,2,3))
    parser.add_argument('--cse',type=int,choices=(0,1))
    parser.add_argument('--fusion',type=int,choices=(0,1))
    parser.add_argument('--groups',type=int,choices=(0,1))
    parser.add_argument('--rne',type=int,choices=(0,1))
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent/'variants')
    args=parser.parse_args()
    if '/' in args.name or args.name in ('.','..'):raise ValueError('name must be a single fresh directory component')
    out=args.output_root.resolve()/args.name
    out.mkdir(parents=True,exist_ok=False)
    compiler=(REPO/args.compiler).resolve(strict=True)
    config=compiler/'optimization.json'
    opt=args.opt if args.opt else (json.loads(config.read_text())['native_optimization_level'] if config.exists() else 1)
    capture=out/'capture';capture.mkdir()
    for name in ['model.mlir','weights.safetensors','weights.safetensors.manifest.json','inputs.npz','input_order.json','golden.npy','capture_summary.json']:
        source=args.capture/name
        if source.exists():shutil.copyfile(source,capture/name)
    import numpy as np
    output_shape=np.load(capture/'golden.npy',allow_pickle=False).shape
    output_elements=int(np.prod(output_shape))
    capture_info=json.loads((capture/'capture_summary.json').read_text())
    full_block='decoder layer' in capture_info.get('architecture','')
    source=(BASE/'build.py').read_text()
    source=source.replace('REPO = ROOT.parents[1]',f'REPO = Path({str(REPO)!r})\nCOMPILER = Path({str(compiler)!r})')
    source=source.replace("REPO/'compiler/compiler.py'","COMPILER/'compiler.py'")
    source=source.replace("{REPO}/compiler/compiler_library", "{COMPILER}/compiler_library")
    source=source.replace("'-O1','-ffp-contract=off'",f"'-O{opt}','-ffp-contract=off'")
    source=source.replace('using Clang -O1',f'using Clang -O{opt}')
    source=source.replace('output[256]',f'output[{output_elements}]').replace('i<256',f'i<{output_elements}')
    (out/'build.py').write_text(source)
    source=(Path(__file__).resolve().parent/'inspect_schedules.py').read_text()
    source=source.replace("ROOT = Path(__file__).resolve().parents[2]",f'ROOT = Path({str(REPO)!r})')
    source=source.replace("str(ROOT/'compiler')",repr(str(compiler)))
    (out/'inspect_schedules.py').write_text(source)
    for name in ['run.py','run_verilator.py']:
        source=(BASE/name).read_text()
        if full_block:
            source=source.replace('self-attention block','transformer block')
            source=source.replace('default=180)','default=1800)')
        if name=='run.py':
            source=source.replace('range(256)',f'range({output_elements})')
            source=source.replace('.reshape(1,8,32)',f'.reshape{output_shape!r}')
            source=source.replace('all 256 float32 outputs',f'all {output_elements} float32 outputs')
        (out/name).write_text(source)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    for name,value in [('GEMMINI_OPT_SOURCE_CSE',args.cse),('GEMMINI_OPT_FUSION',args.fusion),('GEMMINI_OPT_GROUPS',args.groups),('GEMMINI_OPT_RNE',args.rne)]:
        if value is not None:env[name]=str(value)
    selection={'name':args.name,'compiler':str(compiler),'capture':str(args.capture.resolve()),
               'scalar_codegen_level':opt,'source_cse':args.cse,'fusion':args.fusion,'groups':args.groups,'rne':args.rne,
               'compiler_settings':json.loads(config.read_text()) if config.exists() else None,
               'environment':{k:v for k,v in env.items() if k.startswith('GEMMINI_OPT_')}}
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    for name in ['build.py','run.py']:
        with (out/(name+'.stdout.log')).open('w') as stdout,(out/(name+'.stderr.log')).open('w') as stderr:
            done=subprocess.run([PYTHON,out/name],cwd=REPO,env=env,stdout=stdout,stderr=stderr,timeout=240)
        if done.returncode:
            print((out/(name+'.stderr.log')).read_text()[-6000:])
            print((out/(name+'.stdout.log')).read_text()[-3000:])
            raise SystemExit(done.returncode)
    summary=json.loads((out/'build/build_summary.json').read_text())
    print(json.dumps({'variant':str(out),'elf_bytes':summary['elf_bytes'],
                      'exact_bits':summary['numerical_execution']['exact_bits'],
                      'device_calls':summary['device_calls']},indent=2))


if __name__=='__main__':main()
