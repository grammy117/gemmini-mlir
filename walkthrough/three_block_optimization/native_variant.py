"""Bounded native-codegen experiments on identical checked compiler LLVM.

Only the model object changes. Runtime objects, main, input/weight bytes and
numerical flags are inherited from the selected variant.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--policy',choices=('o2','os','no_unroll','internalize','inline_all'),required=True)
    args=parser.parse_args()
    if '/' in args.name:raise ValueError('fresh single-component name required')
    old=args.source.resolve();out=ROOT/'exploration'/args.name;out.mkdir(exist_ok=False)
    shutil.copytree(old/'capture',out/'capture');shutil.copytree(old/'build',out/'build')
    original=old/'build';build=out/'build'
    text=(build/'combined.ll').read_text()
    if args.policy in ('internalize','inline_all'):
        lines=[]
        for line in text.splitlines():
            if line.startswith('define ') and '@kernel_entry(' not in line:
                line=re.sub(r'^define (?:private |internal |dso_local )?','define internal ',line)
                if args.policy=='inline_all':line=line.rstrip()[:-1]+' alwaysinline {'
            lines.append(line)
        text='\n'.join(lines)+'\n';(build/'combined.ll').write_text(text)
    old_commands=json.loads((original/'commands.json').read_text())
    def mapped(argv):return [v.replace(str(original),str(build)) for v in argv]
    commands=[]
    for row in old_commands:
        argv=mapped(row['argv'])
        if str(build/'combined.ll') in argv and ('-c' in argv or '-S' in argv):
            if args.policy in ('o2','os'):argv=[('-O2' if args.policy=='o2' else '-Os') if v=='-O3' else v for v in argv]
            if args.policy=='no_unroll':argv.append('-fno-unroll-loops')
            done=subprocess.run(argv,capture_output=True,text=True,timeout=60)
            if done.returncode:raise RuntimeError(done.stderr)
        commands.append({**row,'argv':argv})
    link=next(row['argv'] for row in commands if '-nostdlib' in row['argv'])
    done=subprocess.run(link,capture_output=True,text=True,timeout=60)
    if done.returncode:raise RuntimeError(done.stderr)
    (build/'commands.json').write_text(json.dumps(commands,indent=2))
    summary=json.loads((build/'build_summary.json').read_text());elf=build/'attention_block.elf'
    summary.update(elf=str(elf),elf_bytes=elf.stat().st_size,elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest())
    (build/'build_summary.json').write_text(json.dumps(summary,indent=2))
    (out/'selection.json').write_text(json.dumps({'policy':args.policy,'source_variant':str(old),
        'original_llvm_sha256':hashlib.sha256((original/'combined.ll').read_bytes()).hexdigest(),
        'selected_llvm_sha256':hashlib.sha256(text.encode()).hexdigest(),'scope':'model object only; unchanged runtime and numeric flags'},indent=2))
    for script in ['run.py','run_verilator.py']:shutil.copyfile(old/script,out/script)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['/scratch/grammy/blast/model2MLIR/.venv/bin/python',str(out/'run.py')],env=env,check=True,timeout=30,stdout=subprocess.DEVNULL)
    print(args.policy,'ELF bytes',elf.stat().st_size,'bit-exact Spike passed',flush=True)


if __name__=='__main__':main()
