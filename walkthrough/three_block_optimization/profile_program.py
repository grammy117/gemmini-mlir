"""Profile host seams, device calls and runtime categories in a checked ELF.

Counters perturb code generation. Use this for attribution, never as the clean
performance benchmark. All counters are printed after kernel_entry returns.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
CATEGORIES=['memrefCopy','expf','rsqrtf','powf','sinf','cosf','malloc','free','aligned_alloc']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'variants/one_rne_final')
    parser.add_argument('--name',default='profile_one_rne')
    args=parser.parse_args();source=args.source.resolve()
    out=ROOT/'exploration'/args.name;out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'capture',out/'capture')
    shutil.copytree(source/'build',out/'build')
    build=out/'build';original=source/'build'
    text=(original/'combined.ll').read_text()
    groups=len(json.loads((original/'placement.json').read_text())['device_regions'])
    observed=[];lines=[];in_entry=False
    for line in text.splitlines():
        if line.startswith('define ') and '@kernel_entry(' in line:
            in_entry=True;lines.append(line);continue
        if in_entry and line.strip()=='entry:':
            lines.extend([line,'  call void @merlin_profile_mark(i32 0)']);continue
        if in_entry and line.strip().startswith('ret '):
            lines.append(f'  call void @merlin_profile_mark(i32 {2*groups+1})')
        match=re.search(r'call .*@merlin_device_group_(\d+)\.[\d]+\(',line)
        if match:
            index=len(observed);observed.append(int(match[1]))
            lines.extend([f'  call void @merlin_profile_mark(i32 {2*index+1})',line,
                          f'  call void @merlin_profile_mark(i32 {2*index+2})'])
        else:lines.append(line)
        if in_entry and line=='}':in_entry=False
    if observed!=list(range(groups)):raise ValueError('profile boundaries do not cover every routed device group')
    lines.append('declare void @merlin_profile_mark(i32)')
    (build/'combined.ll').write_text('\n'.join(lines)+'\n')
    c='''#include <stdint.h>
#include <stddef.h>
#include "htif.h"
static uint64_t segments[128],last;
static uint64_t calls[9],ticks[9];
static inline uint64_t now(void) {uint64_t v;__asm__ volatile("csrr %0,mcycle":"=r"(v)::"memory");return v;}
void merlin_profile_mark(int i) {uint64_t t=now();if(i)segments[i]+=t-last;last=t;}
static void end(int i,uint64_t begin) {ticks[i]+=now()-begin;calls[i]++;}
extern void __real_memrefCopy(int64_t,void*,void*);
void __wrap_memrefCopy(int64_t n,void*a,void*b) {uint64_t t=now();__real_memrefCopy(n,a,b);end(0,t);}
'''
    for i,name in enumerate(CATEGORIES[1:6],1):
        argspec='float a,float b' if name=='powf' else 'float a'
        argv='a,b' if name=='powf' else 'a'
        c+=f'extern float __real_{name}({argspec});\nfloat __wrap_{name}({argspec}) {{uint64_t t=now();float v=__real_{name}({argv});end({i},t);return v;}}\n'
    c+='''extern void *__real_malloc(size_t);
void *__wrap_malloc(size_t n) {uint64_t t=now();void*p=__real_malloc(n);end(6,t);return p;}
extern void __real_free(void*);
void __wrap_free(void*p) {uint64_t t=now();__real_free(p);end(7,t);}
extern void *__real_aligned_alloc(size_t,size_t);
void *__wrap_aligned_alloc(size_t a,size_t n) {uint64_t t=now();void*p=__real_aligned_alloc(a,n);end(8,t);return p;}
void merlin_profile_dump(void) {
'''
    c+=f'for(int i=1;i<={2*groups+1};i++) {{htif_puts("SEG ");htif_putd(i);htif_putc(\' \');htif_putd(segments[i]);htif_putc(\'\\n\');}}\n'
    c+='''for(int i=0;i<9;i++) {htif_puts("CAT ");htif_putd(i);htif_putc(' ');htif_putd(calls[i]);htif_putc(' ');htif_putd(ticks[i]);htif_putc('\\n');}
}
'''
    (build/'profile.c').write_text(c)
    main=(build/'main.c').read_text().replace('int main(long hart)',
        'void merlin_profile_dump(void);\nint main(long hart)')
    call=next(line for line in main.splitlines() if line.startswith('  kernel_entry('))
    (build/'main.c').write_text(main.replace(call,call+'\n  merlin_profile_dump();'))
    records=json.loads((original/'commands.json').read_text())
    def mapped(argv):return [a.replace(str(original),str(build)) for a in argv]
    model=next(r['argv'] for r in records if str(original/'combined.ll') in r['argv'] and '-c' in r['argv'])
    main_compile=next(r['argv'] for r in records if str(original/'main.c') in r['argv'] and '-c' in r['argv'])
    link=next(r['argv'] for r in records if '-nostdlib' in r['argv'])
    profile_compile=[a.replace(str(original/'main.c'),str(build/'profile.c')).replace(str(original/'main.o'),str(build/'profile.o')) for a in main_compile]
    link=mapped(link)
    link.insert(link.index('-lm'),str(build/'profile.o'))
    link.extend(f'-Wl,--wrap={name}' for name in CATEGORIES)
    commands=[]
    for argv in [mapped(model),mapped(main_compile),profile_compile,link]:
        done=subprocess.run(argv,capture_output=True,text=True,timeout=60)
        commands.append({'argv':argv,'returncode':done.returncode,'stderr':done.stderr})
        if done.returncode:raise RuntimeError(done.stderr)
    (build/'commands.json').write_text(json.dumps(commands,indent=2))
    summary=json.loads((build/'build_summary.json').read_text())
    elf=build/'attention_block.elf';summary.update(elf=str(elf),elf_bytes=elf.stat().st_size,elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest())
    (build/'build_summary.json').write_text(json.dumps(summary,indent=2))
    for script in ['run.py','run_verilator.py']:shutil.copyfile(source/script,out/script)
    description={'source':str(source),'groups':groups,'categories':CATEGORIES,
                 'segments':[{'id':i,'kind':'device' if i%2==0 else 'host','group':i//2-1 if i%2==0 else None} for i in range(1,2*groups+2)],
                 'scope':'intrusive mcycle attribution; math/runtime categories are included in host segments, not additive to them'}
    (out/'profile_description.json').write_text(json.dumps(description,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['/scratch/grammy/blast/model2MLIR/.venv/bin/python',str(out/'run.py')],env=env,check=True,timeout=30)
    print('Profile variant:',out)


if __name__=='__main__':main()
