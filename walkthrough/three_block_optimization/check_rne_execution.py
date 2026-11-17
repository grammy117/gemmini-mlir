"""Check the emitted scalar RNE on RV64GC across widths and rounding modes."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path[:0]=[str(REPO/'compiler_optimized'),str(REPO/'compiler_optimized/compiler_library')]
from backend.host_quant_rne import rewrite
from check_rne import native,SOURCE


def main():
    out=ROOT/'edge_checks/build';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(117)
    values=[np.float32(x) for x in [0.,-0.,float('inf'),-float('inf'),1e-40,-1e-40,1e38,-1e38]]
    for bound in [128,32768,8388608]:
        for v in [-bound-1.,-bound-.5,-bound,-bound+.5,bound-1.5,bound-1.,bound-.5,bound]:
            values.extend([np.nextafter(np.float32(v),np.float32(-np.inf)),np.float32(v),np.nextafter(np.float32(v),np.float32(np.inf))])
    for n in range(-130,131):
        v=np.float32(n+.5)
        values.extend([np.nextafter(v,np.float32(-np.inf)),v,np.nextafter(v,np.float32(np.inf))])
    bits=rng.integers(0,2**32,size=2048,dtype=np.uint32)
    values.extend(bits.view(np.float32)[~np.isnan(bits.view(np.float32))])
    bits=np.array(values,dtype=np.float32).view(np.uint32)
    modules=[];receipts=[]
    for width in (8,16,24):
        lo,hi=-(1<<(width-1)),(1<<(width-1))-1
        source=native(width=width,lower=f'{lo:.6e}',upper=f'{hi:.6e}')
        # The fixture's result ABI is widened explicitly, independently of the matcher.
        source=source.replace(f'define i{width} ', 'define i32 ')
        source=source.replace('@quantize(',f'@quantize{width}(')
        source=source.replace(f'ret i{width} %result',f'%wide = sext i{width} %result to i32\n ret i32 %wide')
        selected,r=rewrite(source,SOURCE)
        if len(r['routes'])!=1:raise ValueError(r)
        modules.append(selected.replace('declare float @roundevenf(float)',''));receipts.append(r)
    # The fixture uses @quantize; add the single compiler-owned source declaration.
    llvm='declare float @roundevenf(float)\n'+'\n'.join(modules)
    (out/'quantizers.ll').write_text(llvm)
    declarations='\n'.join(f'extern int quantize{w}(float);' for w in (8,16,24))
    c='''#include <stdint.h>
#include "htif.h"
extern float roundevenf(float);
'''+declarations+'\nstatic const uint32_t cases[] = {'+','.join(str(int(v))+'u' for v in bits)+'''};
int main(long hart) {
 if(hart)for(;;)__asm__ volatile("wfi");
 console_init();
 for(int mode=0;mode<5;mode++) {
  __asm__ volatile("csrw frm, %0"::"r"(mode):"memory");
  for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++) {
   union {uint32_t u;float f;} value={.u=cases[i]};
   for(int width=8;width<=24;width+=8) {
    int lo=-(1<<(width-1)),hi=(1<<(width-1))-1;
    float original=roundevenf(value.f);
    if(original<(float)lo)original=(float)lo;
    if(original>(float)hi)original=(float)hi;
    int expected=(int)original;
    int actual=width==8?quantize8(value.f):width==16?quantize16(value.f):quantize24(value.f);
    if(actual!=expected) {htif_puts("QUANT_RNE_FAIL ");htif_putd(i);htif_putc(' ');htif_putd(width);htif_putc(' ');htif_putd(mode);htif_putc('\\n');htif_exit(1);}
   }
  }
 }
 __asm__ volatile("csrwi frm, 0":::"memory");
 htif_puts("QUANT_RNE_PASS\\n");htif_exit(0);return 0;
}
'''
    (out/'main.c').write_text(c)
    baseline=REPO/'walkthrough/attention_optimization/variants/block_selected_o3/build'
    records=json.loads((baseline/'commands.json').read_text())
    main_command=next(r['argv'] for r in records if str(baseline/'main.c') in r['argv'] and '-c' in r['argv'])
    link_command=next(r['argv'] for r in records if '-nostdlib' in r['argv'])
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def run(argv):
        done=subprocess.run(list(map(str,argv)),env=env,capture_output=True,text=True,timeout=30)
        if done.returncode:raise RuntimeError(done.stderr+done.stdout)
        return done
    mapping={str(baseline/'main.c'):str(out.resolve()/'main.c'),str(baseline/'main.o'):str(out.resolve()/'main.o'),
             str(baseline/'model.o'):str(out.resolve()/'quantizers.o'),str(baseline/'attention_block.elf'):str(out.resolve()/'quantizers.elf'),
             '-Wl,-Map='+str(baseline/'attention_block.map'):'-Wl,-Map='+str(out.resolve()/'quantizers.map')}
    run([mapping.get(x,x) for x in main_command])
    llvm_command=next(r['argv'] for r in records if str(baseline/'combined.ll') in r['argv'] and '-c' in r['argv'])
    mapping.update({str(baseline/'combined.ll'):str(out.resolve()/'quantizers.ll')})
    run([mapping.get(x,x) for x in llvm_command])
    run([mapping.get(x,x) for x in link_command])
    execution=json.loads((baseline/'build_summary.json').read_text())['numerical_execution']['command'][:-1]+[str(out.resolve()/'quantizers.elf')]
    done=run(execution)
    if 'QUANT_RNE_PASS' not in done.stdout:raise ValueError(done.stdout)
    (ROOT/'edge_checks/result.json').write_text(json.dumps({'passed':True,'engine':'Spike RV64GC functional execution',
        'cases':len(bits),'widths':[8,16,24],'rounding_modes':[0,1,2,3,4],'comparisons':len(bits)*15,
        'oracle':'original runtime roundevenf, ordered clamps and signed conversion; NaN source poison excluded',
        'elf_sha256':hashlib.sha256((out/'quantizers.elf').read_bytes()).hexdigest(),
        'implementation_sha256':hashlib.sha256((REPO/'compiler_optimized/backend/host_quant_rne.py').read_bytes()).hexdigest(),
        'routes':receipts},indent=2))
    print('Passed',len(bits)*15,'RV64GC comparisons across five rounding modes')


if __name__=='__main__':main()
