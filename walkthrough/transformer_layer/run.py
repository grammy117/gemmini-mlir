"""Run the ELF with the installed Gemmini functional extension and check all outputs."""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BUILD = ROOT/'build'
SPIKE = Path('/scratch/grammy/blast/chipyard/.conda-env/riscv-tools/bin/spike')
EXTENSION = Path('/scratch/grammy/blast/merlin-tvm-tuning/out/build/baselines/tvm-gemmini/simulator-a26/libgemmini.so')
ELF = BUILD/'transformer_layer.elf'

def main():
    command = [str(SPIKE),'--isa=rv64gc','-p1','-m256','--instructions=20000000',
               '--extlib='+str(EXTENSION),'--extension=gemmini',str(ELF)]
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '1'
    start = time.monotonic()
    run = subprocess.run(command,capture_output=True,text=True,env=env,timeout=30)
    (BUILD/'spike.stdout.log').write_text(run.stdout)
    (BUILD/'spike.stderr.log').write_text(run.stderr)
    matches = re.findall(r'^OUT (\d+) (0x[0-9a-fA-F]+)$',run.stdout,re.MULTILINE)
    indices = [int(i) for i,_ in matches]
    complete = indices == list(range(256)) and 'TRANSFORMER_END' in run.stdout
    report = {'command':command,'seconds':time.monotonic()-start,'returncode':run.returncode,
              'elf_sha256':hashlib.sha256(ELF.read_bytes()).hexdigest(),
              'extension_sha256':hashlib.sha256(EXTENSION.read_bytes()).hexdigest(),
              'engine':'Spike plus Gemmini functional model; no timing claim',
              'complete_output':complete,'observed_elements':len(matches),
              'comparison':'quantized PyTorch layer, all 256 float32 outputs',
              'tolerance':{'atol':1e-5,'rtol':1e-5}}
    if complete:
        actual=np.array([int(bits,16) for _,bits in matches],dtype=np.uint32).view(np.float32).reshape(1,8,32)
        golden=np.load(ROOT/'capture/golden.npy',allow_pickle=False)
        np.save(BUILD/'actual.npy',actual)
        report.update(max_abs_error=float(np.max(np.abs(actual-golden))),
                      exact_bits=bool(np.array_equal(actual.view(np.uint32),golden.view(np.uint32))),
                      passed=bool(run.returncode==0 and np.allclose(actual,golden,atol=1e-5,rtol=1e-5)))
    else:
        report['passed']=False
    (BUILD/'execution_summary.json').write_text(json.dumps(report,indent=2))
    summary=json.loads((BUILD/'build_summary.json').read_text())
    summary['numerical_execution']=report
    (BUILD/'build_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(report,indent=2))
    if not report['passed']:
        raise SystemExit(1)

if __name__=='__main__':
    main()
