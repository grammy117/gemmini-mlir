"""Record schedules from the frozen backend and compare actual LLVM bytes."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from backend.compiler import compile_buffer
from backend.source_groups import source_to_buffer

mixed, output = map(Path, sys.argv[1:])
rows = []
for source in sorted(mixed.glob('merlin_device_group_*/source.mlir')):
    original = source.with_name('device.mlir')
    if not original.exists():
        continue  # This call reuses a previously compiled kernel.
    cb = source_to_buffer(source.read_text(), transpose_transfers=True)
    cb.setdefault('params',{})['address_space_boundary'] = True
    inputs = ['arg0','arg1']
    cb['kernel_abi'] = {'kind':'whole_program','args':[
        *[{'tensor':name,'access':'read'} for name in inputs],
        {'tensor':'Y0','access':'write'}],'outputs':['Y0']}
    cb, llvm = compile_buffer(cb, symbol=source.parent.name+'_kernel')
    if llvm != original.read_text():
        raise RuntimeError('device re-emission differs from original compiler output: '+str(source))
    source.with_name('commands.json').write_text(json.dumps(cb,indent=2))
    rows.append({'source':str(source),'commands':str(source.with_name('commands.json')),
                 'llvm':str(original),'identical_reemission':True,
                 'llvm_sha256':hashlib.sha256(llvm.encode()).hexdigest(),
                 'lowering':cb.get('params',{}).get('compiler_lowering')})
output.write_text(json.dumps(rows,indent=2))
print('Recorded and reproduced',len(rows),'checked kernels')
