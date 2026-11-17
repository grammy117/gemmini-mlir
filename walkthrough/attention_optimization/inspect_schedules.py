"""Reproduce every emitted kernel, including plural result groups."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from backend.compiler import compile_buffer
from backend.source_groups import source_to_buffer
from backend.frontend import lower_linalg
from merlin.frontends.linalg_mlir import parse_mlir_text

mixed, output = map(Path, sys.argv[1:])
rows = []
for source in sorted(mixed.glob('merlin_device_group_*/source.mlir')):
    original = source.with_name('device.mlir')
    if not original.exists():
        continue
    text = source.read_text()
    module = parse_mlir_text(text)
    fn = next(op for op in module.body.block.ops if op.name == 'func.func' and op.body.blocks)
    returns = fn.body.block.last_op.operands
    if len(returns)>1:
        cb=lower_linalg(text)
        inputs=cb['params']['frontend_entry_bindings']
        outputs=cb['params']['frontend_output_bindings']
        for key in ('frontend_entry_bindings','frontend_output_bindings','frontend_source_values'):
            cb['params'].pop(key)
    else:
        cb=source_to_buffer(text,transpose_transfers=True)
        inputs=[f'arg{i}' for i in range(len(fn.body.block.args))]
        outputs=['Y0']
    cb.setdefault('params',{})['address_space_boundary']=True
    cb['kernel_abi']={'kind':'whole_program','args':[
        *[{'tensor':name,'access':'read'} for name in inputs],
        *[{'tensor':name,'access':'write'} for name in outputs]],'outputs':list(outputs)}
    cb, llvm=compile_buffer(cb,symbol=source.parent.name+'_kernel')
    if llvm!=original.read_text():
        raise RuntimeError('device re-emission differs: '+str(source))
    source.with_name('commands.json').write_text(json.dumps(cb,indent=2))
    rows.append({'source':str(source),'commands':str(source.with_name('commands.json')),
                 'llvm':str(original),'identical_reemission':True,'results':len(outputs),
                 'llvm_sha256':hashlib.sha256(llvm.encode()).hexdigest(),
                 'lowering':cb.get('params',{}).get('compiler_lowering')})
output.write_text(json.dumps(rows,indent=2))
print('Recorded and reproduced',len(rows),'checked kernels')
