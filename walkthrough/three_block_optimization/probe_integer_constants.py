"""Explore exact folding of small closed integer index tensor producers.

This is an experimental source preparation, not an enabled compiler pass.
Floating arithmetic, runtime operands, overflow and unknown annotations refuse.
"""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'compiler_optimized'),str(REPO/'compiler_optimized/compiler_library')]


def fold(source):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    from xdsl.dialects.builtin import TensorType,IntegerType,IntegerAttr,DenseIntOrFPElementsAttr,NoneAttr
    from xdsl.dialects.arith import ConstantOp
    from xdsl.ir.affine import AffineMap
    module=parse_mlir_text(source);rows=[]
    for op in tuple(module.walk()):
        if op.name!='linalg.generic' or len(op.results)!=1 or len(op.operands)!=1:continue
        t=op.results[0].type
        if not isinstance(t,TensorType) or not isinstance(t.element_type,IntegerType) or not isinstance(t.encoding,NoneAttr):continue
        shape=t.get_shape();count=1
        for n in shape:count*=n
        if not shape or any(n<=0 for n in shape) or count>4096:continue
        if any(i.data.value!='parallel' for i in op.iterator_types):continue
        if len(op.indexing_maps)!=1 or op.indexing_maps.data[0].data!=AffineMap.identity(len(shape)):continue
        allowed={'indexing_maps','iterator_types','operandSegmentSizes'}
        if any(k not in allowed and not k.startswith('prov.') for k in (*op.attributes,*op.properties)):continue
        if len(op.regions)!=1 or len(op.regions[0].blocks)!=1:continue
        block=op.regions[0].block
        if len(block.args)!=1 or block.args[0].uses:continue
        if any(i.name not in {'linalg.index','arith.constant','arith.index_cast','arith.addi','arith.muli','linalg.yield'} for i in block.ops):continue
        values=[]
        try:
            for position in itertools.product(*(range(n) for n in shape)):
                env={};yielded=None
                for i in block.ops:
                    if any(not k.startswith('prov.') and k not in {'value','dim','overflowFlags'} for k in (*i.attributes,*i.properties)):raise ValueError('unknown annotation')
                    if 'overflowFlags' in i.properties and i.properties['overflowFlags'].data:raise ValueError('integer flags')
                    if i.name=='linalg.yield':yielded=env[i.operands[0]];continue
                    if i.name=='linalg.index':v=position[i.dim.value.data]
                    elif i.name=='arith.constant':
                        if not isinstance(i.value,IntegerAttr):raise ValueError('non-integer constant')
                        v=i.value.value.data
                    elif i.name=='arith.index_cast':v=env[i.operands[0]]
                    else:
                        a,b=(env[x] for x in i.operands)
                        v=a+b if i.name=='arith.addi' else a*b
                    dtype=i.results[0].type
                    width=dtype.width.data if isinstance(dtype,IntegerType) else 64
                    if not -(1<<(width-1))<=v<(1<<(width-1)):raise ValueError('overflow')
                    env[i.results[0]]=v
                if yielded is None:raise ValueError('missing yield')
                values.append(yielded)
        except (KeyError,ValueError,IndexError):continue
        constant=ConstantOp(DenseIntOrFPElementsAttr.from_list(t,values))
        constant.attributes.update({k:v for k,v in op.attributes.items() if k.startswith('prov.')})
        op.parent.insert_op_before(constant,op);op.results[0].replace_all_uses_with(constant.result)
        op.parent.erase_op(op);rows.append({'shape':list(shape),'type':str(t),'elements':len(values)})
    module.verify();return text(module,generic=True),rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();shutil.copytree(a.capture,a.output)
    selected,rows=fold((a.capture/'model.mlir').read_text())
    (a.output/'model.mlir').write_text(selected);(a.output/'integer-fold.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows))


if __name__=='__main__':main()
