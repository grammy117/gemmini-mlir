"""Lower proved disjoint static strided copies to typed scalar loop producers.

Distinct fresh allocation roots establish nonaliasing. Contiguous copies retain
the stock memcpy path; unknown roots, dynamic shapes, annotations and layouts
outside the ranked host memref ABI keep the original memref.copy operation.
"""
from __future__ import annotations

import hashlib
from collections import Counter


def _root(value):
    views={'memref.subview','memref.cast','memref.collapse_shape','memref.expand_shape',
           'memref.reinterpret_cast','memref.view'}
    seen=set()
    while value not in seen:
        seen.add(value);owner=value.owner
        if getattr(owner,'name','') not in views:return owner
        if not owner.operands:return None
        value=owner.operands[0]
    return None


def rewrite(source):
    from xdsl.dialects.builtin import (MemRefType,NoneAttr,StridedLayoutAttr,AffineMapAttr,
                                      IntegerType,Float16Type,Float32Type,Float64Type,BFloat16Type)
    from xdsl.dialects.linalg.ops import GenericOp,YieldOp
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
    from xdsl.ir import Block,Region
    from xdsl.ir.affine import AffineMap
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    module=parse_mlir_text(source);module.verify()
    rows=[];refusals=Counter()
    for op in tuple(module.walk()):
        if op.name!='memref.copy':continue
        if any(not k.startswith('prov.') for k in (*op.attributes,*op.properties)):
            refusals['unknown_copy_annotation']+=1;continue
        src,dst=op.operands;st,dt=src.type,dst.type
        if (not isinstance(st,MemRefType) or not isinstance(dt,MemRefType)
                or st.get_shape()!=dt.get_shape() or st.element_type!=dt.element_type
                or st.memory_space!=dt.memory_space or not isinstance(st.memory_space,NoneAttr)
                or not 1<=len(st.get_shape())<=8 or any(x<=0 for x in st.get_shape())):
            refusals['unsupported_static_host_memref']+=1;continue
        element=st.element_type
        if not (isinstance(element,(Float16Type,Float32Type,Float64Type,BFloat16Type))
                or isinstance(element,IntegerType) and element.width.data>=8 and element.width.data%8==0):
            refusals['unsupported_element_representation']+=1;continue
        if isinstance(st.layout,NoneAttr) and isinstance(dt.layout,NoneAttr):
            refusals['already_contiguous']+=1;continue
        if any(not isinstance(t.layout,(NoneAttr,StridedLayoutAttr)) for t in (st,dt)):
            refusals['unsupported_layout']+=1;continue
        sr,dr=_root(src),_root(dst)
        fresh={'memref.alloc','memref.alloca'};known=fresh|{'memref.get_global'}
        if (sr is None or dr is None or sr is dr or getattr(sr,'name','') not in known
                or getattr(dr,'name','') not in known or not ({sr.name,dr.name}&fresh)):
            refusals['unproved_disjoint_roots']+=1;continue
        body=Block(arg_types=[st.element_type,st.element_type]);body.add_op(YieldOp(body.args[0]))
        rank=len(st.get_shape())
        generic=GenericOp([src],[dst],Region(body),[AffineMapAttr(AffineMap.identity(rank))]*2,
                          [IteratorTypeAttr.parallel()]*rank,result_types=[])
        generic.attributes.update(op.attributes)
        op.parent.insert_op_before(generic,op);op.parent.erase_op(op)
        rows.append({'shape':list(st.get_shape()),'element_type':str(st.element_type),
                     'source_root':sr.name,'destination_root':dr.name,
                     'proof':'distinct fresh/global allocation roots; same static domain and element representation'})
    module.verify()
    output=text(module,generic=True) if rows else source
    return output,{'schema':'gemmini.static_host_copies.v1','copies':rows,'refusals':dict(refusals),
                   'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                   'output_sha256':hashlib.sha256(output.encode()).hexdigest()}
