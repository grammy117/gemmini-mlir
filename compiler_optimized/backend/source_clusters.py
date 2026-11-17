"""Expose independent shared-operand products to the existing group compiler.

Only immutable tensor/scalar work may be crossed. Every real input must already
dominate the first product; live initializers, calls, effects and control flow
refuse motion. Original zero constructors are cloned into the closed group.
The existing region closure checks and target schedule proofs still run later.
"""
from __future__ import annotations

from .source_cse import _pure_body


def _pure(op):
    from xdsl.dialects.builtin import TensorType, MemRefType, UnrankedMemRefType
    if any('strictfp' in str(k) or 'strictfp' in str(v)
           for inner in op.walk() for k,v in (*inner.attributes.items(),*inner.properties.items())):
        return False
    if op.name.startswith(('arith.','math.')):
        from xdsl.traits import Pure
        return not op.regions and (op.has_trait(Pure) or op.name == 'arith.extsi')
    if op.name.startswith('tensor.'):
        from xdsl.traits import NoMemoryEffect
        return not op.regions and op.has_trait(NoMemoryEffect)
    return (op.name.startswith('linalg.') and _pure_body(op) and bool(op.results)
            and all(isinstance(v.type,TensorType) for v in op.results)
            and all(not isinstance(v.type,(MemRefType,UnrankedMemRefType)) for v in op.operands))


def _clone_initializer(value, mapping, operations):
    if value in mapping:
        return mapping[value]
    owner=value.owner
    if (getattr(owner,'name','') not in {'arith.constant','tensor.empty','tensor.splat','linalg.fill'}
            or len(operations)>=32 or not _pure(owner)):
        raise ValueError('initializer is not a bounded closed constant constructor')
    for operand in owner.operands:
        _clone_initializer(operand,mapping,operations)
    clone=owner.clone(value_mapper=mapping)
    operations.append(clone)
    return mapping[value]


def cluster_products(module, *, max_members=8):
    from xdsl.ir import BlockArgument
    from .frontend import _zero
    from .source_groups import _contraction_plans
    from .source_shared_groups import _raw_operands
    module.verify()
    plans=tuple(_contraction_plans(module))
    rows=[]
    index=0
    while index<len(plans):
        first=plans[index]
        original=_raw_operands(first)
        if original is None or not _zero(original[2]):
            index+=1;continue
        block=first.root.parent
        if block is None:
            index+=1;continue
        function=first.root.parent_op()
        if function is None or any('strictfp' in str(k) or 'strictfp' in str(v)
                for k,v in (*function.attributes.items(),*function.properties.items())):
            index+=1;continue
        positions={op:i for i,op in enumerate(block.ops)}
        anchor=positions[first.root]
        lhs,rhs=original[:2]
        members=[first]
        for candidate in plans[index+1:index+max_members]:
            operands=_raw_operands(candidate)
            if operands is None or candidate.root.parent is not block or not _zero(operands[2]):
                break
            next_lhs=lhs if operands[0] is lhs else None
            next_rhs=rhs if operands[1] is rhs else None
            if next_lhs is None and next_rhs is None:
                break
            if any(not (isinstance(v,BlockArgument) and v.owner is block)
                   and (getattr(v.owner,'parent',None) is not block or positions.get(v.owner,anchor)>=anchor)
                   for v in operands[:2]):
                break
            end=positions[candidate.root]
            if end<=anchor or any(not _pure(op) for op in tuple(block.ops)[anchor:end+1]):
                break
            members.append(candidate)
            lhs,rhs=next_lhs,next_rhs
        if len(members)<2:
            index+=1;continue
        constructors,chunks=[],[]
        try:
            seeds=[]
            for plan in members:
                start=len(constructors)
                # Private seeds keep every candidate prefix a closed region.
                seeds.append(_clone_initializer(_raw_operands(plan)[2],{},constructors))
                chunks.append(tuple(constructors[start:]))
        except ValueError:
            for op in reversed(constructors):op.drop_all_references()
            index+=1;continue
        previous=first.root
        for ordinal,(plan,seed,chunk) in enumerate(zip(members,seeds,chunks,strict=True)):
            operands=list(plan.root.operands)
            operands[-1]=seed
            plan.root.operands=operands
            if ordinal:
                plan.root.detach()
                block.insert_op_after(plan.root,previous)
            for op in chunk:block.insert_op_before(op,plan.root)
            previous=plan.root
        rows.append({'members':len(members),'shared':'lhs' if lhs is not None else 'rhs',
                     'shapes':[[list(v.type.get_shape()) for v in _raw_operands(p)[:2]] for p in members],
                     'proof':'same SSA operand, dominating inputs, pure crossed operations, exact cloned zero seeds'})
        index+=len(members)
    module.verify()
    return rows


def optimize_source(source):
    import hashlib
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    module=parse_mlir_text(source)
    rows=cluster_products(module)
    output=text(module,generic=True) if rows else source
    return output,{'schema':'optimized.shared_product_clusters.v1','clusters':rows,
                   'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                   'output_sha256':hashlib.sha256(output.encode()).hexdigest()}
