"""Share identical pure tensor computations before device region discovery.

Matching uses operand SSA identity, complete types/maps/scalar programs and all
semantic attributes. Only provenance is ignored and merged. Static, unused
tensor.empty destinations may differ because their contents are never read.
No arithmetic is reassociated, approximated, or selected from model names.
"""
from __future__ import annotations

import hashlib
from collections import Counter

from xdsl.dialects.builtin import NoneAttr, TensorType
from xdsl.traits import Pure

from merlin.llvmlower.tensor_provenance_cse import (
    _retain_provenance, _stripped, _supported as preparation_supported,
)


def _pure_body(op):
    return all(
        inner is op or inner.name == 'linalg.yield' or (
            not inner.regions and inner.name.startswith(('arith.', 'math.', 'linalg.index'))
            # These two registered operations lack Pure in the pinned xDSL.
            and (inner.has_trait(Pure) or inner.name in {'arith.extsi','linalg.index'})
        ) for inner in op.walk()
    )


def _eligible(op):
    if any('strictfp' in str(k) or 'strictfp' in str(v)
           for inner in op.walk() for k,v in (*inner.attributes.items(),*inner.properties.items())):
        return False
    if preparation_supported(op):
        return True
    if (op.name != 'linalg.generic' or len(op.results) != 1 or len(op.outputs) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
        return False
    allowed={'indexing_maps','iterator_types','operandSegmentSizes'}
    if any(k not in allowed and not k.startswith('prov.') for k in (*op.attributes,*op.properties)):
        return False
    if any(not isinstance(v.type, TensorType) or not isinstance(v.type.encoding, NoneAttr)
           or any(d <= 0 for d in v.type.get_shape()) for v in (*op.operands, *op.results)):
        return False
    if any(k not in {'indexing_maps', 'iterator_types', 'operandSegmentSizes'}
           and not k.startswith('prov.') for k in (*op.attributes, *op.properties)):
        return False
    block = op.regions[0].block
    if len(block.args) != len(op.operands) or block.args[-1].uses:
        return False
    from xdsl.ir.affine import AffineMap
    rank = len(op.results[0].type.get_shape())
    return (len(op.iterator_types) == rank
            and all(v.data.value == 'parallel' for v in op.iterator_types)
            and op.indexing_maps.data[-1].data == AffineMap.identity(rank)
            and op.results[0].type == op.outputs[0].type and _pure_body(op))


def _return_cone(op):
    pending, seen = list(op.results), set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        for use in value.uses:
            if use.operation.name == 'func.return':
                return True
            if _eligible(use.operation):
                pending.extend(use.operation.results)
    return False


def _barrier(op):
    # Tensor SSA computations are immutable. Unknown calls/memory/control keep
    # their own boundaries; no borrowed pointer or external purity is inferred.
    if op.name.startswith(('arith.', 'math.', 'tensor.')):
        return False
    if op.name.startswith('linalg.'):
        return not _pure_body(op)
    return True


def optimize_module(module):
    from xdsl.dialects.tensor import EmptyOp
    counts = Counter()
    for fn in tuple(module.body.block.ops):
        if fn.name != 'func.func' or not fn.body.blocks:
            continue
        if any('strictfp' in str(k) or 'strictfp' in str(v)
               for k, v in (*fn.attributes.items(), *fn.properties.items())):
            continue
        for block in fn.body.blocks:
            candidates = {}
            for op in tuple(block.ops):
                if _barrier(op):
                    candidates.clear()
                    continue
                if not _eligible(op) or _return_cone(op):
                    continue
                operands = tuple(op.operands)
                destination = None
                if op.name.startswith('linalg.'):
                    if not isinstance(operands[-1].owner, EmptyOp):
                        continue
                    operands, destination = operands[:-1], operands[-1]
                key = (op.name, operands, tuple(v.type for v in op.results))
                normalized = _stripped(op)
                match = None
                for previous in candidates.get(key, ()):
                    other = _stripped(previous)
                    context = {} if destination is None else {destination: previous.operands[-1]}
                    equal = normalized.is_structurally_equivalent(other, context)
                    other.drop_all_references()
                    if equal:
                        match = previous
                        break
                normalized.drop_all_references()
                if match is None:
                    candidates.setdefault(key, []).append(op)
                    continue
                _retain_provenance(match, op)
                for old, new in zip(op.results, match.results, strict=True):
                    old.replace_all_uses_with(new)
                block.erase_op(op)
                counts[op.name] += 1
    module.verify()
    return dict(sorted(counts.items()))


def optimize_source(source):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    module = parse_mlir_text(source)
    changes = optimize_module(module)
    output = text(module, generic=True) if changes else source
    return output, {'schema':'optimized.pure_tensor_cse.v1', 'rewrites':changes,
                    'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                    'output_sha256':hashlib.sha256(output.encode()).hexdigest()}
