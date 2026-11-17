"""Bounded scalar round/clamp/convert after the owned host lowering.

Consume actual typed LLVM and the routed tensor source, never provenance or a
model name. Integral signed bounds commute with ties-even rounding. The target
conversion names RNE explicitly, independently of the dynamic rounding mode.
Only a closed, single-consumer chain from an owned math.roundeven lowering may
be removed. Unknown callees, extra observers, flags and strict FP retain source.
"""
from __future__ import annotations

import hashlib
from collections import Counter

from merlin.llvmlower.late_quant_rne import _tokens, _functions, _instruction, _identity, _number


def _clamp(value, definitions):
    select=definitions.get(_identity(value))
    if select is None or select.opcode!='select' or select.dtype!='float':return None
    condition=definitions.get(_identity(select.operands[0]))
    if condition is None or condition.opcode!='fcmp' or condition.dtype!='float':return None
    left,right=condition.operands
    a,b=select.operands[1:]
    # Ordered comparisons must propagate NaN through the false/value arm.
    constant=_number(a)
    if constant is None:return None
    if (left,right)==(b,a):
        kinds={'olt':'lower','ogt':'upper'}
    elif (left,right)==(a,b):
        kinds={'olt':'upper','ogt':'lower'}
    else:return None
    kind=kinds.get(condition.predicate)
    if kind is None:return None
    return b,kind,constant,(condition,select)


def rewrite(llvm_text, routed_source, *, host_isa='rv64gc', emit_call_contracts=False):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    module=parse_mlir_text(routed_source)
    report={'schema':'gemmini.scalar_bounded_quant_rne.v1','routes':[],
            'source_sha256':hashlib.sha256(llvm_text.encode()).hexdigest(),
            'routed_source_sha256':hashlib.sha256(routed_source.encode()).hexdigest(),
            'host_isa':host_isa,'refusals':{},'contract':'defined scalar f32-to-signed-int results; explicit ties-even; NaN fptosi is poison'}
    def unchanged(reason):
        report['refusals'][reason]=1
        report['output_sha256']=report['source_sha256']
        return llvm_text,report
    if host_isa!='rv64gc':return unchanged('unsupported_host_isa')
    if any(x in llvm_text for x in ('strictfp','llvm.experimental.constrained')):
        return unchanged('strict_floating_environment')
    operations=tuple(module.walk())
    if any('strictfp' in str(k) or 'strictfp' in str(v) or 'fenv' in str(k)
           for op in operations for k,v in (*op.attributes.items(),*op.properties.items())):
        return unchanged('strict_source_floating_environment')
    if any('strictfp' in str(k) or 'strictfp' in str(v)
           for op in operations for k,v in (*op.attributes.items(),*op.properties.items())):
        return unchanged('strict_source_floating_environment')
    if any(op.name=='func.func' and op.sym_name.data=='roundevenf' for op in operations):
        return unchanged('source_owned_roundevenf_symbol')
    if any(op.name=='func.call' and op.callee.root_reference.data=='roundevenf' for op in operations):
        return unchanged('source_owned_roundevenf_call')
    rounds=sum(op.name=='math.roundeven' for op in operations)
    if not rounds:return unchanged('no_owned_roundeven_source')
    tokens=_tokens(llvm_text)
    functions=_functions(tokens)
    # A source operation may disappear in canonicalization, never multiply here.
    call_count=sum(t.text=='@roundevenf' for t in tokens)-sum(
        t.text=='declare' and any(x.text=='@roundevenf' for x in tokens[i+1:i+6])
        for i,t in enumerate(tokens))
    if call_count>rounds:return unchanged('host_round_call_count_not_source_bound')
    edits=[];refusals=Counter()
    for function in functions:
        definitions={}
        for i in range(len(function)):
            op=_instruction(function,i)
            if op is not None:definitions[_identity(op.result)]=op
        for result in tuple(definitions.values()):
            if result.opcode!='fptosi' or result.dtype!='float':continue
            dtype=result.output_type
            if not dtype.startswith('i') or not dtype[1:].isdigit() or not 1<=int(dtype[1:])<=24:
                refusals['unsupported_integer_width']+=1;continue
            current=result.operands[0];bounds={};chain=[]
            for _ in range(2):
                match=_clamp(current,definitions)
                if match is None:break
                current,kind,bound,ops=match
                bounds[kind]=bound;chain.extend(ops)
            width=int(dtype[1:]);expected={'lower':-(1<<(width-1)),'upper':(1<<(width-1))-1}
            if bounds!=expected or len(chain)!=4:
                refusals['unproved_complete_integral_bounds']+=1;continue
            rounded=definitions.get(_identity(current))
            if (rounded is None or rounded.opcode!='call' or rounded.dtype!='float'
                    or rounded.callee!='roundevenf' or len(rounded.operands)!=1):
                refusals['unproved_roundeven_root']+=1;continue
            chain.append(rounded)
            spans=[(op.start,op.end) for op in (*chain,result)]
            names={_identity(op.result) for op in chain}
            if any(t.text.startswith('%') and _identity(t.text) in names
                   and not any(start<=t.start<end for start,end in spans) for t in function):
                refusals['observed_intermediate']+=1;continue
            stem='merlin.quant.rne';ordinal=len(report['routes'])
            temporary=f'%{stem}.{ordinal}'
            while any(t.text==temporary for t in function):temporary+='x'
            lower,upper=expected['lower'],expected['upper']
            raw=rounded.operands[0]
            replacement=(f'{temporary} = call i32 asm "fmax.s ft0, $1, $2\\0Afmin.s ft0, ft0, $3\\0A'
                         f'fcvt.w.s $0, ft0, rne", "=r,f,f,f,~{{ft0}}"'
                         f'(float {raw}, float {lower:.6e}, float {upper:.6e})'
                         +(' nounwind willreturn memory(none)' if emit_call_contracts else '')+'\n'
                         f'  {result.result} = trunc i32 {temporary} to {dtype}')
            edits.append((result.start,result.end,replacement))
            edits.extend((op.start,op.end,'') for op in chain)
            report['routes'].append({'integer_dtype':dtype,'bounds':[lower,upper],
                                      'source_form':'owned roundevenf -> ordered compare/select clamp -> fptosi',
                                      'closed_intermediates':len(chain),'rounding':'explicit RNE'})
    edits.sort()
    if any(a[1]>b[0] for a,b in zip(edits,edits[1:])):
        raise ValueError('bounded rounding proposals overlap')
    output=llvm_text
    for start,end,replacement in reversed(edits):output=output[:start]+replacement+output[end:]
    report['refusals']=dict(refusals)
    report['output_sha256']=hashlib.sha256(output.encode()).hexdigest()
    return output,report
