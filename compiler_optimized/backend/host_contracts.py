"""Retain allocation and owned scalar-math contracts in linked LLVM.

No value expression changes. The standard allocator's result is fresh; the
compiler's already memory(none) math lowerings terminate without unwinding.
User-declared functions with these names never acquire inferred contracts.
"""
import hashlib
import re


def rewrite(source,routed_source):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    module=parse_mlir_text(routed_source);ops=tuple(module.walk())
    report={'schema':'gemmini.host_llvm_contracts.v1','allocators':[],'math':[],
            'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'routed_source_sha256':hashlib.sha256(routed_source.encode()).hexdigest()}
    if any('strictfp' in str(k) or 'strictfp' in str(v) or 'fenv' in str(k)
           for op in ops for k,v in (*op.attributes.items(),*op.properties.items())):
        report['refusal']='strict floating environment'
        report['output_sha256']=report['source_sha256'];return source,report
    owned={op.sym_name.data for op in ops if op.name=='func.func'}
    owned.update(op.callee.root_reference.data for op in ops if op.name=='func.call')
    names={'math.rsqrt':'rsqrt','math.sqrt':'sqrt','math.exp':'exp','math.sin':'sin',
           'math.cos':'cos','math.powf':'pow','math.log':'log','math.tanh':'tanh','math.erf':'erf'}
    math_symbols=set()
    for op in ops:
        if op.name in names and op.results and str(op.results[0].type) in ('f32','f64'):
            math_symbols.add(names[op.name]+('f' if str(op.results[0].type)=='f32' else ''))
    groups=dict(re.findall(r'^attributes #(\d+) = \{([^\n]*)\}',source,re.M))
    lines=[]
    for line in source.splitlines():
        allocator=re.fullmatch(r'declare ptr @(malloc|aligned_alloc)\((i64(?:\s*,\s*i64)?)\)',line)
        if allocator and allocator[1] not in owned and len(allocator[2].split(','))=={'malloc':1,'aligned_alloc':2}[allocator[1]]:
            line=line.replace('declare ptr','declare noalias ptr',1)
            report['allocators'].append(allocator[1])
        declaration=re.fullmatch(r'declare (float|double) @([A-Za-z0-9_.]+)\(([^)]*)\)(.*)',line)
        if declaration and declaration[2] in math_symbols-owned:
            suffix=declaration[4];group=re.search(r'#(\d+)',suffix)
            attributes=suffix+' '+(groups.get(group[1],'') if group else '')
            if 'memory(none)' in attributes:
                added=[a for a in ('nounwind','willreturn') if a not in attributes.split()]
                if added:
                    line+=' '+' '.join(added)
                    report['math'].append({'symbol':declaration[2],'added':added})
        lines.append(line)
    output='\n'.join(lines)+'\n'
    report['output_sha256']=hashlib.sha256(output.encode()).hexdigest()
    return output,report
