"""Set helper visibility for the materializer's single-entry executable ABI.

External runtime declarations and the public entry are unchanged. Helpers with
explicit no-inline or unusual linkage contracts retain those restrictions.
"""
import hashlib
import re


def rewrite(source, *, entry='kernel_entry', policy='inline_all'):
    if policy not in ('none','internalize','inline_all'):raise ValueError('unknown native helper policy')
    if not entry.isascii() or not entry.isidentifier():raise ValueError('invalid executable entry')
    groups=dict(re.findall(r'^attributes #([0-9]+) = \{([^\n]*)\}',source,re.M))
    report={'schema':'gemmini.native_helpers.v1','entry':entry,'policy':policy,'helpers':[],
            'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    lines=[];entries=0
    for line in source.splitlines():
        if not line.startswith('define '):lines.append(line);continue
        match=re.search(r'@([A-Za-z_.$][A-Za-z_.$0-9]*)\(',line)
        if not match or not line.rstrip().endswith('{'):raise ValueError('unsupported LLVM definition header')
        name=match[1]
        if name==entry:entries+=1;lines.append(line);continue
        if policy=='none':lines.append(line);continue
        if re.match(r'^define (?:available_externally|linkonce(?:_odr)?|weak(?:_odr)?|common|appending|extern_weak|dllimport|dllexport)\b',line):
            report['helpers'].append({'symbol':name,'refused':'unusual linkage'});lines.append(line);continue
        if re.search(r'\b(hidden|protected)\b',line.split('@',1)[0]):
            report['helpers'].append({'symbol':name,'refused':'explicit visibility'});lines.append(line);continue
        line=re.sub(r'^define (?:(?:private|internal|dso_local) )*','define internal ',line)
        attributes=line+' '+' '.join(groups.get(g,'') for g in re.findall(r'#([0-9]+)',line))
        forbidden=re.search(r'\b(noinline|optnone|naked|returns_twice|presplitcoroutine)\b',attributes)
        inline=policy=='inline_all' and not forbidden
        if inline and not re.search(r'\balwaysinline\b',attributes):line=line.rstrip()[:-1]+' alwaysinline {'
        report['helpers'].append({'symbol':name,'internal':True,'alwaysinline':inline,
                                   'inline_refusal':forbidden[1] if forbidden else None})
        lines.append(line)
    if entries!=1:raise ValueError('single-entry executable definition required')
    output='\n'.join(lines)+'\n' if policy!='none' else source
    report['output_sha256']=hashlib.sha256(output.encode()).hexdigest()
    return output,report
