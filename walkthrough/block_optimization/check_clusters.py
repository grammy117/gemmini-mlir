"""Check source motion and closure at actual SSA/effect boundaries."""
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'compiler_optimized'),str(REPO/'compiler_optimized/compiler_library')]
from backend.source_clusters import optimize_source,cluster_products
from backend.source_groups import discover_regions
from merlin.frontends.linalg_mlir import parse_mlir_text


def product(index,lhs='%x',rhs=None,init=None):
    rhs=rhs or f'%b{index}'
    seed=init or f'%s{index}'
    return f'''
    %z{index}=arith.constant 0 : i32
    %s{index}=tensor.splat %z{index} : tensor<4x12xi32>
    %p{index}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]}}
      ins({lhs},{rhs} : tensor<4x8xi8>,tensor<8x12xi8>) outs({seed} : tensor<4x12xi32>) {{
      ^bb0(%a:i8,%b:i8,%c:i32):
        %a32=arith.extsi %a : i8 to i32
        %b32=arith.extsi %b : i8 to i32
        %mul=arith.muli %a32,%b32 : i32
        %sum=arith.addi %c,%mul : i32
        linalg.yield %sum : i32
      }} -> tensor<4x12xi32>
    '''


def source(*,members=2,shared=True,live=False,opaque=False,late=False,strict=False):
    attributes='attributes {strictfp="yes"}' if strict else ''
    body=product(0,init='%init' if live else None)
    body+='func.call @opaque() : () -> ()' if opaque else '%host=arith.constant 7 : i32'
    if late:body+='\n%late=tensor.cast %b1 : tensor<8x12xi8> to tensor<8x12xi8>'
    for i in range(1,members):
        body+=product(i,lhs='%x' if shared else '%other',rhs='%late' if late and i==1 else None,init='%init' if live else None)
    results=','.join(f'%p{i}' for i in range(members))
    types=','.join(['tensor<4x12xi32>']*members)
    return f'''builtin.module {{
      func.func private @opaque()
      func.func @forward(%x:tensor<4x8xi8>,%other:tensor<4x8xi8>,%b0:tensor<8x12xi8>,%b1:tensor<8x12xi8>,%b2:tensor<8x12xi8>,%init:tensor<4x12xi32>)->({types}) {attributes} {{
        {body}
        func.return {results} : {types}
      }}
    }}'''


class ClusterChecks(unittest.TestCase):
    def test_independent_products_group_across_pure_work(self):
        output,report=optimize_source(source())
        self.assertEqual([r['members'] for r in report['clusters']],[2])
        self.assertEqual([len(r.outputs) for r in discover_regions(parse_mlir_text(output),transpose_transfers=True)],[2])

    def test_three_private_seeds_form_closed_prefixes(self):
        output,_=optimize_source(source(members=3))
        self.assertEqual([len(r.outputs) for r in discover_regions(parse_mlir_text(output),transpose_transfers=True)],[3])

    def test_different_ssa_input_refuses(self):self.assertFalse(optimize_source(source(shared=False))[1]['clusters'])
    def test_live_initializers_refuse(self):self.assertFalse(optimize_source(source(live=True))[1]['clusters'])
    def test_opaque_call_is_barrier(self):self.assertFalse(optimize_source(source(opaque=True))[1]['clusters'])
    def test_input_available_only_later_refuses(self):self.assertFalse(optimize_source(source(late=True))[1]['clusters'])
    def test_strict_fp_function_refuses(self):self.assertFalse(optimize_source(source(strict=True))[1]['clusters'])

    def test_constructor_outside_motion_interval_is_checked(self):
        from xdsl.dialects.builtin import StringAttr
        module=parse_mlir_text(source())
        first=next(op for op in module.walk() if op.name=='tensor.splat')
        first.attributes['strictfp']=StringAttr('preserve')
        self.assertFalse(cluster_products(module))


if __name__=='__main__':unittest.main()
