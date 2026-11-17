"""Semantic positive/negative checks for tensor sharing (no numerical oracle shortcuts)."""
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'compiler_optimized'))
sys.path.insert(0,str(REPO/'compiler_optimized/compiler_library'))
from backend.source_cse import optimize_source


def source(*, different_input=False, different_body=False, read_init=False,
           opaque=False, direct_return=False, strict=False, unknown_annotation=False):
    second='%y' if different_input else '%x'
    body0='''%v = math.roundeven %a : f32
      linalg.yield %v : f32'''
    body1=body0 if not different_body else '''%v = math.floor %a : f32
      linalg.yield %v : f32'''
    if read_init:
        body0=body1='''%v = arith.addf %a, %init : f32
      linalg.yield %v : f32'''
    if opaque:
        body0=body1='''%v = func.call @opaque(%a) : (f32) -> f32
      linalg.yield %v : f32'''
    attributes='attributes {strictfp = "yes"}' if strict else ''
    annotation=', unknown_semantics = "retain"' if unknown_annotation else ''
    end=('func.return %r0, %r1 : tensor<4xf32>, tensor<4xf32>' if direct_return else
         '%z = func.call @consume(%r0, %r1) : (tensor<4xf32>, tensor<4xf32>) -> f32\n    func.return %z : f32')
    returns='(tensor<4xf32>, tensor<4xf32>)' if direct_return else 'f32'
    return f'''builtin.module {{
  func.func private @consume(tensor<4xf32>, tensor<4xf32>) -> f32
  func.func private @opaque(f32) -> f32
  func.func @forward(%x: tensor<4xf32>, %y: tensor<4xf32>) -> {returns} {attributes} {{
    %e0 = tensor.empty() : tensor<4xf32>
    %r0 = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
      ins(%x : tensor<4xf32>) outs(%e0 : tensor<4xf32>) attrs = {{prov.site = "first"{annotation}}} {{
      ^bb0(%a: f32, %init: f32):
      {body0}
    }} -> tensor<4xf32>
    %e1 = tensor.empty() : tensor<4xf32>
    %r1 = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
      ins({second} : tensor<4xf32>) outs(%e1 : tensor<4xf32>) attrs = {{prov.site = "second"{annotation}}} {{
      ^bb0(%a: f32, %init: f32):
      {body1}
    }} -> tensor<4xf32>
    {end}
  }}
}}'''


class SharingChecks(unittest.TestCase):
    def test_identical_value_computations_share_and_keep_provenance(self):
        output,report=optimize_source(source())
        self.assertEqual(report['rewrites'].get('linalg.generic'),1)
        self.assertIn('first',output);self.assertIn('second',output)

    def test_different_input(self):self.assertFalse(optimize_source(source(different_input=True))[1]['rewrites'])
    def test_different_scalar_program(self):self.assertFalse(optimize_source(source(different_body=True))[1]['rewrites'])
    def test_live_initializer(self):self.assertFalse(optimize_source(source(read_init=True))[1]['rewrites'])
    def test_opaque_scalar_call(self):self.assertFalse(optimize_source(source(opaque=True))[1]['rewrites'])
    def test_direct_tensor_return_aliases_are_preserved(self):self.assertFalse(optimize_source(source(direct_return=True))[1]['rewrites'])
    def test_strict_fp(self):self.assertFalse(optimize_source(source(strict=True))[1]['rewrites'])
    def test_unknown_semantic_annotation(self):self.assertFalse(optimize_source(source(unknown_annotation=True))[1]['rewrites'])


if __name__=='__main__':unittest.main()
