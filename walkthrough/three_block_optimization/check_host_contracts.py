"""Check the source custody and conservative scope of LLVM host facts."""
import sys
import unittest
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'compiler_optimized'),str(REPO/'compiler_optimized/compiler_library')]
from backend.host_contracts import rewrite

SOURCE='''builtin.module {func.func @forward(%x:f32)->f32 {
 %r=math.rsqrt %x : f32
 func.return %r : f32
}}'''
LLVM='''declare ptr @malloc(i64)
declare ptr @aligned_alloc(i64,i64)
declare float @rsqrtf(float) #0
attributes #0 = { memory(none) }
'''


class ContractChecks(unittest.TestCase):
    def test_owned_allocations_and_pure_math(self):
        out,r=rewrite(LLVM,SOURCE)
        self.assertEqual(r['allocators'],['malloc','aligned_alloc'])
        self.assertEqual(r['math'][0]['added'],['nounwind','willreturn'])
        self.assertIn('declare noalias ptr @aligned_alloc',out)

    def test_user_allocator_symbol_preserved(self):
        source=SOURCE.replace('builtin.module {','builtin.module {func.func private @malloc(i64)->i64 ')
        out,r=rewrite(LLVM,source)
        self.assertIn('declare ptr @malloc',out);self.assertNotIn('malloc',r['allocators'])

    def test_user_math_symbol_preserved(self):
        source=SOURCE.replace('builtin.module {','builtin.module {func.func private @rsqrtf(f32)->f32 ')
        out,r=rewrite(LLVM,source);self.assertFalse(r['math'])

    def test_existing_purity_is_required(self):self.assertFalse(rewrite(LLVM.replace('memory(none)',''),SOURCE)[1]['math'])
    def test_no_source_math_means_no_math_facts(self):
        self.assertFalse(rewrite(LLVM,SOURCE.replace('%r=math.rsqrt %x : f32','%r=arith.addf %x,%x : f32'))[1]['math'])
    def test_strict_source_refuses(self):
        out,r=rewrite(LLVM,SOURCE.replace('->f32 {','->f32 attributes {strictfp="yes"} {'))
        self.assertEqual(out,LLVM);self.assertIn('refusal',r)
    def test_idempotent(self):
        first,_=rewrite(LLVM,SOURCE);second,_=rewrite(first,SOURCE);self.assertEqual(first,second)


if __name__=='__main__':unittest.main()
