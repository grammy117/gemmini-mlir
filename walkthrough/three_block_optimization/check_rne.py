"""Check complete typed rounding chains and refusal at semantic boundaries."""
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'compiler_optimized'),str(REPO/'compiler_optimized/compiler_library')]
from backend.host_quant_rne import rewrite

SOURCE='''builtin.module {func.func @forward(%x:f32)->f32 {
 %r=math.roundeven %x : f32
 func.return %r : f32
}}'''


def native(*,width=8,lower='-1.280000e+02',upper='1.270000e+02',extra='',callee='roundevenf',predicate='olt',flags=''):
    return f'''declare float @{callee}(float)
define i{width} @quantize(float %x) {flags} {{
entry:
 %rounded = call float @{callee}(float %x)
 %low = fcmp {predicate} float %rounded, {lower}
 %a = select i1 %low, float {lower}, float %rounded
 %high = fcmp ogt float %a, {upper}
 %b = select i1 %high, float {upper}, float %a
 %result = fptosi float %b to i{width}
 {extra}
 ret i{width} %result
}}'''


class RoundingChecks(unittest.TestCase):
    def test_closed_chain_uses_explicit_rne(self):
        out,r=rewrite(native(),SOURCE)
        self.assertEqual(len(r['routes']),1)
        self.assertIn('fcvt.w.s $0, ft0, rne',out)
        self.assertNotIn('%rounded =',out)

    def test_narrow_width_is_read_from_ir(self):
        out,r=rewrite(native(width=4,lower='-8.000000e+00',upper='7.000000e+00'),SOURCE)
        self.assertEqual(r['routes'][0]['bounds'],[-8,7]);self.assertIn('to i4',out)

    def test_fractional_clamp_refuses(self):self.assertFalse(rewrite(native(lower='-1.275000e+02'),SOURCE)[1]['routes'])
    def test_incomplete_range_refuses(self):self.assertFalse(rewrite(native(upper='6.300000e+01'),SOURCE)[1]['routes'])
    def test_wide_integer_refuses(self):self.assertFalse(rewrite(native(width=32),SOURCE)[1]['routes'])
    def test_unknown_rounding_callee_refuses(self):self.assertFalse(rewrite(native(callee='other'),SOURCE)[1]['routes'])
    def test_unordered_comparison_refuses(self):self.assertFalse(rewrite(native(predicate='ult'),SOURCE)[1]['routes'])
    def test_additional_observer_refuses(self):
        original=native(extra='%observer = fadd float %rounded, 1.000000e+00')
        out,r=rewrite(original,SOURCE);self.assertEqual(out,original);self.assertFalse(r['routes'])
    def test_strict_fp_refuses(self):self.assertFalse(rewrite(native(flags='strictfp'),SOURCE)[1]['routes'])
    def test_strict_source_fp_refuses(self):
        source=SOURCE.replace('->f32 {','->f32 attributes {strictfp = "yes"} {')
        self.assertNotEqual(source,SOURCE)
        self.assertFalse(rewrite(native(),source)[1]['routes'])
    def test_fast_fp_refuses(self):
        self.assertFalse(rewrite(native().replace('fcmp olt','fcmp fast olt'),SOURCE)[1]['routes'])
    def test_source_defined_symbol_refuses(self):
        source=SOURCE.replace('builtin.module {','builtin.module {func.func private @roundevenf(f32)->f32 ')
        self.assertFalse(rewrite(native(),source)[1]['routes'])
    def test_unsupported_host_refuses(self):self.assertFalse(rewrite(native(),SOURCE,host_isa='other')[1]['routes'])
    def test_source_strict_fp_refuses_even_if_native_attribute_was_lost(self):
        source=SOURCE.replace('-> f32 {','-> f32 attributes {strictfp = "yes"} {')
        if source==SOURCE:
            from merlin.frontends.linalg_mlir import parse_mlir_text
            from merlin.xdsl_dialects._common import text
            from xdsl.dialects.builtin import StringAttr
            module=parse_mlir_text(SOURCE)
            fn=next(op for op in module.body.block.ops if op.name=='func.func')
            fn.attributes['strictfp']=StringAttr('yes');source=text(module)
        self.assertFalse(rewrite(native(),source)[1]['routes'])
    def test_no_source_math_operation_refuses(self):
        source='builtin.module {func.func @forward(%x:f32)->f32 {func.return %x : f32}}'
        self.assertFalse(rewrite(native(),source)[1]['routes'])
    def test_nan_select_behavior_must_be_preserved(self):
        original=native().replace('select i1 %lo, float -1.280000e+02, float %rounded','select i1 %lo, float %rounded, float -1.280000e+02')
        if original==native():
            original=native().replace('select i1 %a,','select i1 %a,')
            original=original.replace('float -1.280000e+02, float %rounded','float %rounded, float -1.280000e+02')
        self.assertNotEqual(original,native())
        self.assertFalse(rewrite(original,SOURCE)[1]['routes'])
    def test_arbitrary_runtime_callee_refuses(self):self.assertFalse(rewrite(native().replace('@roundevenf','@custom_round'),SOURCE)[1]['routes'])
    def test_multiple_host_calls_need_source_coverage(self):
        original=native().replace('ret i8 %result','%extra = call float @roundevenf(float %x)\n ret i8 %result')
        out,r=rewrite(original,SOURCE);self.assertEqual(out,original);self.assertIn('host_round_call_count_not_source_bound',r['refusals'])
    def test_temporary_names_do_not_collide(self):
        out,r=rewrite(native(extra='%merlin.quant.rne.0 = add i8 %result, 1'),SOURCE)
        self.assertEqual(len(r['routes']),1);self.assertIn('%merlin.quant.rne.0x = call',out)


if __name__=='__main__':unittest.main()
