"""Check the scope of executable helper visibility and inline contracts."""
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'compiler_optimized'))
from backend.native_helpers import rewrite

SOURCE='''declare float @expf(float)
define void @helper() {
  ret void
}
define void @kernel_entry() {
  call void @helper()
  ret void
}
'''


class Checks(unittest.TestCase):
    def test_helper_only(self):
        out,r=rewrite(SOURCE)
        self.assertIn('define internal void @helper()  alwaysinline {',out)
        self.assertIn('define void @kernel_entry() {',out)
        self.assertIn('declare float @expf(float)',out)
    def test_none_is_byte_preserving(self):self.assertEqual(rewrite(SOURCE,policy='none')[0],SOURCE)
    def test_internal_only(self):self.assertNotIn('alwaysinline',rewrite(SOURCE,policy='internalize')[0])
    def test_noinline_contract_retained(self):
        out,r=rewrite(SOURCE.replace('@helper() {','@helper() #0 {')+'attributes #0 = { noinline }\n')
        self.assertNotIn('alwaysinline',out);self.assertEqual(r['helpers'][0]['inline_refusal'],'noinline')
    def test_unusual_linkage_refuses(self):
        out,r=rewrite(SOURCE.replace('define void @helper','define weak void @helper'))
        self.assertIn('define weak void @helper',out);self.assertIn('refused',r['helpers'][0])
    def test_existing_internal_is_not_duplicated(self):
        out,r=rewrite(SOURCE.replace('define void @helper','define internal void @helper'))
        self.assertNotIn('internal internal',out)
    def test_missing_entry_refuses(self):
        with self.assertRaises(ValueError):rewrite(SOURCE.replace('kernel_entry','another_entry'))


if __name__=='__main__':unittest.main()
