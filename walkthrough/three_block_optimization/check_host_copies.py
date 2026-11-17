"""Check disjoint-root and static-domain requirements for copy loop lowering."""
from pathlib import Path
import sys
import unittest
import subprocess

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'compiler_optimized'),str(REPO/'compiler_optimized/compiler_library')]
from backend.host_copies import rewrite


def source(dtype='f32',alias=False,borrowed=False,contiguous=False):
    signature=f'%arg:memref<4x8x{dtype}>' if borrowed else ''
    allocation=(f'%a = memref.alloc() : memref<4x8x{dtype}>' if not alias else
        f'''%base = memref.alloc() : memref<4x16x{dtype}>
        %a = memref.subview %base[0,0][4,8][1,1] : memref<4x16x{dtype}> to memref<4x8x{dtype},strided<[16,1]>>''')
    if borrowed:allocation=f'%a = memref.cast %arg : memref<4x8x{dtype}> to memref<4x8x{dtype}>'
    if contiguous:destination=f'%d = memref.alloc() : memref<4x8x{dtype}>'
    else:
        base='%base' if alias else '%b'
        destination=('' if alias else f'%b = memref.alloc() : memref<4x16x{dtype}>')+f'''
        %d = memref.subview {base}[0,0][4,8][1,1] : memref<4x16x{dtype}> to memref<4x8x{dtype},strided<[16,1]>>'''
    st=f'memref<4x8x{dtype},strided<[16,1]>>' if alias else f'memref<4x8x{dtype}>'
    dt=f'memref<4x8x{dtype}>' if contiguous else f'memref<4x8x{dtype},strided<[16,1]>>'
    text=f'builtin.module {{func.func @copy({signature}) {{ {allocation}\n{destination}\nmemref.copy %a,%d : {st} to {dt}\nfunc.return }} }}'
    # The compiler bridge consumes stock MLIR's generic serialization.
    done=subprocess.run(['/scratch/grammy/blast/merlin/third_party/llvm-build/bin/mlir-opt',
        '--mlir-print-op-generic'],input=text,capture_output=True,text=True,timeout=10)
    if done.returncode:raise ValueError(done.stderr)
    return done.stdout


class CopyChecks(unittest.TestCase):
    def test_static_disjoint_strided_copy(self):
        output,r=rewrite(source());self.assertEqual(len(r['copies']),1)
        self.assertNotIn('"memref.copy"',output);self.assertIn('"linalg.generic"',output)
    def test_integer_copy(self):self.assertEqual(len(rewrite(source(dtype='i8'))[1]['copies']),1)
    def test_overlapping_allocation_refuses(self):self.assertFalse(rewrite(source(alias=True))[1]['copies'])
    def test_unknown_borrowed_origin_refuses(self):self.assertFalse(rewrite(source(borrowed=True))[1]['copies'])
    def test_contiguous_copy_keeps_memcpy_route(self):self.assertFalse(rewrite(source(contiguous=True))[1]['copies'])
    def test_subbyte_storage_refuses(self):self.assertFalse(rewrite(source(dtype='i1'))[1]['copies'])
    def test_noncanonical_integer_storage_refuses(self):self.assertFalse(rewrite(source(dtype='i24'))[1]['copies'])
    def test_unknown_annotation_refuses(self):
        from merlin.frontends.linalg_mlir import parse_mlir_text
        from merlin.xdsl_dialects._common import text
        from xdsl.dialects.builtin import StringAttr
        m=parse_mlir_text(source());op=next(o for o in m.walk() if o.name=='memref.copy')
        op.attributes['unknown']=StringAttr('keep')
        self.assertFalse(rewrite(text(m,generic=True))[1]['copies'])


if __name__=='__main__':unittest.main()
