"""Build the captured complete self-attention block into a bare-metal RISC-V ELF.

The frozen CLI emits device kernels, host LLVM, descriptor glue, and combined
LLVM. This machine lacks its final `opt` executable, so we compile that exact
combined LLVM with Clang -O1 instead of the CLI's opt/import-LLVM round trip.
No frozen compiler source or scheduling checks are changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MERLIN = Path('/scratch/grammy/blast/merlin')
LLVM = MERLIN / 'third_party/llvm-build/bin'
GCC = Path('/scratch/grammy/blast/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')
PYTHON = MERLIN / '.venv/bin/python'
RUNTIME = MERLIN / 'merlin/runtime'
BUILD = ROOT / 'build'
COMMANDS = []


def run(argv, *, env=None, permit_missing_opt=False):
    argv = list(map(str, argv))
    start = time.monotonic()
    completed = subprocess.run(argv, cwd=REPO, env=env, capture_output=True,
                               text=True, timeout=180)
    row = dict(argv=argv, returncode=completed.returncode,
               seconds=time.monotonic()-start,
               stdout=completed.stdout, stderr=completed.stderr)
    COMMANDS.append(row)
    (BUILD / 'commands.json').write_text(json.dumps(COMMANDS, indent=2))
    if completed.returncode:
        if permit_missing_opt and f"No such file or directory: '{LLVM / 'opt'}'" in completed.stderr:
            return completed
        raise RuntimeError(f'command failed: {argv}\n{completed.stderr[-8000:]}')
    return completed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    BUILD.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(ROOT),
               PYTHONPATH=f'{REPO}/compiler/compiler_library:{REPO}/dependencies/generic-library',
               MERLIN_REPO_ROOT=str(MERLIN), MERLIN_WORK_DIR=str(ROOT),
               MERLIN_OUT_ROOT=str(ROOT/'compiler_out'),
               MERLIN_MLIR_INSTALL=str(LLVM.parent), MERLIN_COMPILER_PYTHON=str(PYTHON),
               MERLIN_RUNTIME_DIR=str(RUNTIME))
    original = run([PYTHON, REPO/'compiler/compiler.py', '--convert-iface-to-gemmini',
                    '--emit-target-artifact', '--debug', ROOT/'capture/model.mlir'],
                   env=env, permit_missing_opt=True)
    (BUILD/'compiler.stderr.log').write_text(original.stderr)
    candidates = list((ROOT/'fresh10-authored-compiler').glob('*/mixed/combined/combined.ll'))
    if len(candidates) != 1:
        raise RuntimeError('expected one source-bound combined LLVM product')
    combined = candidates[0]
    mixed = combined.parent.parent
    shutil.copyfile(combined, BUILD/'combined.ll')
    shutil.copyfile(mixed/'placement.json', BUILD/'placement.json')
    shutil.copyfile(mixed/'host/lowering_recipe.json', BUILD/'lowering_recipe.json')
    # Save the actual checked device schedules and verify reproducible emission.
    run([PYTHON, ROOT/'inspect_schedules.py', mixed, BUILD/'schedules.json'], env=env)

    weights = load_file(ROOT/'capture/weights.safetensors')
    manifest = json.loads((ROOT/'capture/weights.safetensors.manifest.json').read_text())
    inputs = np.load(ROOT/'capture/inputs.npz', allow_pickle=False)
    buffers = []
    for i in range(len(manifest)):
        item = manifest[str(i)]
        buffers.append(weights[item['weight']] if 'weight' in item else inputs[item['name']])
    lines = ['#include <stdint.h>', '#include "htif.h"',
             'void merlin_arena_reset(void);',
             'extern void kernel_entry(' + ','.join(['void *']*(len(buffers)+1)) + ');']
    for i, value in enumerate(buffers):
        data = np.ascontiguousarray(value).tobytes()
        lines.append(f'static unsigned char tensor_{i}[{len(data)}] __attribute__((aligned(64))) = {{'+','.join(map(str,data))+'};')
    lines += ['static float output[256] __attribute__((aligned(64)));',
              'int main(long hart) {', '  if (hart) for (;;) __asm__ volatile("wfi");',
              '  console_init(); merlin_arena_reset();',
              '  htif_puts("TRANSFORMER_BEGIN\\n");',
              '  kernel_entry('+','.join([f'tensor_{i}' for i in range(len(buffers))]+['output'])+');',
              '  for (int i=0;i<256;i++) { union {float f; uint32_t u;} v={.f=output[i]};',
              '    htif_puts("OUT "); htif_putd(i); htif_puts(" "); htif_puthex(v.u); htif_putc(\'\\n\'); }',
              '  htif_puts("TRANSFORMER_END\\n"); htif_exit(0);', '  return 0;', '}']
    (BUILD/'main.c').write_text('\n'.join(lines)+'\n')
    sysroot = Path(run([GCC, '-print-sysroot']).stdout.strip()).resolve()
    flags = ['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d',
             '-mcmodel=medany','-mno-relax','-O1','-ffp-contract=off',
             '-fno-vectorize','-fno-slp-vectorize','-ffunction-sections','-fdata-sections']
    run([LLVM/'clang-23',*flags,'-c',BUILD/'combined.ll','-o',BUILD/'model.o'])
    run([LLVM/'clang-23',*flags,'-S',BUILD/'combined.ll','-o',BUILD/'model.s'])
    rt = RUNTIME/'baremetal/spike'
    sources = [BUILD/'main.c',rt/'crt.S',rt/'htif.c',rt/'libc_min.c',
               rt/'merlin_malloc.c',RUNTIME/'abi/mlir_runtime.c']
    objects = [BUILD/'model.o']
    for source in sources:
        obj = BUILD/(source.stem+'.o')
        run([LLVM/'clang-23',*flags,'-ffreestanding','-fno-builtin',
             f'--sysroot={sysroot}','-isystem',sysroot/'include','-I',rt,
             '-DMERLIN_ARENA_BASE_ADDR=0x88000000ULL',
             '-DMERLIN_ARENA_SIZE_BYTES=0x400000ULL','-c',source,'-o',obj])
        objects.append(obj)
    elf = BUILD/'attention_block.elf'
    run([GCC,'-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-mno-relax',
         '-nostdlib','-nostartfiles','-static','-Wl,--gc-sections',
         f'-Wl,-Map={BUILD / "attention_block.map"}','-T',rt/'link.ld',
         *objects,'-lm','-lgcc','-o',elf])
    disassembly = run([LLVM/'llvm-objdump','-d',elf]).stdout
    (BUILD/'attention_block.disassembly.txt').write_text(disassembly)
    (BUILD/'elf_headers.txt').write_text(run([LLVM/'llvm-readelf','-h','-l','-s',elf]).stdout)
    (BUILD/'undefined_symbols.txt').write_text(run([GCC.with_name('riscv64-unknown-elf-nm'),'-u',elf]).stdout)
    placement = json.loads((BUILD/'placement.json').read_text())
    report = {'elf':str(elf),'elf_sha256':sha(elf),'elf_bytes':elf.stat().st_size,
              'source_mlir_sha256':sha(ROOT/'capture/model.mlir'),
              'compiler_commit':run(['git','rev-parse','HEAD']).stdout.strip(),
              'original_cli_completed':original.returncode==0,
              'local_adaptation':'compile emitted combined.ll using Clang -O1; missing opt prevents original final optimizer/import round trip',
              'device_calls':len(placement['device_regions']),
              'distinct_device_kernels':len({r['kernel_symbol'] for r in placement['device_regions']}),
              'numerical_execution':'not yet run',
              'compiled_source_files':[str(p) for p in sources]}
    (BUILD/'build_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
