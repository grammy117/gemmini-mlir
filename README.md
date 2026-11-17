# Optimized Merlin compiler for Gemmini

This branch develops a general compiler for mixed host/Gemmini execution from the archived Merlin-generated backend. It adds graph-wide tensor sharing, bounded scalar quantization lowering, allocation-aware host optimization, static strided-copy lowering, and measured helper inlining. The experiments compile complete quantized Llama-style decoder blocks from PyTorch to bare-metal RISC-V executables and validate them on an existing Gemmini RTL simulator.

The optimized compiler is in [`compiler_optimized/`](compiler_optimized/). The original [`compiler/`](compiler/) is preserved unchanged.

Read the [optimization report](OPTIMIZATION_REPORT.md) for the full PyTorch-to-ELF walkthrough, stage ownership, group boundaries, transformation predicates, profiling, ablations, rejected candidates and reproduction instructions.

| Workload | Prior optimized checkpoint | Final default | Fewer cycles |
|---|---:|---:|---:|
| One complete decoder block | 537,032 | **357,674** | **33.4%** |
| Three sequential decoder blocks | 1,639,267 | **1,060,118** | **35.3%** |

All final outputs are bit-exact to the supplied quantized PyTorch references. These are complete-entry RTL cycle measurements, including host and device work. The default was selected from a grouping/inlining sweep on the three-block chain. An alternative inlining policy reaches 356,696 cycles for one block; the default is faster on three.

A block has seven logical projections: Q, K, V, O, gate, up and down. The default clusters them into **four physical device groups**: Q/K/V, O, gate/up, and down. Three sequential blocks have **12 device calls**, four reused kernel bodies, and 21 distinct projection weights. Normalization, RoPE, FP32 attention, softmax, SiLU and residuals run on the scalar host. The compiler shares equivalent preparation across those boundaries while preserving the dependencies between blocks.

The reference uses explicit W8A8 linears, FP32 host arithmetic and deterministic demonstration weights. Reported output equality is against that quantized PyTorch model. The measurements use the same unchanged GemminiRocketConfig Verilator binary with magic memory; they are not FPGA measurements.

## Build and inspect

The checked-in captures and scripts use the existing model2MLIR, Merlin, LLVM, RISC-V runtime and simulator installations on a26. From this checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /scratch/grammy/blast/model2MLIR/.venv/bin/python \
  walkthrough/three_block_optimization/reproduce.py \
  --name three_repeat --blocks 3
```

This builds the complete executable, checks device re-emission, compares all final outputs in Spike, and exports intermediate stages. Choose a fresh name. Add `--rtl` to run the bounded simulator, or follow the report's command to pin it to a chosen CPU. The original standalone `opt` binary is absent locally; these scripts compile the emitted combined LLVM with Clang. The report describes that adaptation explicitly.

Defaults and ablation switches are in [`optimization.json`](compiler_optimized/optimization.json) and [`optimization.py`](compiler_optimized/backend/optimization.py). Compact captures, results, logs and reproduction scripts are versioned; large generated build trees stay local. [`final_report.json`](walkthrough/three_block_optimization/final_report.json) binds the final measurements and binaries.

## Archived paper compiler

The starting point is upstream branch `paper/refinement/exo-kernels-4plus4h`, archived experiment `CCA-EXO-R3`, at commit `7cd030e13360fd0961bd83fa288bd9009ba813df`. Its source digest is `4994f7b478047decf473e8263f6246b493f5189f0fbfef70b921ba8f99b012f7`, covering 1,251 original compiler files. Verify it with:

```bash
python3 experiment/verify_snapshot.py
```

The [manifest](experiment/manifest.json), [experimental setup](experiment/setup.json), [recorded metrics](experiment/metrics.json), and [measurement records](experiment/measurement-records.csv) describe the historical paper experiment. Those measurements remain separate from the new transformer runs under `walkthrough/`. The original archive README is available at the [starting commit](https://github.com/ucb-bar/gemmini-mlir/blob/7cd030e13360fd0961bd83fa288bd9009ba813df/README.md).
