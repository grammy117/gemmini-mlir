#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/scratch/grammy/blast/gemmini-mlir/walkthrough/attention_block
cd /scratch/grammy/blast/gemmini-mlir/walkthrough/attention_block
task_python=/scratch/grammy/blast/model2MLIR/.venv/bin/python
"$task_python" capture.py
"$task_python" build.py
"$task_python" run.py
"$task_python" export_stages.py
