"""Explicit scalar upstream lowering through stock tools, without Python bindings.

This transport runs the ordinary scalar pass pipeline. Runner-only rewrites,
proof callbacks and guarded fusion require the Python transport and are refused.
Actual lowering/translation invocations retain their inputs and outputs; these
records establish custody, not numerical or deployment qualification.
"""

from __future__ import annotations

import math
import os
import time
from io import StringIO
from pathlib import Path

from merlin.common import compile_trace, invocation_record
from merlin.common.cpu_pool import shared_compute_pool
from merlin.runtime.build_runtime import run_in_build_view

from . import fusion_guard
from .lowering_recipe import LoweringRecipe
from .toolchain import llvm_install, mlir_translate


def _identifiers(text):
    """Conservatively include names inside nested pipelines and option strings."""
    names, token = set(), []
    for char in text + ",":
        if char.isalnum() or char in "_.-":
            token.append(char)
        elif token:
            names.add("".join(token))
            token.clear()
    return names


def lower(
    text, *, work, pipeline, features, timeout, audit, data_layout, index_bits,
    lowering_selection, unsupported,
):
    """Run a selected scalar pipeline with a shared total deadline and byte custody."""
    from . import pipeline as pipeline_owner
    from .pipeline import SINK_DEALLOC_PASS, PipelineError, _bind_index_width, _fix_float_literals, _upstream_pipeline

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("stock-cli requires a finite positive total timeout")
    deadline = time.monotonic() + timeout

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("stock-cli upstream lowering deadline expired")
        return seconds

    if features or any(unsupported.values()):
        requested = sorted(set(features) | {key for key, value in unsupported.items() if value})
        raise PipelineError("stock-cli does not implement Python-runner options: " + ", ".join(requested))
    if compile_trace.active() is not None or (audit is not None and audit.directory is not None):
        raise PipelineError("stock-cli does not implement native per-pass inspection")
    pipeline = _upstream_pipeline(frozenset()) if pipeline is None else pipeline
    names = _identifiers(pipeline)
    if names.intersection(compile_trace.split_pipeline(fusion_guard.FUSE_PASS)) and fusion_guard.enabled():
        raise PipelineError("stock-cli requires explicit fusion-guard deselection for its unguarded fusion pass")
    if names.intersection(compile_trace.split_pipeline(SINK_DEALLOC_PASS)):
        raise PipelineError("stock-cli does not implement the Python deallocation placement check")
    if index_bits is not None:
        if not data_layout:
            raise ValueError("explicit index width requires selected compiler data layout")
        pipeline = _bind_index_width(pipeline, index_bits)
    if os.environ.get('MERLIN_STOCK_ALIGNED_ALLOC','0') == '1':
        # Standard memref lowering can retain allocation provenance by calling
        # the C11 aligned allocator instead of integer-rounding malloc pointers.
        import re
        def aligned(match):
            options=match.group(1) or ''
            if 'use-aligned-alloc' in options:
                raise ValueError('aligned allocation selection conflicts with explicit pipeline options')
            return 'finalize-memref-to-llvm{'+options+' use-aligned-alloc=true}'
        pipeline,count=re.subn(r'finalize-memref-to-llvm(?:\{([^}]*)\})?',aligned,pipeline)
        if count!=1:raise ValueError('aligned allocation needs one final memref lowering pass')
    if data_layout is not None:
        from xdsl.dialects.builtin import StringAttr
        from xdsl.printer import Printer

        from merlin.frontends.linalg_mlir import parse_mlir_text

        module = parse_mlir_text(text)
        module.attributes["llvm.data_layout"] = StringAttr(data_layout)
        module.verify()
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        text = stream.getvalue() + "\n"
    remaining()
    source, lowered, translated = [work / name for name in ("model.mlir", "model.llvm.mlir", "model.ll")]
    source.write_text(text, encoding="utf-8")
    # Previous products cannot satisfy a failed new invocation in a reused workdir.
    for product in (lowered, translated):
        product.unlink(missing_ok=True)
    recipe = LoweringRecipe(
        work, features=features, sources={
            "prepared_mlir": source, "pipeline_driver": Path(pipeline_owner.__file__),
            "translation_transport": Path(__file__),
        },
    )
    def invoke(command, stage, input_path, output_path):
        command = list(map(str, command))
        recipe.command(command)
        with shared_compute_pool(timeout_s=remaining()):
            kwargs = dict(
                directory=work.resolve(), stage=stage, inputs=(input_path,), outputs=(output_path,),
                capture_output=True, text=True, timeout=remaining(), env=dict(os.environ),
            )
            result = run_in_build_view(command, **kwargs)
            if result is None:
                result = invocation_record.run(
                    command, dependencies=(Path(__file__), Path(pipeline_owner.__file__)), **kwargs,
                )
        remaining()
        if result.returncode or not output_path.is_file():
            raise PipelineError(f"stock-cli {stage} failed:\n{result.stdout}\n{result.stderr}")

    if os.environ.get('MERLIN_STOCK_STATIC_COPIES','0') == '1':
        import json
        from backend import host_copies
        from backend.host_copies import rewrite
        recipe.bind_source('static_host_copy_implementation',Path(host_copies.__file__))
        marker='func.func(convert-linalg-to-loops)'
        if pipeline.count(marker)!=1:raise ValueError('static copy lowering needs one scalar loop boundary')
        prefix,suffix=pipeline.split(marker)
        buffered=work/'model.bufferized.mlir'
        invoke([llvm_install()/'bin/mlir-opt',source,
            '--pass-pipeline=builtin.module('+prefix.rstrip(',')+')','--mlir-print-op-generic','-o',buffered],
            'stock_bufferization',source,buffered)
        rewritten,receipt=rewrite(buffered.read_text())
        selected=work/'model.static-copies.mlir';selected.write_text(rewritten)
        (work/'static-host-copies.json').write_text(json.dumps(receipt,indent=2))
        recipe.bind_source('static_host_copy_input',buffered)
        recipe.bind_source('static_host_copy_selection',selected)
        recipe.bind_source('static_host_copy_receipt',work/'static-host-copies.json')
        invoke([llvm_install()/'bin/mlir-opt',selected,
            '--pass-pipeline=builtin.module('+marker+suffix+')','-o',lowered],
            'stock_static_copy_lowering',selected,lowered)
    else:
        invoke(
            [llvm_install() / "bin" / "mlir-opt", source,
             "--pass-pipeline=builtin.module(" + pipeline + ")", "-o", lowered],
            "stock_upstream_lowering", source, lowered,
        )
    invoke(
        [mlir_translate(), "--mlir-to-llvmir", lowered, "-o", translated],
        "stock_llvm_translation", lowered, translated,
    )
    result = _fix_float_literals(translated.read_text(encoding="utf-8"))
    if lowering_selection is not None:
        lowering_selection.update(driver="stock-cli", effective_pipeline=pipeline)
        if index_bits is not None:
            lowering_selection.update(index_bits=index_bits, data_layout=data_layout)
    remaining()
    recipe.returned(result)
    try:
        remaining()
    except TimeoutError:
        recipe.path.unlink(missing_ok=True)
        raise
    return result
