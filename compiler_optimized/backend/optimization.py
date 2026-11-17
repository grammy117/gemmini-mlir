"""Explicit, source-independent optimization settings and ablation controls."""
import json
import os
from pathlib import Path


def settings():
    values = json.loads((Path(__file__).resolve().parents[1]/'optimization.json').read_text())
    for key, env in [('pure_tensor_cse','GEMMINI_OPT_SOURCE_CSE'),
                     ('cluster_shared_products','GEMMINI_OPT_GROUPS'),
                     ('wide_shared_rhs','GEMMINI_OPT_WIDE_SHARED'),
                     ('literal_shared_epochs','GEMMINI_OPT_LITERAL_SHARED'),
                     ('shared_readout_epochs','GEMMINI_OPT_READOUT_SHARED'),
                     ('scalar_quant_rne','GEMMINI_OPT_RNE'),
                     ('aligned_host_allocations','GEMMINI_OPT_ALIGNED_ALLOC'),
                     ('host_optimization_contracts','GEMMINI_OPT_CONTRACTS'),
                     ('generalize_before_fusion','GEMMINI_OPT_FUSION')]:
        if env in os.environ:
            if os.environ[env] not in ('0','1'):
                raise ValueError(env+' must be 0 or 1')
            values[key] = os.environ[env] == '1'
        if type(values[key]) is not bool:
            raise ValueError(key+' must be Boolean')
    if values['native_optimization_level'] not in (1,2,3):
        raise ValueError('native_optimization_level must be 1, 2 or 3')
    return values
