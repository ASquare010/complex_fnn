"""Analytic two-core BTT shape accounting; no PyTorch, GPU, or external execution."""

import json
import math
from pathlib import Path


def factors(value):
    parts = [1, 1]
    divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            i = 0 if parts[0] <= parts[1] else 1
            parts[i] *= divisor
            value //= divisor
        divisor += 1
    if value > 1:
        i = 0 if parts[0] <= parts[1] else 1
        parts[i] *= value
    return sorted(parts)


rows = []
for d_in, d_out in ((512, 608), (608, 512), (512, 2048)):
    m0, m1 = factors(d_in)
    n0, n1 = factors(d_out)
    for rank in (1, 2):
        shapes = [(m1, m0, rank * n0), (n0, rank * m1, n1)]
        count = sum(math.prod(shape) for shape in shapes)
        assert count == rank * m1 * n0 * (m0 + n1)
        intermediate = rank * m1 * n0
        middle = min(d_in, d_out)
        rows.append(
            dict(
                d_in=d_in,
                d_out=d_out,
                rank=rank,
                input_factors=[m0, m1],
                output_factors=[n0, n1],
                core_shapes=shapes,
                core_parameters=count,
                intermediate_width=intermediate,
                rank_upper_bound=min(d_in, d_out, intermediate),
                fp32_intermediate_mib=8192 * intermediate * 4 / 2**20,
                blockshuffle_g8_parameters=middle * (d_in + d_out) // 8,
                blockshuffle_intermediate_width=middle,
                minimum_rank_to_avoid_this_bottleneck=math.ceil(middle / (m1 * n0)),
            )
        )
assert factors(608) == [8, 76]


def closest_pair(value):
    for factor in range(math.isqrt(value), 0, -1):
        if value % factor == 0:
            return [factor, value // factor]
    raise AssertionError(value)


balanced = []
for d_in, d_out in ((512, 608), (608, 512)):
    m0, m1 = closest_pair(d_in)
    n0, n1 = closest_pair(d_out)
    rank = 1
    balanced.append(
        dict(
            d_in=d_in,
            d_out=d_out,
            input_factors=[m0, m1],
            output_factors=[n0, n1],
            core_parameters=rank * m1 * n0 * (m0 + n1),
            intermediate_width=rank * m1 * n0,
            rank_upper_bound=min(d_in, d_out, rank * m1 * n0),
            fp32_intermediate_mib=8192 * rank * m1 * n0 * 4 / 2**20,
        )
    )
assert closest_pair(608) == [19, 32]
output = dict(
    balanced_factor_ablation=balanced,
    rows=rows,
    normalization_scalars_excluded=2,
    bias_excluded=True,
    gpu_work=False,
    optimizer_updates=0,
    external_code_executed=False,
)
Path("research/references/btt_official/shape_accounting.json").write_text(
    json.dumps(output, indent=2) + "\n"
)
print(json.dumps(balanced))
