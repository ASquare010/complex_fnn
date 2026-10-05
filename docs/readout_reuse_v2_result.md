# Readout reuse execution v2: numerical failure

Retire v2 before resource profiling or language training. The
[registered revision](readout_reuse_v2_hypothesis.md) preserves the original FFN
equations, but its fused forward failed the original full-model BF16 tolerance.
Neither speed, VRAM nor language quality was measured for v2.

CPU equations/all gradients, finite differences, counts/quadrature,
initialization and exact recovery checks passed. CUDA completed 192 independent
FFN fixtures and 18 whole-model comparisons before failing the actual full
BF16 grouped model. The registered comparison rejects 30,346 of 8,388,608 logits;
maximum disallowed absolute difference .01171875 exceeds atol .004/rtol .02.
Maximum difference over all logits is .015625; some larger differences still
meet relative tolerance. The unchanged repeat reproduced the rejection exactly
and preserved its model/tokens. Post-model raw fixtures and cast-edge checks
were not reached; no complete CUDA pass record exists.

The preserved-fixture diagnosis compares operations at identical normalized
inputs in each layer. Raw scalar responses equal native scalar responses
exactly for all 2,359,296 entries per layer. Native grouped BMM on those raw
responses also matches native BF16 mixed values and FFN outputs exactly.
Fused grouped accumulation instead changes 57/55/51/42 BF16 mixed values in
layers 0/1/2/3; largest differences .001953125/.0009765625/.0009765625/.0009765625.
Same-input FFN differences reach .00048828125. These local changes propagate
through the full Transformer. This isolates the forward accumulation difference
on this fixture; it does not identify all backward errors or a performance
bottleneck. The BF16 tolerance is not relaxed.

2,506,752 FFN weights (70.35% reduction) and mathematical forward work
9,732,096 FLOPs/token are unchanged accounting, not empirical efficiency or
information-capacity gains. Revision 3 requires a separate registration,
unchanged whole-model precision checks and the complete resource screen.

Evidence: records/readout-reuse-v2-cpu-checks.json,
records/readout-reuse-v2-numerical-failure.json,
records/readout-reuse-v2-numerical-diagnostic.json and
records/readout-reuse-v2-development-errors.json. Exact failing fixture:
dump/readout-reuse-v2-failing-model.pt. Old sources/records remain preserved.
