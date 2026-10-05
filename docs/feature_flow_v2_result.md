# Feature flow execution revision 2: retired on resources

[Registered execution](feature_flow_v2_hypothesis.md), with the unchanged
[FFN hypothesis](feature_flow_hypothesis.md). Independent CPU and CUDA checks
passed before profiling. All 48 whole-Transformer profiles completed, with
finite updates and reserved memory below 6 GiB. The primary fails every dense
resource comparison on both corpora. Retire this execution recipe before
language training; language quality remains unknown.

| Corpus | Dense reference | Allocated memory reduction | Throughput ratio |
| --- | --- | ---: | ---: |
| TinyStories | Full GELU | 25.54% | 0.897 |
| TinyStories | Full native SwiGLU | 28.36% | 0.900 |
| TinyStories | Full kernel SwiGLU | 25.68% | 0.898 |
| WikiText | Full GELU | 25.54% | 0.825 |
| WikiText | Full native SwiGLU | 28.36% | 0.870 |
| WikiText | Full kernel SwiGLU | 25.68% | 0.859 |

Primary allocated peak is 401.7 MiB. Median time per 100 updates is 4.83 seconds
on TinyStories and 4.79 seconds on WikiText. Relative to the separately measured
retired version 1, throughput increased 30.60%/33.35%; these are separate screens,
not paired simultaneous timings. Against dense references the update-time
increase remains about 11–21%, exceeding the registered 5% limit. Three
alternating-order rounds, 20 warmup and 100 measured updates per profile were
used; these short measurements are not sustained confirmation.

CPU FP64 independent equations, all adjoints and finite differences, counts,
calibration and five complete-model causality, locality, save/load and exact
optimizer recovery checks passed. CUDA FP32/BF16 checks passed for group sizes
1/8/32, explicit size-37 fallback, incomplete tiles and the actual 2,048-token
training shape. Nonzero mixing matrices, all removals and full-model adjoints
were compared against ordinary autograd. Errors and registered tolerances are
retained in `records/feature-flow-v2-checks.json`.

Revision 2 fuses each grouped state update and its input derivative. Matrix
gradients use deterministic 32-token tiles accumulated across reversed steps.
Full input/output projections remain unchanged; forward/transposed matrix
copies, partial buffers, reconstructed states and Python dispatch are included
in measurements. Final BF16 casts use Torch and TF32 remains disabled.

Evidence is under `dump/feature-flow-v2/`, including the completed `result.md`,
frozen sources and protocol, and `records/feature-flow-v2-profiles.json`.
The earlier version's failed screen remains preserved. No language campaign
starts for this recipe. The next [execution hypothesis](feature_flow_v3_hypothesis.md)
keeps the mathematics fixed and fuses the entire internal trajectory/adjoint;
it is registered but not implemented or measured.
