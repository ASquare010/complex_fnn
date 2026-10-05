# Feature flow execution revision 3: keep the whole trajectory inside a warp

Registered after version 2's completed failed resource screen, before any
implementation or measurement of this revision. Preserve the
[original FFN equations](feature_flow_hypothesis.md), parameter counts,
initialization, three steps, controls and acceptance thresholds. Version 2
improved throughput versus version 1 but retained 11–21% longer updates than
dense. Neither version has a language result.

Hypothesis: fusing an individual step leaves significant repeated launch,
state-buffer and Python dispatch cost. Keep each group's three-step trajectory
in warp registers and fuse its discrete reverse pass. Version 2 did not isolate
those costs, so their dominance is a hypothesis, not a measured fact.

Forward: one warp per token/group. Keep the scalar state in each channel lane,
broadcast each bounded reaction and apply the small group matrix for all three
steps inside one kernel, then write the final unbounded reaction. Preserve
FP32 scalar/dot operations, transpose M once for coalesced forward reads, and
cast output to BF16 only with Torch. All lanes participate in shuffles.

Backward: one block per group/32-token tile, eight token warps, each handling
four tokens. Reconstruct each token's three states in lane registers using the
same operation order as forward. Start with the final reaction derivative;
walk the trajectory backwards, accumulating every output/input M coefficient
and propagating the identity plus M-transpose reaction derivative. Write the
input adjoint once per token. Reduce the eight token warps deterministically
through shared memory, then reduce tile partials with Torch. No atomics, saved
token-sized derivative bank or separate per-step matrix/input-gradient launches.
Master M and raw buffers stay FP32. No TF32, fast math or FMA fusion. Native
Torch handles BF16 adjoint casts. Include transpose/casts/partials and every
remaining dispatch in measurements. The reconstruction projection count
remains unchanged; register pressure and scalar work can still make this slower.

Handle zero/one/three steps and every registered intervention. Cached execution
remains ordinary autograd. Diagonal controls retain their actual small parameter
count while using the same physical group work. Groups above 32 retain the
explicit ordinary grouped fallback. Validate shapes, layouts, dtypes, device,
step counts and launch/index limits. Incomplete tokens and channels must retain
full barrier/shuffle participation.

Use separate version-3 staged files; preserve both earlier executions. Repeat
CPU FP64 independent equations, all adjoints/finite differences, calibration,
counts and all five complete-model causality/locality/save-load/exact AdamW
recovery checks. Repeat the version-2 CUDA suite, including group sizes 1/8/32,
size-37 fallback, incomplete tiles, nonzero M, removals, complete-model adjoints,
native cast edgecases and actual 2048-token gradients. Retain version-2
registered tolerances: FP32 output/input atol 1e-6, rtol 2e-4; FP32 M adjoint
atol 1e-5, rtol 3e-4 even with BF16 inputs; BF16 output/input atol .00025,
rtol .025; full-model BF16 output .004/.02 and parameter adjoints .0001/.05.
Record maximum errors; preserve any development failure and its repair.

After all checks pass, repeat exactly the eight variants and 48 profiles:
both corpora, three alternating-order rounds, 20 warmup/100 measured updates,
same Transformer/data/BF16/optimizer and unchanged resource gates. Freeze
sources before the screen; no source changes or concurrent GPU work. Only a
passing primary can enter the originally registered eleven-variant,
two-corpus language screen. Retire a resource failure without substituting a
one-step control or relaxing the slowdown gate.

This is an execution hypothesis for established recurrence and warp reductions,
not a novelty claim. No language, inference or sustained resource benefit is
established by an implementation or short resource screen alone.
