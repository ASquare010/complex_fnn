# H090 - One explicit precision-region recovery, then the frozen FP32 fitting grid

H088 remains a failed BF16 qualification with21 completed CPU checks and zero
training. H089's separately frozen32-evaluation diagnostic finds that converting
the activation input to FP32 and casting the complete activation once removes
all independent-reference tolerance violations for all four candidates. Every
FP32 native/region output and gradient is bit-exact. This is local diagnostic
evidence and an explicit reason for one qualification recovery, not a training
result or a reason to loosen tolerance.

## Sole execution change

Create subclasses of H088 PairTwist and LocalCurve whose forward methods call
the original forward on z.float() and cast the returned activation to z.dtype.
If z is FP64, keep it FP64. Restore original curve state and parameter/buffer
roles exactly. The independent reference receives the same FP32-region policy.
The rational and trigonometric/polynomial formulas remain separate.

At FP32/FP64, these conversions are identity operations. Thus the original
equations, parameters, optimizer coordinates and planned FP32 training arithmetic
are unchanged. BF16 forward/backward rounding is deliberately changed and must
be recorded as a different execution recipe. No BF16 training claim follows.

Bind only H088's model factory, reference evaluator and output roots using
individual scoped patch.object bindings. Do not mutate any H088/H089 file.
Keep all shapes, seeds, initialization, qualification thresholds and comparison
counts. Run all27 qualification cases once in results/neuron_geometry_recovery_v1.
Compare all six new CPU tensor payloads and the15 non-tensor CPU observations
with H088's completed results. Tensor payload equality must be exact; tensor
archive byte hashes need not be equal. Preserve H088's41 file hashes/sizes/mtimes.

## Conditional training allocation

Only after all27 checks and original-CPU comparisons pass, execute the UNCHANGED
H088 FP32 fitting loop with the new factory and output root. Use its exact14
forms, four target families, seeds17/29/43, rates0.001/0.003,600 updates,batch256,
input seed9814 and all diagnostics/statistics/plots/gates. H088 never generated
this data or ran training, so these336 cells are fresh, not training repetitions.
There are201,600 optimizer updates and51,609,600 example presentations if complete.

Keep full/narrow GELU,ReLU,SwiGLU, GroupSort, learned-offset, fixed-twist and linear
controls. All quality gates, fixed-twist checks and per-task caps remain exactly
those of [H088](neuron_geometry_plan.md). Native timing and memory still cannot
establish full-model practical superiority. No automatic longer run or promotion.

Freeze H089 result/process/protocol/source archive, H088 source and failure
records, the H087 release anchor,68 prior scientific source hashes and the three
new recovery sources plus this plan. Use durable source archive/readback,
one hidden GPU process, four CPU threads and TF32 off. Stop on failure. This is
one explicit qualification repetition across H088/H090, with zero H088 training
and no automatic second recovery. Preserve every failed or unexecuted stage.

Independently audit the complete checkpoints, every selection/reporting score,
initializations, data, decisions, summaries, plots and source preservation as
specified in H088. For independently reduced summary statistics, use atol/rtol
1e-12; actual checkpoint scores must reproduce exactly on the original device.
Keep raw tensors/histories local and commit only compact evidence and source.
