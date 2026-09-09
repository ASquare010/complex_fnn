# H038: frozen correction-only compilation diagnostic

Written before executing this backend. Full rational-product compilation passed
local tolerances but its 200-step repeat lost1.925% NLL. Hypothesis: retaining
native SiLU and gate-product forward/backward while compiling only the FP32
rational correction reduces gradient differences and keeps useful memory savings.
This is an execution ablation, not a new architecture or an attribution of the
previous failure to a proven cause. Keep the failed H036 artifacts unchanged.

First test zero-correction BF16 input/value gradients exactly equal to native,
and nonzero randomized coefficients for finite output/all parameter derivatives.
Use the existing trained native rational checkpoint and precisely the H036
fidelity batch(seed9017), full validation322,688 targets and 30 transient
profiling batches(seed19017;10warmup+20measured). No updated checkpoint is saved.
The native rational and both full-reference H036 memory measurements may be
reused only after their source archives and profiling-loop AST match, identical
minibatch hashes, original checkpoint hashes and software/hardware are verified.
Serial timings are descriptive, not paired comparisons.

Freeze a stricter all-parameter relative L2 gradient limit of0.0001 (H036
measured0.00078429). Retain logits max0.05/relative L2 0.003, shape-gradient
relative L2 0.02, all finite, full validation change<=0.01%, unchanged registered
parameter objects and memory<=1.1both full references. Stop on numerical or
infrastructure failure; retain the diagnostic/failure, do not relax thresholds.
A failed fidelity check does not earn transient memory profiling. One GPU worker,
900s deadline. The existing production training defaults remain eager.

Passing all gates only earns a separately frozen 200-step training repeat;
it cannot establish trajectory fidelity or longer-training promotion itself.

Historical archive qualification: the original rational-compiled worker recorded
a later hash for `activation_report.py` than its parent archive. This reporting
file is not part of profiling. Record this exact difference; require all other
worker source hashes and all parent-archive hashes to match. This known
non-execution difference is not evidence of a measurement change.
