# H135: ordinary-policy complete training

Previous goal turn: progress. H134 established 24.7-25.0% lower peak allocation
but failed ordinary-control runtime by 15.8-17.3%. Do not rescue that gate.

Hypothesis: using ordinary attention retains the memory benefit without the
strict-deterministic bundle's runtime cost. This is a new matched experiment;
a cross-study difference alone cannot identify a particular kernel cause.

Reuse H134's frozen complete-update loop and original Adam/clipping/evaluation.
Each corpus (WikiText-2 then TinyStories) runs reuse, ordinary, native, native,
ordinary, reuse in fresh processes. Ordinary = native loss, resident checkpoint
inputs. Native = native loss plus CPU checkpoint-input offload. Reuse = H128
buffer loss plus the same offload. ALL use ordinary/default attention,
deterministic=False, cudnn deterministic=False/benchmark=False, no cuBLAS env
or workspace setter, expected 8.125-MiB default capacity. FP32, TF32 off, four
CPU threads, PYTHONMALLOC=pymalloc, UV-managed Python. No clocks/power changes.

Unchanged budget: 360 updates/backwards (12x30), 1,474,560 training targets,
24 full study plus 12 independent native validation scores, 36 tensor artifacts,
804 memory intervals and 37 zero allocator boundaries. Same fixed source states,
data, batches, seeds and optimizer schedule. Repeats are not independent seeds.

Every corpus/control/repeat must meet GPU allocated peak ratio <=0.90, complete
CUDA/wall time ratio <=1.15, validation NLL ratio <=1.01, candidate pinned host
peak <=128 MiB, halves timing stability <=1.15 and passive telemetry coverage.
No discard or averaging away failed repeats. Whole-update and peak-memory scope
is exactly H134's; retain raw steps, mean, median, variance and sensor values.

Numerical policy differs prospectively because ordinary attention is already
known to be nondeterministic (H129), while H128's scalar operator qualification
is exact. Reuse inherited NumPy clipping/Adam checks and independent native
checkpoint scoring unchanged. Compare first raw AND clipped gradients against
each matched control. For each corpus/control/kind, measure native-repeat
relative L2 variation globally and maximum per-tensor. Candidate limits are:
  global = min(1e-5, max(1e-6, 10 * native_repeat_global));
  tensor = min(1e-4, max(1e-5, 10 * native_repeat_tensor)).
Native-repeat variation must itself be <= the respective caps. Both candidate
repeats must pass. Relative L2 uses the inherited symmetric norm denominator
with 1e-12 floor. Two repeats provide only a noise diagnostic, not a confidence
interval. Fixed caps prevent noisy controls relaxing acceptance without limit.

Do not demand full-model bitwise equality under this policy. Report bitwise
matches, first-gradient distances, native-repeat noise, and final model weight
distances descriptively. Final checkpoints must pass independent first-update,
finite/state/sampler checks and native validation. Long-run quality remains open.
H128 exact operator qualification is retained; H128/H130/H132/H134 failures stay.

Pass earns a fresh multi-seed longer-training test with ordinary native controls.
Failure picks a narrower follow-up from observed evidence. No architecture,
parameter reduction, novelty or breakthrough claim from this experiment.
