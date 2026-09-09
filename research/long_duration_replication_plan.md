# H051: Independent seeds for the 3,200-step comparison

Frozen after H050 completes and verifies all four seed-17 trials. BlockShuffle
passes the local engineering gates narrowly: +0.9819% NLL versus full SwiGLU,
+0.4323% versus full GELU and -0.1763% versus narrow, at 70.3125% FFN reduction.
The separate >=0.2% narrow margin and every late-plateau screen fail. These
observations motivate replication without changing the selected recipe.

## Fixed cohort and budget

Retain all four H050 seed-17 outcomes without retraining. Train precisely eight
fresh trials: the same four recipes for seeds 29 and 43. Use reverse H050 order
for seed 29 (full GELU, BlockShuffle, calibrated narrow, full SwiGLU), then H050
order for seed 43 (full SwiGLU, calibrated narrow, BlockShuffle, full GELU).

Only seed changes from H050: width 384, eight layers, attention heads 6, context
128, vocabulary 4096, batch 16, 3,200 steps, logging every 800, native BF16,
peak LR 0.0012, 320-step warmup and cosine decay to 0.1 peak. Keep initialization,
optimizer groups and decay, clip-1, recomputation, tokenizer, validation, and all
shared computations fixed. No activation, compiler or architecture changes.
These are fresh full schedules, not checkpoint continuations.

Each new model consumes 6,553,600 sampled training tokens; eight total 52,428,800.
Evaluate all 322,688 validation targets at initialization and steps
1/800/1600/2400/3200. Official test remains unscored. One fresh GPU worker at a
time, 2400-second ceiling per training worker. H050 wall times imply roughly
30 minutes of additional GPU work; actual durations are retained.

## Qualification and integrity

Require H050's verified local pass and exact 227-test computation snapshot.
The full suite used -p no:anyio after retained optional-compiler and plugin
startup failures. No assertions or GPU checks were removed. This qualification
does not assert that the intermittent native/toolchain cause is fixed.

Verify all H050 archives, plan and result hashes, twelve earlier selected
800-step reference cells, the eight immutable data files and exact validation
stream. Each new seed has a separate read-only CUDA sampler worker: reconstruct
3,200 actual batch calls, verify state after 800 against its retained checkpoint,
and record first-batch and final-state hashes. No CPU RNG substitution, optimizer
update or validation score occurs in that worker.

Each trial must reproduce its same-seed 800-step reference initialization NLL
and first pre-update training loss within 1e-7. Then stretched warmup changes
the updates. Verify exact model/training configuration, optimizer/init metadata,
counts and logical FLOPs, histories/schedule, finite weights/gradients/layer
statistics, source archives and final checkpoint sampling state. Keep all results.

Trainer-recorded nonfinite clip-norm failures are failed numerical cells; retain
them and continue the other prespecified cells. Unexplained import/native/process
failures stop for diagnosis. Resume requires an explicit failure/source/plan
qualification and no live worker. Never overwrite, restart a completed cell,
reuse a partial checkpoint, silently retry or assign a fabricated final NLL.

## Frozen decisions

Report every seed and mean/sample SD. Primary replication requires every one of
seeds 17/29/43 to pass the original local gates: >=70% fewer FFN weights, <=1%
relative final NLL cost versus BOTH full controls, strictly better final NLL than
calibrated narrow, allocated training peak <=1.1 times EACH full control, and
finite diagnostics. A passing mean cannot rescue a failed seed. Report the
separate >=0.2% narrow margin for each seed and for the mean without changing the
primary threshold after outcomes.

Report paired candidate-minus-control NLL differences, sample SD and exploratory
95% Student-t intervals (n=3, df=2), plus all relative differences. These are
small-sample descriptions, not proof of universal superiority or rate-optimal
performance. Keep the H050 operational late-plateau screen (absolute final-800
NLL change <=0.2%, final <=0.2% above best recorded endpoint) per recipe/seed;
no plateau test proves optimal convergence. Three-seed plateau qualification
requires every recipe/seed to pass. Preserve clipping and descriptive throughput;
no cross-session paired speed claim.

If any primary gate fails, qualify H050 as a single-seed pass and investigate
duration-specific optimization/representation before promotion. If all pass,
retain a replicated fixed-budget result, with convergence, duration-specific
rate ranking, larger scale, broader comparators and novelty still open. No
automatic additional steps, rates, activations or architecture searches follow
from either outcome. Freeze the next experiment separately after reviewing it.
