# H074 - Full-model ungated GELU resources and update fidelity

Freeze before implementation and any resource worker. H073 completed fitting and
independently qualified both conventional GELU forms. This is the next earned
measurement, not language training or an active architecture addition. Preserve
the three-folder/six-variant tree, all prior source bytes, results and frozen plans.

## Models and initialization

Seven forms: full SwiGLU h1024, full GELU h1536, narrow SwiGLU h304, narrow GELU
h456, plain BlockShuffle SwiGLU h2048, same-width BlockShuffle GELU h2048, and
matched BlockShuffle GELU h3264. All use d384/L8/heads6/context128/vocab4096,
batch16, structured groups8, BF16 autocast with FP32 parameters, TF32 disabled,
four CPU threads, native eager Torch and one CUDA worker at a time.

Use fresh shared Transformer initialization at seed17; no trained checkpoint or
corpus cache. The existing name-local initializer gives identical non-FFN tensors
across forms and identical common up/down factors for plain and same-width GELU.
Dense narrow down weights receive the existing fan-in correction exactly once.
This uses standard Transformer initialization (down residual scale1/4), not
H073's standalone unit-row-variance scaling. Models have distinct initial
functions. Compare checkpoint execution options only within each model.

An isolated ModelConfig subclass labels blockshuffle_gelu and overrides its
parameter formula to two structured projections. A scoped factory adapter uses
the existing gated=False operator inside the unchanged Transformer constructor.
No active registry change. Count actual tensors and independently verify:
full FFN9,437,184/total15,735,168; both narrow and plain/matched GELU
FFN2,801,664/total9,099,648; same-width GELU FFN1,867,776/total8,165,760.
Each projection weight contributes one MAC per token; report that theoretical
linear-projection cost separately from measured full-model runtime.

## Equal execution options and bounded work

For each form run three fresh sequential processes in order: none (no outer or
inner checkpoint), block (existing whole-block checkpoint), block_inner (same
block checkpoint plus native GELU(up) or SiLU(up)*gate checkpoint). Inner uses
use_reentrant=False and preserve_rng_state=False; outer preserves RNG. It wraps
only the activation/product, preserving parameter objects, names and optimizer
references. Evaluation and no-grad calls use original forward functions.

Run forms in the table order above, completing all21 workers unless a runtime
or finite-value failure stops the coordinator. Each worker has one initial
loss/backward probe then20 synthetic updates. Token stream is the existing
CPU-generator seed60017, shape20x16x129, integers[0,4096). First128 tokens are
inputs, shifted128 targets. Save the exact stream once and verify every worker.
There are420 optimizer updates,860,160 training targets and43,008 no-update probe
targets. Diagnostics add no scored-loss targets; zero corpus targets.

AdamW base rate0.0012 held constant, betas(0.9,0.95), eps1e-8, weight decay0.1,
parameter decay, global clip1. Existing structured fan-in factor LR correction;
calibrated narrow down LR correction; full dense rates unchanged. All forms have
one fixed rate and equal update budgets. This rate is a resource/fidelity probe,
not selected for GELU quality. Save initial weights, parameter-group metadata,
all initial logits/gradient signatures, all20 losses/preclip norms/timings,
final weights/moments and CPU/CUDA RNG. Inspect initial/final per-layer activation
and gradient magnitudes, near-zero fractions and sampled activation slope
saturation. Require finite values and unchanged RNG during training.

Synchronize around every update. First10 updates warm up; measure median of
updates11-20 and allocated/reserved peaks reset after update10. Initial probes,
diagnostics and checkpoint serialization are excluded from the memory/time
window. Include actual model, gradients and optimizer states. No serving claim.

## Exactness and gates

Within each form, compare block and block_inner to none: identical initial
weights/logits/loss/every gradient, every20 loss/preclip norm, final weights and
all optimizer moments, stream and RNG. No cross-architecture exactness claim.
Check adapter parameter identities and optimizer references before/after binding.
Require finite diagnostics, final weights and moments. Do not relax tolerances
or retry scientifically on failure; retain failure logs/artifacts.

Select each form's lowest allocated-memory option, ties fastest measured median
update then fixed mode order. Both full references receive the same choices.
Each GELU candidate earns a separately frozen short language screen only if:

1. All21 workers finish and all within-form fidelity comparisons are exact.
2. Actual FFN reduction is at least70% and all diagnostics/state are finite.
3. Its selected peak allocation is <=110% of the lowest allocation of EACH full
   dense control across their three modes.
4. Its selected median update is <=125% of plain BlockShuffle's update using
   that same execution mode.

The runtime gate limits cost versus the compressed reference; it does not
establish a speedup over full dense. Report all modes, absolute time, throughput,
reserved memory and full-control ratios. No self-memory-repair gate applies to
these distinct architectures. Report both candidates even if one fails. Passing
qualifies only a separately frozen language comparison, followed by quality,
three-seed/convergence/scaling/broader-data requirements. Failure earns no
automatic kernel, width, rate, batch, dtype, offload or checkpoint expansion.

## Provenance and verification

Pin H073 final/result and all97 preceding sources, this plan and three new source
files. Record actual git state, environment, source archive, seed/configuration,
stream hash, PID/UTC, logs, return codes and terminal coordinator outcomes. Five
isolated harness tests verify actual/formula counts and scoped configuration,
shared non-FFN and same-width factor bytes, optimizer coverage/calibration,
inner adapter identity/evaluation fidelity, and gate/selection failure behavior.
They do not rerun or expand the prior108-test active suite. Use a hidden durable
coordinator with no automatic retries. Observe live processes before recovery.

Independent CPU analysis reloads all21 checkpoints, recomputes weight/moment
hashes and finite checks, independently compares signatures/losses/norms and
rederives mode selection and resource gates. Preserve data, all former results,
frozen plans and active sources. Update the ledger/current state with the actual
outcome. Known checkpoint/structured/GELU components imply no novelty claim.

References: [earned H073 comparison](ungated_fit_results.md),
[local GELU qualification and limits](ungated_blockshuffle_results.md),
[prior equivalent dense checkpoint controls](staged_resource_results.md).
