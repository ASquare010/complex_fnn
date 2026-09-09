# H050: Fixed-recipe 3,200-step duration test

Frozen after H049 completes all sixteen rate cells and all four recipes retain
an interior peak rate of 0.0012. Existing independent seeds for these unchanged
800-step recipes remain retained. Their final 200 steps still improve NLL by
2.6%-3.3%, so converged superiority is unproven. This is a duration test with
full and narrow controls, not a new architecture or an assertion of convergence.

## Fixed experiment

Train four fresh models from the same seed-17 initializations for 3,200 steps:
full SwiGLU, calibrated narrow SwiGLU, plain BlockShuffle, full GELU, in that
order. Keep each selected recipe's peak LR 0.0012, initialization, optimizer and
decay calibration, recomputation, precision and all non-FFN dimensions fixed.
Use width 384, eight layers, attention heads 6, context 128, vocabulary 4096,
batch 16, native BF16 and the original AdamW/clip-1 implementation. No activation,
compiler, data, architecture or optimizer change is allowed after scores.

Only steps and log_every change from the selected 800-step configuration:
steps=3200, log_every=800. The existing relative schedule automatically becomes
320-step warmup followed by cosine decay to 0.1 peak. This deliberately stretches
the schedule; it is not a checkpoint continuation or a claim that early partial
checkpoints equal separately trained shorter-budget models. Fewer diagnostic
logging calls affect overhead; matched four-recipe duration comparisons use the
same logging schedule. Model/evaluation computations and sampling remain fixed.

Each trial uses 6,553,600 sampled training tokens; four total 26,214,400 tokens.
This is about 2.125 training-cache token exposures per model, not 2.125 complete
ordered epochs because windows are sampled with replacement. Evaluate all
322,688 validation targets at initialization and steps 1/800/1600/2400/3200.
Official test remains unscored. The primary endpoint is final-step NLL, not the
best intermediate validation score. One fresh GPU worker at a time, at most
2400 seconds per trial. Preserve all four outcomes, diagnostics and failures.

## Preflight and reproducibility

Require H049 complete, every selected rate unchanged at 0.0012, all four winners
interior to its discrete grid and all local gates passing. Verify its complete
sixteen-cell records. Current computation must match the passing 225-test
snapshot; no shared NN/trainer/data/optimizer/diagnostic source changes.

In a separate read-only GPU sampling worker, reconstruct the exact CUDA data
sampler seeded 10017 for 3,200 batch calls. Validation/diagnostics use deterministic
validation windows and do not consume the training generator. Check the first
training batch hash against the retained qualification batch, the 800-batch
state against every selected old checkpoint, and record the expected final
3,200-batch generator state. Do not substitute a CPU RNG: the generator is
explicitly device-specific. No optimizer update or new validation score occurs
in this sampling worker.

Verify all eight immutable data hashes and the exact validation target stream,
including the last nine-window batch. Every new trial must reproduce its
same-recipe initial validation NLL and first pre-update training loss within
1e-7, preserving the first-batch contract while its first update uses the newly
stretched warmup. Verify complete histories, actual schedule, optimizer groups,
counts, finite parameter/gradient/layer statistics, source archives, checkpoints
and the reconstructed final CUDA sampling state. Source/plan and reference
hashes accompany the audit. No bitwise long-training trajectory claim is made.

A trainer-recorded nonfinite clip-norm failure is a numerical cell with no final
score. Preserve it and continue other planned recipes; exclude it from quality
promotion. Unexplained native/import/process failures stop for diagnosis and
an explicitly qualified unchanged continuation. No automatic retries, overwritten
runs, reuse of partial checkpoints or silent reassignment of failed work.

## Gates, trajectory qualification and next step

At the final 3,200-step endpoint, apply the same local gold gates: >=70% fewer
FFN weights, <=1% relative NLL cost against BOTH full controls, strictly beating
calibrated narrow, allocated training peak <=1.1 times EACH full control, and
finite diagnostics. Report a separate >=0.2% narrow margin, all per-recipe NLLs,
parameters, logical matrix FLOPs, allocation, clipping and descriptive throughput.
Different measurement sessions do not provide paired training-speed evidence.

Report NLL changes from steps 2400 to 3200 and within-duration trajectories.
Predeclare an operational late-plateau screen: absolute relative NLL change
between these endpoints <=0.2% for every recipe, with final NLL no more than
0.2% above its best recorded intermediate endpoint. This screen is a diagnostic,
not proof of optimal convergence. Failure means the duration is still insufficient
or the trajectory is degrading; do not declare convergence. Passing also requires
independent longer-budget/rate evidence before such a claim.

If quality fails, qualify the shorter-budget advantage and inspect the gap before
further promotion. If quality passes but memory fails, require a separate execution
study. If all local gates pass, freeze independent-seed longer-budget controls
before replication claims; rate rankings may change with duration, so a separate
duration-specific rate investigation can still be required. No automatic extra
steps or rate cells, novel primitive, universal optimality, state-of-the-art
superiority or whole-network gradient lower bound follows from this round.
