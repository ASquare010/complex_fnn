# H146: does recomputation change the modulation memory tradeoff?

Previous turn: progress. H145 rejected two isolated eager FFNs because reduced
parameter count did not meet measured VRAM savings. Preserve that failure.
Test eight residual FFNs, F_stack(x)=composition of x+FFN(x)/sqrt(8), at d384.
This introduces realistic accumulation of saved intermediates with depth, without
claiming a Transformer or a language benchmark. No normalization or stochastic ops.

Five architectures: full GELU384, full SwiGLU256, budget GELU288, shared gate192,
split base192/gate96. Use H145's exact individual module implementation and
name-local initialization with independent block seeds seed+101*layer.
Each architecture gets both eager and whole-FFN non-reentrant checkpoint execution.
Checkpoint each entire residual step; preserve_rng_state=False is valid because
there are no stochastic operations inside it. This is known recomputation, not
new architecture or novelty evidence. See PyTorch's official checkpoint docs.
Both baselines receive the same memory optimization as candidates.

Before training: FP64 eager/checkpoint outputs and all input/parameter gradients
must agree within rtol1e-10/atol1e-12 for all five architectures, with gate weights
perturbed away from zero. Trace saved tensors on CPU at batch2048/d384/depth8:
deduplicate storage, separately classify parameters, original input and other
activations. Hooks count retained storage after forward, not total GPU peak.
Keep hook traces outside all timed/VRAM GPU runs.

Frozen resource screen: batches128/2048, seeds83/97/109, ten architecture/execution
arms, 12 AdamW updates each: 60 runs,720 updates/backwards,61 clean GPU boundaries.
H145 recipe otherwise: one Gaussian batch with independent orthogonal linear
target; LR.003, betas(.9,.95), zero decay, clip1, FP32/TF32off, four CPU threads,
ordinary eager kernels/default workspace, input gradients enabled. Four warmups,
eight complete CUDA/wall update timings, reset gradients outside timer. Rotate
arm order by fixture. Final diagnostic score in job peak and outside timing.
No validation or quality claim. Save final states, histories, gradients and counts.

Gates, prospectively fixed: each dynamic arm must use <=80% parameters, <=90%
peak allocation, <=115% median CUDA AND wall time versus BOTH full controls
with the SAME execution mode in every fixture. Candidate and both control
half-window median timing stability must be <=115%. Budget GELU is a mandatory
resource comparator. Also show checkpoint/eager ratios within each architecture.
If all pass, this only earns a separate quality study; no quality is inferred.
Failure closes these settings, with no hidden repeats or changed thresholds.

Independently score all60 saved models on CPU in FP64 using explicit matrices
and differently sized batches257; relative final-MSE tolerance1e-5. Regenerate
all six datasets bitwise, verify optimizer counters12, finite states, counts,
recorded timings and boundary cleanup. Audit performs no backward/GPU work.
Keep maintained code/defaults unchanged and preserve H145 receipt chain.
