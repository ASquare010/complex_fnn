# H066: full-model staged rational resource and update study

Frozen after H065's twelve exact product comparisons and final audit, before
implementation of full-model adapters or any worker. This follows completed
PROGRESS and tests the real remaining memory obstacle. No new architecture.

## Fixed treatment and controls

Use H065's byte-frozen four-region rational implementation. Bind its residual
method to existing activation objects so all parameter identities, state keys and
optimizer references stay intact. Retain native outer product recomputation and
whole-block recomputation. No alternate partitions or other model changes.

For fair dense controls, compare each original whole-block implementation with
one additional native activation/product checkpoint: GELU(up), or SiLU(up)*gate.
The same up/gate/down matrices and native functions remain. Apply these adapters
only in training with gradients enabled; evaluation uses original forward code.
Dense inner checkpointing is also required to preserve measured numerical updates.
Use the lower measured memory of each full control's native and inner treatment
for the candidate's gate. Do not compare only against untreated full memory.

Six fresh sequential CUDA workers, in order: full SwiGLU native, full SwiGLU inner,
full GELU native, full GELU inner, rational native, rational staged. All use scope
block, B16/T128/d384/L8/V4096, BF16 with FP32 parameters, four CPU threads, and
H063's selected seed-17, 200-step checkpoints and optimizer moments. The original
per-recipe learning rate, fan-in treatment, decay, clipping and initialization
remain. No corpus cache is allocated and no language validation/test is scored.

## Identical loop and exactness

Reuse H063's existing worker function directly with isolated root/plan/model-loader
bindings. Its training loop source stays unchanged. Each worker runs one initial
synthetic loss/backward probe, then 20 synthetic updates using H063's fixed token
stream seed 60017. Hold each selected peak rate constant. Time updates 11-20 after
10 warmup updates with CUDA synchronization. Record allocated and reserved peak,
all 20 losses and pre-clip norms, clipping fraction, diagnostics, original and
final states, token/RNG hashes and all initial gradients. Save final checkpoints
at synthetic step 220, distinct from all original LM/profile runs.

Compare each treatment to its own fresh native reference. Require exact initial
logits/loss/every gradient, all 20 losses/norms, and final weights plus all AdamW
moments. Parameter objects and optimizer references must survive adapter binding.
Require finite diagnostics/weights/moments and unchanged sampling/RNG. No claim
that intermediate weight tensors were archived at every update.

Six workers mean 120 optimizer updates, 245,760 synthetic training targets and
12,288 initial-probe targets. Initial/final diagnostic forwards add no scored loss
targets. All are bounded qualification, not corpus training. Preserve every process
failure; no automatic worker retry, source mutation during workers or extra cells.

## Promotion gates fixed in advance

- Every numerical comparison above must be exact for rational and both dense adapters.
- Rational retains at least 70% fewer FFN weights.
- Staged rational peak allocated memory must be <=90% of fresh native rational.
- It must be <=110% of the lower native/inner peak for EACH full dense control.
- Median staged update time must be <=125% of fresh native rational time.

The last gate limits an additional execution penalty; it is not a claim of faster
training than dense. Report absolute timings and single-device/short-window limits.
Passing all gates earns a separately specified full native 200-step language
repeat. It does not establish language quality, convergence, general superiority,
new mathematical capacity, novelty or achievement of the full research goal.
Failure closes this fixed repair without another partition, tolerance relaxation,
compiler/dtype/offload/batch change or longer allocation. Existing negative results
and the three-folder/six-variant shortlist remain intact.

Sources and motivation: [H065 exact local qualification](rational_staged_results.md),
[H064 allocation diagnosis](rational_memory_results.md),
[H063 common worker and same-scope controls](native_recompute_results.md), and
[PyTorch native checkpoint API](https://docs.pytorch.org/docs/2.14/checkpoint.html).
