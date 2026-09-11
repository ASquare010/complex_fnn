# H115 — fixed-gradient transport through the decoder

Prospective, zero-optimizer-update diagnosis following H114's failed global
gradient gate and two failed replay audits. Preserve H114 unchanged. Its fixed
fidelity claim stays rejected; do not retrospectively loosen thresholds.

## Questions and controlled interventions

Separate (a) repeat variability with identical input gradient from (b) transport
of the small difference between native and chunked FP32 classifier gradients.
The former tests observed replay stability; the latter tests how a low-memory
classifier perturbation propagates through a BF16 decoder.

Use six existing H114 starting fixtures: WikiText-2 and TinyStories seed61,
each old block/loss-chunk state at step800, plus full GELU and SwiGLU seed17
at step3200. These are four narrow correlated fixtures and two single-seed full
controls, not six independent training seeds. Keep each original batch/context,
weights, vocabulary and first probe batch (seed +40000) exactly. No training,
validation optimization, checkpoint modification or architectural change.

Five decoder modes per fixture:

1. BF16, default SDPA backend, whole-block checkpoint (H114 reference).
2. BF16, default SDPA, no checkpoint (checkpoint-only intervention).
3. BF16, forced math SDPA, whole-block checkpoint (attention-backend intervention).
4. FP32, default SDPA, whole-block checkpoint (whole-decoder precision intervention).
5. FP32, forced math SDPA, whole-block checkpoint (crossed precision/backend control).

Use one shared encoder traversal and the maintained modules. Keep the chosen
SDPA context active through forward and backward, including recomputation.
Record actual attention-related autograd node names; do not infer a fused
backend from timing. No deterministic flags or cuBLAS workspace configuration
changes are introduced. TF32 stays disabled. Changing to FP32 or math attention
may change forward values; compare within a mode and report forward differences
across modes explicitly. A precision intervention is not a backward-only change.

## Fixed-cotangent experiment

For each fixture/mode capture normalized decoder hidden H with gradients enabled,
then detach it. Using this identical H and classifier W, compute loss, dH and dW
for native FP32 CE and existing checkpointed 512-token FP32 chunks. Save both
classifier derivative pairs. These two incoming dH tensors stay fixed thereafter.

For each incoming gradient, perform four fresh decoder forwards/backwards with
unchanged weights/tokens. Interleave native/chunk input gradients per repetition.
Inject dH with `H.backward(dH)`, isolating decoder transport from classifier
execution. Save decoder-only parameter gradients, plus the tied-embedding total
obtained by adding the saved direct classifier dW at the embedding weight.
This is the chain rule in real arithmetic; verify reconstruction against full
autograd on four tiny CPU-double GELU/SwiGLU cases before the main study.

Capture gradient tensors at the embedding output, eight block outputs and final
normalization output using hooks on those original tensors. Save forward hashes
at every boundary and require exact same-mode forward values for interpretation.
Store both first-repeat gradient/trace tensors; later repetitions retain full
hashes and all per-tensor replay/pair errors. Do not select a best repeat or
discard an outlier. Hooks and CPU copies can perturb scheduling; therefore this
is an instrumented diagnosis, not a definitive no-hook determinism proof.

Main budget: **30 conditions, 60 classifier backwards, 240 decoder backwards,
zero optimizer updates**. Qualification adds 12 small CPU classifier/decoder
backwards and eight scalar witness backwards (four CPU, four CUDA). The main
study does not score a corpus or allocate a training grid. Subsequent independent
artifact/math analysis uses zero additional backwards unless separately stated.

## Measured decisions and limits

Keep H114's original full-gradient transport thresholds: global relative L2
<=0.002 and each parameter <=0.02, with norm floors 1e-12 globally / 1e-8 per
tensor. A mode passes that diagnostic only if every paired repetition of every
fixture passes and forward boundaries repeat exactly within mode.

For observed repeat stability, keep the failed H114 recovery's stricter
1e-5 global / 1e-4 per-tensor limits on same-incoming-gradient repeats. Report
bitwise match counts separately. Four repeats cannot establish deterministic
behavior in general; successful instrumentation cannot repair H114's failed
uncontrolled replays. Ordinary threshold failures complete the fixed grid;
runtime/nonfinite/source-integrity failures stop and preserve partial outputs.

Report classifier dH/dW differences, decoder-only and total gradient differences,
boundary trace changes, repeat spread, cross-mode forward drift and checkpoint
comparisons. Relative-error amplification is descriptive; it is not a measured
Jacobian condition number or a proof of exploding/vanishing training gradients.

Count model/data, input-gradient tensors, normal workspaces and all phases in
diagnostic allocated/reserved peaks before any reset. No optimizer is resident,
so these are not whole-job memory measurements. Hooked wall time is only a cost
record, not a practical runtime gate. One GPU process, fresh models per condition,
normal cuBLAS workspace charging inside conditions, zero allocated/reserved
boundary checks between them. Use the established UV-managed Python/PyTorch,
four CPU threads and TF32 off. A mode can earn a separate uninstrumented resource
screen only; no fresh LM training or default change is directly earned.

## Scoped rounding witness and prior work

Around 1, BF16 spacing is 2^-7. Let m=1+2^-8 and g-=m-2^-23,
g+=m+2^-23. Their FP32 difference is 2^-22. Rounding them to BF16 gives
1 and 1+2^-7, so the absolute perturbation grows by 2^15=32768 at this cast.
Test this in actual CPU/CUDA autocast linear backward and a FP32 control.
It demonstrates a rounding-boundary effect in gradient transport, not an
ordinary derivative of quantization, a new theorem, or proof of H114's cause.

[PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
documents backend-dependent numerical behavior and FP32 intermediates for the
math backend with BF16 inputs. [Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
does not promise universal reproducibility. The result must identify measured
effects separately from these general warnings and from the existing classifier
mechanism in [H113](classifier_precision_results.md).
