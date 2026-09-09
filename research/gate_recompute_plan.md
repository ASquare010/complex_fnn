# Gate-only recomputation: frozen memory check

The h=1024 candidate improves NLL to 3.098982 with 70.3125% fewer FFN weights,
but its 300809216-byte training peak exceeds the dense SwiGLU reference by
about 12.36%. Keep the original memory gate rather than ignoring this result.

Checkpoint only f(u,v)=SiLU(u)*v after the up/value projections. Backward
recomputes this elementwise expression; it does not rerun any matrix projection.
The mathematical function and gradients are unchanged. Independently compare
CPU FP32 and CUDA BF16 loss and every parameter gradient bit-for-bit first.
Use PyTorch's non-reentrant checkpoint implementation with no RNG preservation,
because this expression contains no random operation.
[Official checkpoint documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html).

This should avoid retaining the SiLU output between forward and backward,
but real savings must be measured. Freeze two 20-step profile runs at the same
h=1024 configuration and seed with recomputation off/on. The short runs are
memory/implementation checks, not new quality evidence. Record every memory
and throughput cost. If the on-profile brings peak memory within the threshold,
repeat the locked h=1024 candidate with recomputation at 800 steps on seeds 17, 29, 43,
and the parameter-matched narrow h152 and full SwiGLU controls on seeds 29, 43.
The existing seed 17 controls already use the same learning/data budget.

The 800-step seed 17 recomputation run must reproduce the existing candidate's
loss; otherwise investigate before interpreting the multi-seed comparison.
Inference is unaffected by recomputation and requires its separate cache audit.
This is an execution-schedule optimization, not a new learned architecture.


## Checkpoint implementation observation and native backward check
The checkpoint profile reduces peak allocation from 300,809,216 to 287,702,016 bytes,
but its 20-step NLL differs by 0.000173. A repeated off-profile is bitwise identical
to the first off-profile, so this is not attributed to ordinary run variation.
A full-size initial backward and five paired training steps match exactly; the
longer finite-precision divergence is still a limitation. Do not claim bitwise
identical complete checkpointed training.

Before the multi-seed run, test a simpler native-backward implementation that
saves u,v, recomputes SiLU(u) in backward, and calls PyTorch's existing
aten.silu_backward(grad*v,u). No custom CUDA kernel and no projection recomputation.
Check first/second derivatives and CPU/BF16 equivalence, then a separate 20-step
native profile. Record this as --gate-recompute-method native so the checkpoint
experiment remains independently reproducible. Select native only if it matches
the ordinary profile and meets the same memory gate; otherwise investigate.


Native profile outcome: all saved model weights and final NLL match the ordinary
20-step control exactly. Peak allocation is 285,604,864 bytes, down 15,204,352 bytes.
This meets the original memory limit in the profile. First and second derivative
tests pass; a differentiable analytical derivative handles higher-order requests
because the installed native aten.silu_backward lacks that derivative.
The locked three-seed 800-step batch now uses the native method.


## Long-run numerical check and interpretation amendment
At 800 steps, native recomputation reaches NLL 3.098955601 versus
3.098981783 without recomputation (difference -0.000026181). The 20-step
bitwise match does not extend to the complete training trajectory. The two
histories begin showing small differences at step 300. This fails a literal
bitwise-reproduction criterion; we do not claim long-run numerical identity.
The backward formula and independent derivative checks are unchanged, and the
native recipe is recorded explicitly in all three replication configurations.
The final comparison therefore evaluates the complete native-recomputation
training recipe. It does not attribute the tiny loss difference to a quality
benefit of recomputation. The raw trajectory and parameter differences are
saved in results/verification/gate_native_800.json.
The measured 800-step native peak remains 285,604,864 bytes, 6.68% above the
full SwiGLU reference and below the 10% memory-increase limit.
