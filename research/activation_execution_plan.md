# Rational activation execution audit: H036

The completed H035 architecture screen has a useful quality result: selected
BlockShuffle + rational NLL 5.898003 passes both full-reference quality limits,
beats calibrated narrow, and improves over its selected unchanged base. However,
eager allocated training memory fails the frozen limit. No 800-step run is earned.
This separate audit tests whether pointwise compiler fusion can remove that
execution cost while preserving the learned function and gradients.

## Fixed implementation and numerical checks

Compile only `curve(up) * value` using installed PyTorch 2.14 Inductor, static
shapes, fullgraph=True, no CUDA graphs, and `emulate_precision_casts=True`.
The installed compiler documents that this option retains low-precision
rounding between fused operations. Keep original parameter objects/names and
activation/product checkpointing. Projections, attention and optimizer stay native.
No packed weights or new learned parameters. Existing serving fusion is not used.
See [torch.compile](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html).

Before memory measurements, require independent randomized CUDA product output
relative L2 <0.003 and every input/shape-parameter gradient relative L2 <0.01.
On the selected trained checkpoint and a fixed real-token batch, compare native
and compiled model logits and all gradients: logits max absolute error <=0.05,
relative L2 <=0.003; global gradient relative L2 <=0.01 and activation-only
relative L2 <=0.02. All values must be finite. Full 322,688-target validation
must reproduce native NLL before replacement and change by <=0.01% relative
after replacement. Check source checkpoint hashes and original weight identities.

Use fixed fidelity sampling seed 9017 and profiling sampling seed 19017. Set the compiler recompile limit to 64 to accommodate layer instances and the final partial validation batch. These are frozen before profiling.

## Four isolated memory workers

Use selected WikiText seed17/200-step full GELU, full SwiGLU and rational
BlockShuffle checkpoints. Cases: full GELU native; full SwiGLU native; rational
BlockShuffle native; rational BlockShuffle compiled product. One GPU worker at
a time, 900-second deadline, BF16 B16/context128, original model dimensions.
Load AdamW state from each selected checkpoint and preserve its parameter-group
assignments. Use the same fixed sampled minibatches in every worker. Ten warmup
optimizer steps plus twenty measured steps exercise real forward/backward,
clipping and AdamW allocation. These updates are transient profiling work;
no updated weights are saved or scored as a new training result.

Run correctness and original-checkpoint validation first. Restore model and
optimizer states before the warmup/profiling phase. Reset allocated peak after
warmup and zeroing gradients. Time with CUDA synchronization; include sampling,
forward/backward, clipping and optimizer step. Report peak/resident/reserved
bytes and all finite gradients. Compiler setup/cache space is separate. Exclude
compilation from steady-state peak/timing and do not equate PyTorch allocations
with total driver VRAM.

Execution passes only if compiled product satisfies numerical checks and its
allocated training peak is <=1.10 times BOTH newly measured full references.
Report reduction versus native rational BlockShuffle, regardless of pass/fail.
Historical memory values are context, not the denominator of this new gate.
Serial timing is descriptive; there is no equal-compiler full-model speed claim.

This is an execution/derivative audit of a selected checkpoint, not evidence
that a compiled 800-step trajectory preserves the architecture result. Even a
pass still needs a bounded full-training repeat before larger-budget promotion,
independent seeds and corpus transfer. Failure stays documented; do not change
precision/tolerances or memory denominators after observing results.
