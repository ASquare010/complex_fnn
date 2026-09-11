# H102: scoped replication after H101's fidelity failure

Frozen after H101 was deliberately stopped at 22 completed cases. Do not rerun
or overwrite those cases. All settings and gates below remain those of H101.

H101 full GELU loss chunking reproducibly changes final NLL from6.935556 to
8.224111 after12 constant-rate updates. A fresh process and saved-checkpoint
rescore reproduce both values. Same-state BF16 gradients differ by up to0.627%
initially and1.977% at the native endpoint; this supports numerical sensitivity,
not a proof of the source of amplification. A local gradient tolerance is not a
training-trajectory guarantee. Full GELU chunking is not promoted.

Narrow GELU's loss-only option saves21.30% allocated training memory in the
completed case, with close final NLL and modest measured time cost. Full SwiGLU
also preserves that case's NLL but saves little memory at T128. Replicate these
two separately; the latter is an explicit counterexample to universal savings.
FFN chunking is not pursued: it adds substantial time without material savings.

Run native, block checkpoint and block+loss chunks for both architectures at
B16/T128 and B8/T512, seeds17/29/43:36 fresh sequential processes,12 steps each.
Use H101's exact run_case function, immutable source and dataset, optimizer,
batch order,512-token chunks,4 warmup/8 timed steps, validation and gates.
The classifier and all model computation remain BF16 autocast with FP32 master
weights. Do not adjust precision, learning rate, tolerance or checkpoint cuts.
All36 endpoints are saved locally by the existing diagnostic worker so they can
be independently loaded and rescored. This is about1.8 GiB of checkpoint weights;
optimizer moments are not archived. Each worker has a separate output directory.

Report all outcomes including failures. The scoped memory option qualifies for
an architecture/shape only if every seed meets H101's >=15% memory reduction,
<=25% update-time cost versus block checkpoint and <=1% final NLL difference
versus native. Compare identical parameter counts and token budgets. Bootstrap
or significance claims are not justified by these brief device timing windows.

Do not label H102 a recovery of full GELU or a new activation/breakthrough. This
is a narrower evidence-based allocation to test whether the partial memory gain
survives repetition. The current general proof/research objective remains open.
