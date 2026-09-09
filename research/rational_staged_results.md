# H065: staged rational execution passes local fidelity

**LOCALLY QUALIFIED, not yet a demonstrated memory repair.** All twelve fixed
native-versus-staged product comparisons have bitwise-identical output and every
gradient. These include full-size CUDA BF16 inputs and actual trained coefficient
states. No active model, recipe, execution default or parameter changes.

## Mechanism and scoped mathematics

The [frozen plan](rational_staged_plan.md) defines exactly four native non-reentrant
checkpoint regions: the homogeneous denominator; the weighted constant basis;
the weighted linear basis; and the joint weighted quadratic/cubic bases. Keeping
the latter together preserves the shared quadratic basis used by the cubic term.
The final weighted terms retain their original left-associated sum from zero.
The existing outer product checkpoint wraps both implementations.

With w=1/(1+abs(z)), u=z*w and D=w^2+beta*u^2, the bases remain
w^2/D, u*w/D, u^2/D and z*u^2/D. Since w is positive for finite real z and
D=w^2*(1+beta*z^2), these equal (1,z,z^2,z^3)/(1+beta*z^2) in real arithmetic.
The bounded beta remains strictly positive, and no trainable degree of freedom
is added or removed. This is an execution partition of the same function family.
It proves neither general bitwise equivalence nor faster or easier learning.

H064 identified 128 MiB of live pointwise allocations at the native rational peak.
The staged partition aims to avoid retaining all basis arrays together during
backward. That is a lifetime hypothesis; no allocation peak has been measured for
this candidate yet. Forward temporaries and unresolved native backward buffers
could still dominate. [Checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html)
and [rematerialization](https://arxiv.org/abs/2006.09616) are established methods;
this result is not a new activation or novelty claim.

## Fixed checks and measured outcome

| Inputs | Coefficient states | Seeds / source layers | Exact comparisons |
|---|---|---|---:|
| CPU FP32, 3x11x64, 8 groups | Zero; trained | 17/29/43 use layers 0/3/7 | 6/6 |
| CUDA BF16, 16x128x2048, 8 groups | Zero; trained | Same selected checkpoint layers | 6/6 |

The trained arrays come from H064's selected native seed-17, 200-step rational
checkpoint. The three comparison seeds generate synthetic inputs, gates and
incoming gradients; they are **not three independently trained models**. Both
products consume identical tensors. Every output and gradient for u, v,
theta_coefficients and theta_denominator matches bit for bit: 60 tensor
comparisons, zero mismatched elements and zero measured error. Parameters and RNG
remain unchanged in every pair. All values are finite.

Before these comparisons, FP64 outputs agree with the direct rational formula on
modest finite inputs, and torch.autograd.gradcheck independently checks input and
both trainable arrays. Zero coefficients match SiLU, state keys and the 40-parameter
activation count match native, and evaluation/no-grad bypass internal checkpointing.
The child qualification process completes first attempt in 8.715 seconds; this is
whole-process elapsed time, not an execution-speed benchmark.

There are **zero optimizer updates and zero corpus targets**. The paired checks
execute products only, not full Transformer training. All existing active source,
configuration and test files retain their H064 bytes and existing 108-test
qualification. No full-suite rerun is needed for unchanged active code; the new
prototype is isolated with this experiment's artifacts.

## Decision and preservation

This earns a separately frozen full-model memory and update-fidelity study. It
must retain exact outputs/gradients and optimizer behavior, and meet memory limits
against equally treated full controls. Local equality does not replace that study,
longer language training, multi-seed replication, convergence or scaling evidence.
No quality promotion is earned and the full research goal remains unmet.

[Candidate and qualification source](../results/rational_staged_v1/source/),
[protocol](../results/rational_staged_v1/protocol.json),
[result](../results/rational_staged_v1/result.json),
[source archive](../results/rational_staged_v1/source.zip),
[process record](../results/rational_staged_v1/process.json), and
[final audit](../results/verification/rational_staged_final_v1.json) preserve the
comparison. Each case includes original FP32 inputs, coefficient state, both
outputs and all gradients in a hashed tensor artifact. The active shortlist stays
at three model folders and six variants.

The earned [H066 full-model follow-up](staged_resource_results.md) now completes:
fidelity remains exact, but the 32 MiB saving fails the memory gates. This fixed
partition is rejected as a memory repair and receives no language repeat.
