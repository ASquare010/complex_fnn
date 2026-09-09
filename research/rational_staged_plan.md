# H065: staged rational recomputation qualification

Frozen before candidate implementation or measurements. The preceding goal turn
H064 made PROGRESS: exact allocator accounting locates eight live 16 MiB rational
pointwise allocations during backward recomputation, while other origins remain
partly unresolved. The research goal and prior failed memory/quality gates remain.

## One fixed mechanism

Keep the homogeneous rational function and every native pointwise dtype/cast.
Let w=1/(1+abs(z)), u=z*w, D=w^2+beta*u^2 and
r0=w^2/D, r1=u*w/D, r2=u^2/D, r3=z*r2. The correction remains
0.25*sum(a_j*r_j), with Python's original left-associated sum starting from zero.

Use exactly four non-reentrant native checkpoint regions: D; the weighted r0
term; the weighted r1 term; and the jointly computed weighted r2/r3 terms. Keep
r2 shared with r3. Keep coefficients, beta and the outer SiLU/BF16 addition and
product unchanged. The existing product checkpoint wraps the candidate too.
No alternate cuts, chunk sizes, compiler, dtype, offload or algebra search.

The proposed cut makes basis arrays internal to small checkpoint regions instead
of all being saved by the containing product's backward. This may reduce their
simultaneous lifetimes. It also changes operation scheduling and can change
floating-point gradient accumulation. Algebraic identity is not bitwise identity;
exactness must be measured before resource or optimizer experiments are earned.
Forward local temporaries and unclassified backward allocations may still dominate.

The meaningful later target is <=110% of both equivalently checkpointed full dense
controls and >=10% memory reduction from native rational. H063's historical
whole-block cap is approximately 361.666 MiB, about 98.666 MiB below H064's rational
peak. This is a target, not a predicted achieved saving. Fresh full controls with
equivalent activation/product checkpoint opportunities would be required for any
resource promotion. This first qualification makes no cross-model memory claim.

Checkpoint/rematerialization is established prior art, not a new primitive:
[PyTorch checkpoint API](https://docs.pytorch.org/docs/2.14/checkpoint.html),
[Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616), and
[PyTorch activation checkpoint overview](https://pytorch.org/blog/activation-checkpointing-techniques/).
Our hypothesis concerns one fixed execution partition of this retained formula.

## Fixed qualification budget

Keep all H064 source/configuration/test files unchanged. Put this unqualified
prototype and its driver only with its experiment artifacts; add no active
variant, model folder, training recipe or execution default.

Twelve paired product comparisons: seeds 17/29/43 x CPU FP32 or CUDA BF16 x zero
or trained coefficients. CPU shape is [3,11,64], GPU shape [16,128,2048], with
eight groups. The trained states are layers 0/3/7 respectively from H064's pinned
selected native rational seed-17, 200-step checkpoint. Input u, gate v and incoming
gradient are sampled on CPU in FP32 with seed 70000+seed, then cast to each test
precision/device. Both products use identical tensors and coefficient states.
Use four CPU threads and one sequential GPU process. No optimizer updates, no
corpus targets, no model training or validation. All twelve comparisons complete
unless a runtime/nonfinite failure makes subsequent work invalid; preserve failures.

For each pair, save input/state hashes and compare output plus every gradient
(u, v, theta_coefficients, theta_denominator). Record exact equality, nonfinite
status, mismatch counts, maximum absolute error and relative L2 error. Require
BITWISE equality for every compared tensor in every case. No tolerance relaxation.
No gradient/reduction implementation change after observing scientific outcomes.

Before paired comparisons, check FP64 values against the direct rational formula
on modest finite inputs and independently run torch.autograd.gradcheck for input
and both trainable parameter arrays. Check zero-shape equivalence to native SiLU,
parameter count and state-key identity, and absence of eval/no-grad checkpointing.
These are necessary mathematical/software checks, not the exactness promotion gate.

## Decision

Any paired exactness failure rejects this staged execution recipe as a faithful
memory repair. Close it without full-model memory workers, optimizer updates,
longer training, another partition, or an automatic tolerance-based retry.
If every gate passes, it earns a separately specified full-model resource and
update-fidelity comparison, not a quality/novelty claim or completed research goal.
Preserve candidate source, raw outputs, process records and all existing artifacts.
Record the measured outcome in the ledger and current state, keeping the shortlist.
