# Hypothesis: a wider nonlinear bank inside learned low-rank projections

Registered on 2026-10-04 after both signed recipes failed the completed
TinyStories screen and after the separately registered weight-spectrum
diagnostic. The signed campaign must finish unchanged on WikiText. No new GPU
work or production Python changes overlap it. All acceptance gates remain fixed.

## Evidence and question

Signed linear scored 2.603935 on TinyStories: better than open gain 2.619272,
but its gain over compact SwiGLU 2.606365 is only 0.093%, and its full-baseline
loss cost is 2.58%. Signed tanh scored 2.617161. Both required recipes fail.
These experiments retain only 576 detectors versus full SwiGLU's 1376.

The [weight diagnostic](ffn_rank_result.md) finds that rank 128 retains about
61–74% of trained readout squared-singular-value energy across both corpora,
versus about 47% initially. Input retention is about 44–63%. These are not
activation statistics or retained semantic information; important directions
may be discarded. They motivate testing allocation, not claiming compression
is safe. No pretrained weights or SVD bases are reused.

The question is whether a larger nonlinear bank can compensate for smaller
learned linear working spaces under the same weight budget. The input bottleneck
may instead discard essential cues, and the readout may restrict useful updates.

## Computation

For a normalized token x of width 512, independently in each layer:

```text
p = P x                              # input rank 128
[u, g] = A p                         # two banks of 1280 detectors
phi = SiLU(u) * g
q = B phi                            # output rank 128
FFN(x) = D q                         # width 512 residual update
```

P is 128-by-512, A is 2560-by-128, B is 128-by-1280, D is 512-by-128.
All four are trained from scratch. Attention, embeddings, normalization and
residual connections remain the common backbone. No attention or token mixing
is added. Detectors and their gates communicate through a common learned
input basis; B combines their responses into a learned readout space.

This restricts both linear projection ranks. The residual carries x, but does
not make discarded FFN input directions recoverable within that branch. The
wide feature bank cannot create information that its input bottleneck loses.
There is no universal expressivity or reasoning guarantee.

Four-layer FFN weights: 2,490,368, 70.54% fewer than full SwiGLU. Forward projection
FLOPs: 4,980,736/token versus 16,908,288. The custom adjoint recomputes A p, adding
2,621,440/token to backward projection work: total training projection work is
17,563,648/token versus full SwiGLU's 50,724,864. Report this additional work;
activation, reductions and optimizer costs are extra. Counts do not prove speed.

## Fixed allocation and controls

The candidate is hidden 1280, input rank 128, output rank 128. The matched thin
control is hidden 512, input rank 320, output rank 128. It has exactly the same
weight count and forward/recomputed training projection FLOPs. It spends more
weights on input directions and fewer on nonlinear detectors. Both use the same
output rank, so the comparison concerns that allocation, not an output-rank change.

Initialize P with standard deviation 0.02, the feature/gate halves of A with
1/sqrt(input_rank), B with 1/sqrt(output_rank), and D with 0.02/sqrt(2*layers).
Use independent named generators per matrix/half. This matches the expected
effective input/readout entry variance of the dense initializer, not its
correlations, distribution or initial logits. No dense-equivalence claim follows.

Include a second thin control with D's initialization multiplied by sqrt(1280/512),
to roughly match the wide bank's expected total initial update variance. The
ordinary thin control retains the standard scale. This is predeclared, not
chosen from validation. It does not match all distributions or prove a mechanism.

Language variants (nine per corpus): full native-activation SwiGLU hidden 1376,
full fused-activation SwiGLU hidden 1376, full GELU hidden 2064; compact fused
SwiGLU hidden 408 and compact GELU hidden 608; wide latent recompute, its identical
cached/native-autograd execution control, thin latent recompute, and thin latent
with calibrated initialization. Compact GELU exactly matches 2,490,368 weights;
compact SwiGLU has 0.658% more. Thin factorized controls are separate mechanism
controls, not ordinary dense references. Select the strongest full/ordinary
compact NLL per corpus. Require the candidate to beat both thin controls as well
as both registered dense quality margins on both corpora. Cached wide is an
execution-equivalence control, not one the candidate must beat in NLL.

## Execution and verification

The wide bank may erase activation-memory savings if cached. The custom inner
operation saves p and references to A/B, recomputes u/g/phi during backward, and
uses the existing fused SwiGLU activation/derivative kernel. P and D remain normal
linear modules. Release large intermediates after their last use. This is exact
recomputation, not a straight-through or approximate gradient. The cached control
uses the same fused activation and explicit matrix dtypes/layouts with ordinary
autograd. BF16 projections/gradient GEMMs and FP32 activation arithmetic match
the fair dense kernel control; return weight gradients in their master dtype.

For incoming t = dL/dq, flattened token axes:

```text
dB = t.T @ phi
dphi = t @ B
du = dphi * g * SiLU'(u)
dg = dphi * SiLU(u)
dA = concat(du,dg).T @ p
dp = concat(du,dg) @ A
```

Check independent CPU FP64 equations, gradients for p/A/B, finite differences,
cached/recomputed output and gradient equality, complete-model causality, token
locality, counts, save/load and exact optimizer recovery before profiling.
Check CUDA FP32/BF16 activation/weight/input-gradient references, recording
tolerances and errors. No higher-order derivative support is claimed.

Profile all three full controls, wide cached/recomputed and both thin controls:
three alternating-order rounds per corpus, 20 warmup plus 100 measured updates.
Same width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight,
BF16, four CPU threads, RTX 4070 Laptop GPU. Record cold warmup/compilation,
maximum allocated peak, median throughput and reserved usage. Require >=20%
memory reduction OR >=1.2x training throughput against every full control, <=5%
deterioration in the other resource, and <6 GiB reserved. Retire the fixed
candidate before language if it fails. Cached/thin controls remain labeled
controls even if their own resource numbers fail; they do not qualify instead.

If the candidate passes, integrate one readable complete Transformer, verify
production equivalence and shared-Trainer CPU resume, then run the nine variants
on both prepared corpora at seed 101, 2,000 updates / 4,096,000 supervised targets,
rate 0.0006, warmup 200 and AdamW decay 0.1. Use identical sampled windows and
full eligible validation. Test loss stays unopened. Survivors require equal
rate/seed tuning, fresh seeds, sustained resources and independent confirmation.

After training, set the expanded gates to one with the same activation backend,
zero even nonlinear detector features before B, and replace each FFN output by
its mean from 32 fixed training batches, seed 2027. These are frozen-weight
interventions, not retraining. Zeroing readout latents is not the same as zeroing
1280 detectors; implement the detector mask inside the inner operation and
verify its gradients if exposed. Report changed statistics and compare retrained
thin/dense controls. A loss penalty alone does not prove semantic roles or
explain improved generalization.

Predicted failures: rank 128 inputs may lose useful directions; output rank 128
may restrict updates; wide detectors may be redundant; factor initialization
may dominate apparent gains; recomputation may exceed the slowdown limit;
transient backward allocations may erase memory savings. Failure of any gate
retires this fixed recipe. Different ranks, width or budgets require a separate
registration before measurement.

## Prior work and originality

[Structured Pruning of Large Language Models](https://aclanthology.org/2020.emnlp-main.496/)
already uses low-rank parameterizations. [MobileBERT](https://arxiv.org/abs/2004.02984)
uses bottleneck structures and a teacher/student training design.
[SVD-LLM](https://arxiv.org/abs/2403.07378) studies compression of existing models.
[Activation recomputation](https://arxiv.org/abs/1604.06174) has established memory
tradeoffs. These precede this experiment. Their quality/resource results do not
establish this from-scratch FFN recipe's performance. Factorization, bottlenecks,
GLUs and recomputation are not inventions here; originality of this allocation
and controlled result remains unverified. The scientific claim must come from
successful paired comparisons and independent confirmation, not a new name.
