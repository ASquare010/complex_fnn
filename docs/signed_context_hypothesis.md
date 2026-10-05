# Hypothesis: active, signed context gates with a wider detector bank

Registered on 2026-10-04 after context-bank v1's completed TinyStories screen,
before this implementation or any signed-context measurements. V1's WikiText
campaign remains live and must finish unchanged. No new GPU work or production
Python changes may overlap it. The acceptance contract remains unchanged.

## Evidence and question

V1 product loss on TinyStories is 2.672406 versus full SwiGLU 2.538486 and compact
SwiGLU 2.615981. It fails both quality margins. Plain SiLU is 2.673912 and additive
context 2.674518. At frozen weights, coefficient zero changes product loss to
2.674651. The gain is small and does not establish richer inference computation.
The mean absolute tanh coefficient is 0.05930. Complete WikiText as registered;
the already failed required corpus rules out qualifying this fixed v1 recipe.

V1 starts its context branch closed: at a=0, C's gradient through the branch is
exactly zero. Its factor `1+tanh(a)*tanh(v)` is always nonnegative. These facts
suggest possible learning/expressivity limitations; the loss results do not
identify either as the cause. Test them with controls rather than assert them.

The next study also uses more of the permitted weight budget to retain 576
detectors. This is a separately registered allocation, not a retroactive change
to v1 or a relaxation of the >=70% reduction requirement.

## Equations and allocation

For one normalized token x, independently in each of four layers:

```text
u = Ux                  # 576 detectors
v = Cx                  # 64 context predicates
p_i = SiLU(u_i)
linear candidate: r_i = p_i * v_group(i)
bounded candidate: r_i = p_i * tanh(v_group(i))
FFN(x) = D r
```

Nine contiguous detectors share one context. U/C are one 640-row input projection
and D reads 576 features. There is no learned per-detector coefficient or direct
unmodulated branch in either signed candidate. C receives a gradient from the
first update, without first opening a coefficient. A predicate can reverse or
suppress a detector's output. The bounded candidate checks whether unbounded
predicate amplitude is needed. C reads all input coordinates; group membership
is fixed. No tokens mix within the FFN. Attention and embeddings remain unchanged.

The linear candidate is mathematically SwiGLU with repeated rows in its gate
matrix: `W_gate = R C`, where R repeats each context nine times. This is a
structural restriction of an established GLU, not a new category of computation.
The question is whether independent gates are redundant enough to spend their
weights on more detectors. The analogy to grouped-query attention is sharing
one conditioning signal across several responses; this is not dot-product
attention, a kernel approximation, or an information-creation guarantee.

Four-layer FFN weights: 2,490,368, 70.5426% fewer than full SwiGLU's 8,454,144.
Projection FLOPs: 4,980,736 per token versus 16,908,288, excluding nonlinear and
reduction work. Compact GELU hidden 608 has exactly the candidate weight count.
Compact fused-activation SwiGLU hidden 408 has 2,506,752, 0.658% more. The two gain
controls have 2,492,672, 0.093% more because they retain 576 coefficients/layer.
Plain SiLU hidden 576 has 2,359,296 and is an additional smaller dense control.

## Ten language variants and mechanism tests

Use full native-activation SwiGLU hidden 1376, fused-activation SwiGLU hidden 1376,
and GELU hidden 2064. Select the lowest full validation loss per corpus. Ordinary
compact controls are fused-activation SwiGLU hidden 408, GELU hidden 608 and plain
SiLU hidden 576; select the lowest compact loss per corpus.

At hidden 576/groups 64, retrain the same v1 gain equation twice: coefficients
initialized to zero (closed gain), and to one (open gain). All remaining U/C/D
initialization is identical. This isolates coefficient initialization within the
same function class. Neither signed candidate has the same initial function as
plain/gain; report this rather than claiming identical initial logits. Named
initialization for the backbone and corresponding U/C/D weights remains common.

Train both signed linear and signed tanh candidates. These are two predeclared
recipes, not an after-the-fact activation search. Each must independently pass
the resource gate and both dense quality margins on both corpora. Require it to
beat the retrained plain, closed-gain and open-gain controls too. Report both
candidates; a per-corpus choice of different candidates cannot qualify one recipe.
If both survive, choose the lower mean normalized validation loss across the
two corpora for further equal tuning and independent confirmation, with the
other reported as a development alternative. Fresh confirmation must account
for this development selection. Do not change the budget or discard failures.

Signed linear versus signed tanh isolates bounded predicate amplitude within
otherwise matching shapes. Signed versus open gain also changes mean/variance
and the function family; it does not isolate sign alone. If a signed candidate
survives, a separate positive-gate control is required before crediting sign
reversal as the explanation. No semantic role follows from a removal penalty.

After training, replace the signed context signal at frozen weights with (a) one
and (b) its per-layer/per-context training mean from 32 fixed training batches,
seed 2027. For tanh, average the post-tanh signal and inject an inverse-tanh
preactivation for the mean; use a direct override for exactly one so the same
activation backend remains available. For gain controls, zero coefficients with
the same backend. Also zero even detector indices at D's input and replace each
FFN's output with its training mean. Report changed activation statistics and
compare independently trained controls. Test loss stays unopened.

## Verification and resource gate

First check CPU FP64 independent detector-to-context selection, analytic
gradients and finite differences, including nonzero contexts. Check that signed
and open-gain C gradients are nonzero at the first update and closed-gain C
gradients are zero. Check complete-model causality, token locality, counts,
save/load and exact CPU optimizer recovery. CPU preparation can occur while v1
trains. CUDA checks and profiling must wait for its process to be terminal.

Use FP32 activation/gradient arithmetic for BF16 projected banks, cast final
features/input gradients to BF16. Save only the projected bank for the signed
adjoint; recompute SiLU and context nonlinearity. Independent FP32 scatter sums
must match the fused gradient's accumulation before the final BF16 cast. Verify
CUDA FP32/BF16 forward and gradient tolerances. No higher-order support is claimed.

Profile all three full controls, plain, closed gain, open gain and both signed
recipes: three alternating-order rounds on each corpus, 20 warmup plus 100
measured updates. Same width 512, four layers, 16 heads, vocabulary 4096, context
256, batch eight, BF16, four CPU threads, RTX 4070 Laptop GPU. Record cold
warmup/compilation, maximum allocated peak, median throughput and reserved memory.
Require >=20% memory reduction OR >=1.2x training throughput, <=5% deterioration
in the other resource against every full control, and <6 GiB reserved. Retire
each signed recipe that fails; use no rounded near misses. No quality claim from
a short profile. A hardware-failed recipe can only remain a labeled mechanism
control, not advance as a qualified candidate.

If any signed recipe passes, integrate a readable complete Transformer, verify
production equivalence and shared-Trainer resume, then execute all ten language
variants on both prepared corpora. Seed 101, 2,000 updates / 4,096,000 supervised
targets, rate 0.0006, warmup 200, AdamW decay 0.1, identical sampled windows and
full eligible validation. These are development results. Survivors need equal
rate/seed tuning, fresh seeds, sustained resources and independent confirmation.

Failure predictions: shared predicates may remain too coarse; nine detectors
may need independent contexts; losing the direct path may hurt easy patterns;
signed products may be unstable; the wider bank may lose memory/headroom; open
gain may be sufficient; both signed variants may lose to equally sized SwiGLU.
Any required failure retires that fixed recipe. A different group count, width
or training budget requires a separate registration before measurement.

## Prior work and originality

[GLU variants](https://arxiv.org/abs/2002.05202) establish products of learned
projections in Transformer FFNs. [Gated Channel Transformation](https://arxiv.org/abs/1909.11519)
studies economical channel modulation in vision. Our signed linear equation is
a tied-gate GLU; neither multiplication nor sharing is an invention here.
Priority for this exact allocation is unverified. Do not claim undiscovered
architecture or transfer prior papers' accuracy results to this language study.
