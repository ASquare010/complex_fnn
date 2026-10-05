# PairFlux: a small FFN with learned feature exchange

Status: exploratory hypothesis, registered before its first training screen.
The first screen is now complete: [results and decision](retired_models/pairflux_transformer/result.md).
The combined recipe failed qualification; the equations below preserve the
original hypothesis rather than retrospectively turning its ablations into a win.
Scope: replace only the token-local FFN. Keep embeddings, full causal attention,
normalization, residual paths, tokenizer, data and optimizer fixed. The original
[Atlas ambition](research_goal.md) remains separate. This proposal implements the
active [Step 1 goal](step_1_ffn.md); it does not implement memory or recurrence.

## The research question

Can a narrow hidden layer recover useful feature interactions through cheap,
input-dependent exchange, rather than paying for a second input projection or a
very wide hidden layer? Then can a small learned curve adapt the resulting units
without needing a large spline table or an iterative curve inversion?

A conventional FFN already mixes every input feature through its projections.
It is incorrect to say its neurons cannot communicate. The proposed change is
explicit **nonlinear interaction between projected hidden features before the
output projection**. This is a hypothesis about parameter efficiency, not a way
to create information absent from the input or increase storage for free.

## Three directions and the first implementation

1. **Conservative pair exchange (implemented):** borrow a local flux update from
   numerical conservation laws. Two hidden units compute a joint signal and send
   it in opposite directions. Learned input projections choose the paired basis.
   Two fixed matchings provide a small interaction graph without hidden attention.
2. **Moment-conditioned groups (unimplemented):** compute group mean and variance,
   then let each unit's activation depend on its contrast against those statistics.
   This could communicate group context at linear cost. It overlaps substantially
   with normalization and channel gating; the control must distinguish those.
3. **Sparse learned competition (unimplemented):** let small feature groups share a
   bounded activity budget, with learned competition and cooperation. This could
   select useful combinations without large routing matrices. It overlaps with
   maxout, divisive normalization and mixture routing, and could discard weak but
   important evidence. Hard routing would also complicate optimization.

We implement direction 1 first because its equations, complexity and attribution
controls are straightforward. These are research directions, not three claims
of undiscovered architectures. None is established as novel or superior.

Two further equations make the unimplemented directions concrete. For a group
of `n` projected features, moment mixing could use:

```text
m = mean(u), v = mean(u^2) - m^2
y_i = SiLU(u_i) + alpha_group u_i m / sqrt(v + epsilon)
```

The mathematical link is `(sum u)^2 = sum u^2 + 2 sum_{i<j} u_i u_j`:
an aggregate of many pair interactions can be computed without enumerating
every pair. This is an exact algebraic identity, not a discovery of new
information. The proposed primitive trades the identity of individual pairs
for an inexpensive group summary. Its key risk is collapsing distinct patterns
to the same moments; a normalization-only control is essential. The equation is
not bounded as written, and a stable coefficient/normalization design must be
registered and checked before implementation or training.

Competition could instead use a small group distribution:

```text
p = softmax(u / tau_group)
y_i = SiLU(u_i) ((1 - lambda_group) + lambda_group n p_i)
```

With positive temperature and lambda constrained to [0,1], lambda=0 recovers
plain SiLU. This borrows the selection idea of attention without Q/K/V or a
pairwise score matrix. It is still softmax gating, with established relatives;
there is no basis to claim it is a new invention. It may amplify a spurious
dominant feature, suppress important weak evidence, or spend more runtime than
it saves. A fixed-temperature/fixed-gate comparison and real-text tests would
be necessary. Both alternatives have linear feature-operation cost, on top of
the same two projections. Neither has been implemented or measured here.

## Equation

For one token, let `u = W_up x`, with model width `d` and hidden width `h`.
For each pair `(a,b)` in a matching:

```text
c = 0.25 tanh(alpha)
q = c a b / (1 + |a| + |b|)
(a', b') = (a + q, b - q)
```

Use two matchings in sequence: adjacent channels `(0,1), (2,3), ...`, then
opposite halves `(0,h/2), (1,h/2+1), ...`. Require `h` divisible by four.
There is one learned coefficient per pair per matching. Each final unit depends
on at most four original projected units; this is not all-to-all hidden mixing.
The dense input projection still supplies global access to input channels.

The exchange preserves the pair sum exactly in real arithmetic. It neither
preserves energy nor proves information preservation or invertibility. Because
`|q| <= 0.25 min(|a|, |b|)`, one exchange cannot transfer an arbitrarily large
amount. This bound is not a proof of whole-network stability. Evaluate the product
in factored form to avoid an unnecessarily large intermediate `a*b`.

Next use a learned local quadratic residual:

```text
t = clip(u, 0, 1)
phi_i(u) = SiLU(u) + 0.25 tanh(beta_i) t (1-t)
FFN(x) = W_down phi(exchange(exchange(W_up x)))
```

The compact quadratic basis is inspired by the supplied fixed-horizontal-control
Bézier idea. It is a bounded correction to SiLU, not the original two-control
parametric Bézier activation. It needs no square root or inverse solve. Its
support boundaries have derivative jumps when beta is nonzero; it is not smooth
there. The correction magnitude is at most 1/16. Initialize beta to zero and
alpha to 0.1. These choices are frozen for the first screen, not tuned winners.

The two matchings use `h` scalar coefficients and the curve uses `h` more.
Total FFN parameters per layer: `2dh + 2h`. Projection FLOPs per token per layer:
`4dh`, counting a multiply-add as two operations. Exchange, division, absolute
value, SiLU, tanh and curve operations are additional linear-in-h work; the reported
projection count excludes them. GPU time determines practical efficiency.
Temporary tensors and saved backward intermediates can negate activation savings.
No KV cache, token mixing, softmax or quadratic hidden interaction matrix is added.

At d=192, h=224 and four layers, the FFN has 345,856 parameters, approximately
70.7% fewer than full SwiGLU h=512. Total model size is 1,723,840 parameters,
approximately 32.6% smaller. Embeddings and attention remain a large fixed cost.

## Predictions and ways this can fail

Prediction: joint pair products capture conjunctions that a similarly small
one-activation FFN may need additional hidden units to approximate. The learned
curve may adapt local feature sensitivity after exchange. The mechanism only
earns credit if removing it damages quality under equal training conditions.

Likely failures: pair conservation restricts useful output directions; fixed
matching is a poor basis; bounded products are too weak; the curve overfits;
multiple elementwise kernels cost more time or memory than saved projections;
parameter efficiency fails on real text. Two matchings do not recover all pairwise
products or provide a general increase in knowledge capacity.

## Closest prior work and limits of originality

* [GLU variants](https://arxiv.org/abs/2002.05202) already use multiplicative
  feature interactions in Transformer FFNs. PairFlux reuses one projected bank,
  replaces an independent gate projection with sparse antisymmetric exchange,
  and is more constrained than general bilinear gating.
* [Bilinear MLPs](https://arxiv.org/abs/2410.08417) and
  [the bilinear technical note](https://arxiv.org/abs/2305.03452) make pair-product
  computation established prior art. Multiplication itself is not our invention.
* [Learned activations](https://arxiv.org/abs/1412.6830) and
  [KAN](https://arxiv.org/abs/2404.19756) establish learning nonlinear shapes.
  The bounded quadratic residual is a small restricted instance, not a new field.
  [CLF's Bézier-based network proposal](https://openreview.net/pdf?id=dxMffCAd4w)
  also illustrates direct curve-based neural parameterization; that manuscript
  proposes a different network and does not validate this FFN.
* [Butterfly transforms](https://arxiv.org/abs/1903.05895) establish sparse staged
  pair connectivity. Our two nonlinear exchange stages are not a full butterfly
  transform or a claim to its algorithmic results.
* [Learnable wavelet compression](https://arxiv.org/abs/2004.09569) and
  [lifting-based graph wavelets](https://arxiv.org/abs/2108.01660) connect learned
  local transforms with efficient representation. PairFlux has no wavelet
  reconstruction constraint, spatial topology or graph attention.

A focused primary-source search found these overlaps. It does not establish
absence of an identical prior method. The potential contribution is the specific
normalized conservative interaction inside a compact FFN, supported by an ablation
and actual resource measurements if it succeeds. Do not call it undiscovered.

## Frozen development screen

Use the existing deduplicated `dump/data/tasks-v1`, width 192, four layers,
six heads, context 256, vocabulary budget 4096, batch eight, CUDA BF16, AdamW,
seed 101, rate 0.0006, 2,000 updates, 200 warmup updates and validation every 200.
Use the shared trainer and identical sampled batches. Score complete validation
at the last checkpoint. Do not open test or OOD for mechanism selection.

Compare full SwiGLU h=512, full GELU h=768, narrow SwiGLU h=152, narrow GELU h=228,
and the previous single-pass BlockShuffle h=1024, groups=8. Also run four PairFlux
modes at h=224: complete, without exchange, without curve, and plain SiLU without
either. Ablations remove their unused coefficients; report this small count
difference. Shared projection initialization is identical across these four modes.

Check transport conservation, independent scalar equation agreement, gradients,
token locality, Transformer causality, parameter counts and finite BF16 backward.
Profile repeated training steps after warmup, alternate order over three rounds,
and record allocated and reserved memory. These short real-task-batch profiles
are hardware screens, not proof of sustained training or serving speed. Keep software source frozen
throughout training; rerun controls rather than mixing old source fingerprints.

Decision: any nonfinite result or 6 GiB reserved-memory breach stops that recipe.
If the complete mechanism loses to the plain ablation or strongest compact dense
control, do not claim its components improve accuracy. If it is slower or uses
more memory than full SwiGLU, do not claim combined efficiency. One seed on generated
tasks cannot establish an LLM improvement. Only a promising screen warrants equal
three-rate/three-seed tuning and the TinyStories/WikiText confirmation from Step 1.
Results and the continue/revise/retire decision belong in the model's `result.md`.
