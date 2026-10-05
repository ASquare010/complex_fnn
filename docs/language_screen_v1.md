# Language screen v1: compact FFN candidates and strong dense controls

Registered before training on 2026-10-03. This is development, not confirmation.
Previous goal work made progress: it implemented and measured PairFlux, retired
the combined recipe, and identified that its apparent component gains might be
training-path effects. No candidate has met the active language requirements.

## Current acceptance contract

The user-defined goal requires all of these, without changing the backbone:

* At least 70% fewer FFN parameters than full SwiGLU.
* Language loss within 1% of the strongest full dense baseline, and at least 1%
  better than the strongest similarly sized dense FFN, on both TinyStories and
  WikiText. Compare within each corpus; its tokenizer differs from the other.
* At least 1.2x training throughput OR 20% lower peak training allocated VRAM.
  Predeclare "no material worsening" as no more than 5% deterioration in the
  other resource. Report reserved memory and all full controls as well.
* Repeated paired-seed improvements and an independent confirmation experiment.
* Removal experiments explaining the benefit and distinguishing training dynamics
  from inference computation. Verify prior work before claiming novelty.

Use the full dense recipe selected for best validation language loss on each
corpus as the primary resource comparator. Also report the full SwiGLU reference.
There is no speed credit for comparing against an unnecessarily fragmented gate
implementation: include SwiGLU with a fused up/gate projection, for both full and
compact widths. It has the same function and exactly the same initial weights
as ordinary SwiGLU; floating-point evaluation order can differ.

These acceptance criteria supersede the older charter's stronger aspiration of
beating full dense language loss by 1% and improving both resources at once.
Neither synthetic gains nor this one-seed development screen satisfies them.

## Hypotheses

**H1: cheap feature competition.** Three learned affine responses per hidden unit
may provide useful piecewise feature selection with only two matrix projections:

```text
z = reshape(W_up x + b, [hidden, 3])
y_i = max(z_i1, z_i2, z_i3)
FFN(x) = W_down y
```

At d=192, hidden=112, four layers, parameters are
`4 * (4*d*hidden + 3*hidden) = 345,408` (70.72% fewer than full SwiGLU).
Projection FLOPs are 688,128 per token across layers, excluding max/bias work.
The affine bias is part of the FFN change. All other model components remain
identical. Initialize weights with the same named normal initialization and
residual scaling as dense controls; biases start at zero.

This is a bias-enabled three-piece implementation of established
[Maxout Networks](https://arxiv.org/abs/1302.4389), not our invention. The research
question is its practical language-model size/quality/resource tradeoff in this
setting. Risks include too few independent output features, inactive branches,
nonsmooth selection, and missing the multiplicative interactions of SwiGLU.
The original paper's dropout results are not evidence for this decoder recipe.

Retrain `meanout` with identical projection shapes, weights, biases and budget,
replacing max by mean. This removes input-dependent selection, but also changes
activation statistics and removes the FFN's nonlinearity; it is a removal control,
not a clean proof that competition beats every nonlinear alternative. Dense
GELU/SiLU/SwiGLU controls remain necessary. After training, replace max by mean
with weights frozen and measure NLL and branch-selection utilization. A substantial
removal penalty is evidence of inference dependence, not automatically quality.

**H2: activation training effects.** Carry forward only PairFlux's curve-only mode
and its identical-width plain-SiLU control. On synthetic tasks the curve-trained
weights retained their loss when the curve was disabled at inference. Test whether
that observation transfers to language. Do not attribute a cross-run loss change
to the curve's inference expressivity without the frozen-weight intervention.
An affine-gain control and fresh paired seeds are required before promoting this
mechanism; this screen only decides whether that further experiment is justified.

**H3: structured reference.** Rerun single-pass BlockShuffle, the strongest local
synthetic recipe, on real text. Its previous runtime/activation-memory drawbacks
remain explicit; this is a prior-art reference rather than a new mechanism.

## Data and fixed recipe

Both datasets are now prepared from the repository's pinned recipes. BPE is trained
on training documents only, with vocabulary 4096. Exact normalized deduplication
prioritizes test, then validation, then training; near-duplicate exclusion is not
guaranteed. Preparation uses held-out document hashes for exclusion, not their
content for tokenizer fitting. No test loss is read during development.

* TinyStories v2: 49,900 training stories / 11,340,178 tokens; 1,000 validation
  stories / 200,105 tokens; 1,000 test stories / 229,270 tokens. The first 1,000
  official validation stories used historically are skipped.
* WikiText-2 raw v2: 624 training articles / 3,083,650 tokens; 60 validation articles
  / 322,802 tokens; 62 test articles / 368,043 tokens, using official source splits.

Backbone: width 192, four layers, six heads, full causal attention, context 256,
tied embeddings, RMSNorm and no dropout. Train each recipe for 2,000 updates with
batch eight, BF16 CUDA, AdamW, rate 0.0006, decay 0.1, seed 101, warmup 200 and
validation every 200 updates. This is 4,096,000 supervised next-token targets,
with identical randomly selected windows within each corpus. Evaluation covers
all complete nonoverlapping validation windows; the final incomplete tail is
discarded equally for every model. Record exact evaluated target counts.

Variants: full SwiGLU h512 (ordinary and fused), full GELU h768, narrow SwiGLU h152
(ordinary and fused), narrow GELU h228, narrow SiLU h224, curve-only h224,
BlockShuffle h1024/groups8, Maxout h112, meanout h112. Each receives the same budget.

## Failure criteria and next decisions

Reject a numerically invalid recipe or one exceeding 6 GiB PyTorch reserved VRAM.
Record all losses and resource costs. No candidate qualifies from one seed.
For this screen, advance only candidates meeting both language margins on both
corpora, or explicitly register an exploratory revision if a near miss suggests
a mechanism change. Do not silently lower the final acceptance thresholds.

Profile actual language batches with 20 warmup and 50 measured training updates
per model, three rounds with reversed order in round two. Report median/range,
allocated/reserved memory and supervised targets/s. These short profiles screen
hardware viability; confirmation needs longer runs, alternating order and timing
uncertainty before claiming a 1.2x gain.

Survivors receive equal three-seed/three-rate validation tuning with the relevant
full and compact controls. Freeze selection and 10,000-update confirmation budget
before five new seeds (4001, 4003, 4007, 4013, 4019) and held-out test evaluation.
Report all paired differences, mean margins and 95% intervals. A lower interval
bound above zero is required in addition to the 1% mean compact-control gain.
If no recipe survives, preserve these results and use the failure pattern to
register the next FFN mechanism. The full goal remains active.
