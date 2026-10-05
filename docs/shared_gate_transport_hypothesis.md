# Hypothesis v2: layer-specific gate transport in a shared FFN bank

Registered after the completed v1 hardware screen and before v2 implementation,
profiling or language training. The acceptance targets remain unchanged.

## What v1 taught us

Sharing a wide input/gate bank saves weights, but its activations still occupy
memory at every layer. At width 512, hidden 816, the largest allocated peak was
452.8 MiB: 20.2% below full fused SwiGLU, but only 17.0% below full GELU.
Neither smaller registered width passed. All v1 fixed recipes are retired;
their language quality is unknown. Evidence is preserved in the model's result.md.

## Proposed computation

Use a smaller shared detector bank and give each layer an inexpensive way to
recombine its features when forming gates. For each normalized token x at layer l:

```text
u, g = split(W_shared x)
g'_i = g_i + tanh(a_l,i) * u_(i-1 mod h)
y_l = D_l (SiLU(u) * g')
```

The shared bank learns features; independent output maps read them differently.
The learned coefficient vector a_l allows a feature from the preceding hidden
channel to affect another feature's gate. This is a single fixed cyclic matching,
not token attention, a search over neighbors, or an additional Transformer pass.
The detector weights can learn how to populate the channel coordinates.

The added term is a cross-feature product `SiLU(u_i) * u_(i-1)`. It may let layers
reuse the same detectors with different conjunctions. This is a hypothesis about
useful capacity, not a guarantee of more information or greater expressivity than
an unrestricted dense FFN. The cyclic ordering is an architectural constraint.
Initialize a_l to zero so the model starts as the plain shared-bank control;
nonzero gradients can learn whether to use the added interaction. Bound its
coefficient with tanh, and preserve ordinary SwiGLU gating without saturation.

## Fixed size and controls

Width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight, BF16.
Full SwiGLU hidden 1376; full GELU hidden 2064. Shared hidden 608.
The plain shared bank has 1,867,776 FFN weights; transport adds 2,432, totaling
1,870,208: 77.88% fewer than full SwiGLU's 8,454,144. Projection FLOPs across
four layers are 7,471,104/token, 55.81% fewer than full SwiGLU. Record the extra
pointwise arithmetic and its actual runtime; projection FLOPs do not include it.

Controls: plain shared hidden 608; untied hidden 608 initialized with identical
detectors; compact fused SwiGLU hidden 304 (1,867,776 weights); compact GELU
hidden 456 (same count); both full dense controls. The compact controls are
0.13% smaller than transport; report this difference rather than claim exact
equality. Attention, embeddings, norms, tokenizer and training budget are fixed.

## Failure criteria and experiment sequence

First profile transport, plain shared, full fused SwiGLU and full GELU on both
corpora: three alternating-order rounds, 20 warmup and 100 measured updates.
Use largest allocated peak across rounds and median throughput. Require at least
20% lower allocated memory OR 1.2x throughput against both full controls, with
at most 5% deterioration in the other resource and less than 6 GiB reserved.
If transport fails this resource screen, retire the recipe before language
training. If plain shared passes, its separately registered capacity hypothesis
can proceed as an established-method control. Do not tune widths after quality.

For hardware survivors, run the seven variants above on both corpora, seed 101,
2,000 updates, learning rate 0.0006, warmup 200, AdamW weight decay 0.1. No test
loss is used. Require within 1% of the strongest full dense validation NLL and
at least 1% improvement over the strongest compact dense validation NLL on both
corpora. Transport must also improve over plain sharing to motivate its added
mechanism; otherwise retire the added mechanism even if compression is useful.

After training transport, set its coefficients to zero without retraining and
evaluate the entire eligible validation split. Compare with the trained plain
shared control: this distinguishes inference dependence from a possible training
effect. Also remove even-index hidden features in candidate and controls, and
replace their FFN outputs by means estimated on 32 fixed training batches only.
These interventions diagnose dependence, not feature semantics. Record learned
coefficient distributions and paired losses. Survivors still require equal
rate/seed tuning, longer resource measurements, fresh seeds and independent
confirmation before any qualified claim.

## Prior art and limits of originality

[One Wide Feedforward Is All You Need](https://aclanthology.org/2023.wmt-1.98/)
and [ALBERT](https://arxiv.org/abs/1909.11942) precede depth sharing.
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) establishes
multiplicative gating. [Circulant projections](https://arxiv.org/abs/1502.03436)
and [diagonal-circulant networks](https://arxiv.org/abs/1901.10255) precede cheap
structured channel mixing. Our previous PairFlux screen already tried bounded
feature exchange and failed. This proposal places layer-specific cross-feature
products inside gates of a shared bank, rather than exchanging activated values
of independent banks. That difference motivates a test, not a novelty claim.
The combination's originality is unverified; a literature search cannot prove
that an idea has never been discovered.
