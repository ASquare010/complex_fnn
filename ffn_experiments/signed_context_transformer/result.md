# Signed-context FFN: results

## Current bounded refinement

The smaller, finer-grouped `signed-fine` advanced to paired-seed confirmation.
Seed101 full validation improves from 2.603935 to 2.602645 on TinyStories and
from 4.277294 to 4.276089 on WikiText. FFN weights fall from 2,490,368 to 2,359,296.
These small gains require repeat-seed confirmation; they are not evidence that
grouping alone caused the improvement because width also changed. All frozen
removals remain in `dump/compact-refinement-v1/`.
[Round results](../../docs/compact_refinement_result.md) and
[confirmation protocol](../../docs/compact_refinement_confirmation.md).

The fresh-seed gains did not hold: mean TinyStories NLL is 2.607223 versus parent
2.600803, with losses in all three pairs. WikiText is 4.271804 versus parent
4.271632, with one win in three pairs. Signed-Fine is not selected. The signed
parent remains a strong compact alternative and has slightly better WikiText
mean loss than Curve-Wide, but ranks behind it on the registered combined score.
[Complete paired analysis](../../docs/compact_refinement_confirmation_analysis.md).

## Historical stricter-gate decision

[Registered hypothesis](../../docs/signed_context_hypothesis.md).

Retire both fixed signed recipes: neither passes all resource, quality and mechanism comparisons on both corpora. Completed 20/20 quality runs.

tinystories/signed-linear: strongest full `full-swiglu-fused`, compact `narrow-swiglu-kernel`; full loss cost 2.58%, compact gain 0.09%; beats plain/closed/open True.

tinystories/signed-tanh: strongest full `full-swiglu-fused`, compact `narrow-swiglu-kernel`; full loss cost 3.10%, compact gain -0.41%; beats plain/closed/open True.

wikitext/signed-linear: strongest full `full-swiglu-fused`, compact `narrow-swiglu-kernel`; full loss cost 0.94%, compact gain 0.27%; beats plain/closed/open True.

wikitext/signed-tanh: strongest full `full-swiglu-fused`, compact `narrow-swiglu-kernel`; full loss cost 1.54%, compact gain -0.33%; beats plain/closed/open True.

## Computation and cost

576 detector features share 64 independently learned contexts in groups of nine.
Signed linear: `r_i=SiLU(u_i)*v_group(i)`. Signed tanh:
`r_i=SiLU(u_i)*tanh(v_group(i))`. Both let context affect output from the first
update. An independent D reads 576 features. U/C execute as one 640-row projection.
The linear recipe is SwiGLU with repeated gate rows; no new primitive is claimed.

2,490,368 FFN weights, 70.54% fewer than full SwiGLU. Projection FLOPs
4,980,736/token, excluding nonlinear/reduction work. Compact GELU hidden 608
matches the weight count exactly; compact SwiGLU hidden 408 has 0.658% more.
Gain controls have 0.093% more; plain SiLU hidden 576 is a smaller ordinary dense
control and is included when selecting the strongest compact language reference.

## Hardware and verification

Both recipes passed the [short hardware screen](../../docs/signed_context_result.md)
on both corpora against all three full controls. They use about 415.9 MiB
allocated. This is about 23% below full GELU/fused-activation SwiGLU and does not
establish sustained savings or 1.2x faster training. CPU FP64 equations,
gradients/finite differences and CUDA FP32/BF16 independent references passed,
including nine-detector gain controls. C's first-update gradient is zero for
closed gain, nonzero for open gain/signed in recorded checks. Full-model
causality/locality, counts and save/load passed. Production weights/counts and
outputs/gradients match prototypes exactly in CPU/CUDA checks, including nonzero
gain coefficients. Shared-Trainer CPU recovery is exact for all five modes.

## Language screen

Same width 512, four layers, 16 heads, context 256, vocabulary 4096, batch eight,
BF16. Ten variants per corpus, 2,000 updates / 4,096,000 supervised targets,
rate 0.0006, warmup 200, AdamW decay 0.1, seed 101 and identical sampled windows.
Full eligible nonoverlapping validation windows omit the same short tail. Test
loss remains unopened. Compare NLL within each corpus, whose tokenizer differs.
Both signed recipes must pass across both corpora as one common recipe; no
per-corpus switching. Closed/open gain differ only in coefficient initialization.
Signed/gain initial functions differ; identical initial logits are not claimed.

| Corpus | Variant | Validation NLL | Targets | Update / wall seconds | Allocated / reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| tinystories | full-swiglu-kernel | 2.544240 | 199,936 | 100.23 / 114.57 | 542.5 / 574.0 |
| tinystories | full-swiglu-fused | 2.538486 | 199,936 | 108.14 / 123.43 | 560.7 / 602.0 |
| tinystories | full-gelu | 2.592983 | 199,936 | 89.06 / 102.05 | 539.4 / 610.0 |
| tinystories | narrow-swiglu-kernel | 2.606365 | 199,936 | 84.02 / 94.97 | 407.6 / 450.0 |
| tinystories | narrow-gelu | 2.616391 | 199,936 | 80.39 / 90.99 | 408.9 / 468.0 |
| tinystories | plain-silu | 2.666190 | 199,936 | 78.77 / 89.06 | 406.7 / 468.0 |
| tinystories | closed-gain | 2.667904 | 199,936 | 87.36 / 98.18 | 415.9 / 468.0 |
| tinystories | open-gain | 2.619272 | 199,936 | 87.08 / 97.84 | 415.9 / 468.0 |
| tinystories | signed-linear | 2.603935 | 199,936 | 85.52 / 96.30 | 415.9 / 468.0 |
| tinystories | signed-tanh | 2.617161 | 199,936 | 83.56 / 94.29 | 415.9 / 468.0 |
| wikitext | full-swiglu-kernel | 4.239026 | 322,560 | 90.18 / 103.35 | 540.5 / 582.0 |
| wikitext | full-swiglu-fused | 4.237625 | 322,560 | 90.58 / 104.21 | 560.7 / 602.0 |
| wikitext | full-gelu | 4.269168 | 322,560 | 88.43 / 101.39 | 539.4 / 610.0 |
| wikitext | narrow-swiglu-kernel | 4.288883 | 322,560 | 83.81 / 94.49 | 407.6 / 450.0 |
| wikitext | narrow-gelu | 4.317237 | 322,560 | 82.54 / 93.34 | 408.9 / 468.0 |
| wikitext | plain-silu | 4.377303 | 322,560 | 80.56 / 91.16 | 406.7 / 468.0 |
| wikitext | closed-gain | 4.373279 | 322,560 | 85.83 / 96.98 | 415.9 / 468.0 |
| wikitext | open-gain | 4.309518 | 322,560 | 85.99 / 96.86 | 415.9 / 468.0 |
| wikitext | signed-linear | 4.277294 | 322,560 | 84.23 / 95.00 | 415.9 / 468.0 |
| wikitext | signed-tanh | 4.302946 | 322,560 | 85.78 / 96.63 | 415.9 / 468.0 |

Single-run update/wall times contain cold/runtime effects. Use repeated profiles
for the short resource decision. Reserved differs from allocated and device-wide
usage. Small-model results do not establish production-LLM accuracy or resources.

## Registered removals

* tinystories/plain-silu: trained 2.666190; half detectors 2.998141; train-mean FFN output 7.062632.
* tinystories/closed-gain: trained 2.667904; half detectors 3.003591; train-mean FFN output 7.271414. Coefficient zero 2.670071; mean |tanh(a)| 0.057786.
* tinystories/open-gain: trained 2.619272; half detectors 3.006451; train-mean FFN output 7.219232. Coefficient zero 2.875068; mean |tanh(a)| 0.766656.
* tinystories/signed-linear: trained 2.603935; half detectors 3.114100; train-mean FFN output 8.322718. Unit context 10.100070; train-mean context 4.728929; maximum injected mean error 0.00377.
* tinystories/signed-tanh: trained 2.617161; half detectors 3.026489; train-mean FFN output 8.060624. Unit context 8.562235; train-mean context 5.706572; maximum injected mean error 0.00125.
* wikitext/plain-silu: trained 4.377303; half detectors 4.720300; train-mean FFN output 9.882908.
* wikitext/closed-gain: trained 4.373279; half detectors 4.713050; train-mean FFN output 9.221490. Coefficient zero 4.376746; mean |tanh(a)| 0.071113.
* wikitext/open-gain: trained 4.309518; half detectors 4.722213; train-mean FFN output 9.164281. Coefficient zero 4.528624; mean |tanh(a)| 0.768441.
* wikitext/signed-linear: trained 4.277294; half detectors 4.788797; train-mean FFN output 10.341308. Unit context 8.730837; train-mean context 5.621343; maximum injected mean error 0.00755.
* wikitext/signed-tanh: trained 4.302946; half detectors 4.797485; train-mean FFN output 10.573190. Unit context 9.234330; train-mean context 6.472440; maximum injected mean error 0.00117.

Unit contexts use raw one for linear and +infinity preactivation for exact tanh
one, only at inference; resulting model outputs remain finite. Train means use
32 fixed training batches, seed 2027. Tanh means average post-tanh context and
are injected through inverse tanh; BF16 quantization errors are recorded. Gain
coefficients zero through the identical arithmetic backend. Half-detector
removal zeros even indices; constant FFN output uses its training mean.
All alter activation statistics and diagnose dependence, not semantic roles.
Signed versus gain also changes distribution/function family. A surviving signed
recipe still needs a positive-gate control before crediting sign reversal.

Originality is unverified. Development selection among two predeclared recipes
requires fresh confirmation. Equal tuning, repeated seeds, sustained resources
and an independent experiment remain required. Evidence:
`records/signed-context-v1-*.json`, `records/signed-context-language-v1-comparison.json`,
frozen run sources and raw interventions under `dump/signed-context-language-v1/`.
# Current refinement phase

The practical compression/VRAM milestone is recognized. This parent is now in the
[bounded refinement comparison](../../docs/compact_refinement_result.md), with
finer context gating as its first variation. The historical screen below retains
its original decision and thresholds.
