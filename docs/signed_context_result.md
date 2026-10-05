# Signed context FFN: hardware result

[Registered hypothesis](signed_context_hypothesis.md).

Retire both fixed recipes after the completed 20-run language screen.
Signed linear's TinyStories/WikiText NLL is 2.603935/4.277294; signed tanh's
is 2.617161/4.302946. Signed linear's gain over the strongest compact dense
control is only 0.09%/0.27%, below the required 1%; its full-baseline loss cost
is 2.58%/0.94%. Signed tanh misses both margins on both corpora.

Subsequent production equivalence passed for all five modes: CPU/CUDA outputs
and gradients match prototypes exactly in the recorded checks, including nonzero
gain coefficients. Shared-Trainer CPU recovery is exact for all five modes.
The [production model report](../ffn_experiments/signed_context_transformer/result.md)
records the completed ten-variant, two-corpus language screen and frozen-weight
removals. Neither recipe establishes the required language advantage.

576 learned detectors share 64 independently learned contexts in groups of nine.
`r_i = SiLU(u_i)*v_group(i)` or `SiLU(u_i)*tanh(v_group(i))`; D projects the
576 features back to the residual stream. This is a tied-gate GLU hypothesis,
not a verified new category of neural computation. Both signed recipes have
2,490,368 FFN weights, 70.54% fewer than full SwiGLU. Projection FLOPs are
4,980,736/token, excluding nonlinear and gradient-reduction work.

## Short hardware screen

| Corpus | Recipe | Allocated MiB | Cut vs native / kernel SwiGLU / GELU | Throughput vs native / kernel / GELU | Screen |
| --- | --- | ---: | ---: | ---: | --- |
| tinystories | signed-linear | 415.9 | 25.82% / 23.04% / 22.89% | 1.007x / 1.098x / 0.988x | pass |
| tinystories | signed-tanh | 415.9 | 25.82% / 23.04% / 22.89% | 1.006x / 1.097x / 0.987x | pass |
| wikitext | signed-linear | 415.9 | 25.82% / 23.04% / 22.89% | 1.023x / 1.025x / 0.981x | pass |
| wikitext | signed-tanh | 415.9 | 25.82% / 23.04% / 22.89% | 1.012x / 1.015x / 0.971x | pass |

Same width 512, four layers, 16 heads, vocabulary 4096, context 256, batch eight,
BF16 and four CPU threads. RTX 4070 Laptop GPU. Three alternating-order rounds,
20 warmup plus 100 measured updates per variant/corpus. Maximum allocated peak
and median throughput; cold warmup/compilation separately recorded. All updates
finite and below 6 GiB reserved. This screen does not establish sustained memory
savings or 1.2x faster training. Report each signed recipe's failure independently.

## Numerical preparation

CPU FP64 independent group-selection equations/gradients and finite differences
passed. CUDA FP32/BF16 forward and gradient references passed for both signed
recipes and open/closed gain controls with groups of nine. Full-model causality,
token locality, save/load, counts and exact CPU optimizer recovery passed.
Closed gain gives C zero gradient on the first update; open gain and both signed
recipes give nonzero gradients in the recorded reference check. This does not
show that immediate learning improves quality. Production equivalence and
shared-Trainer CPU resume were subsequently verified before language training;
see `records/signed-context-v1-integration-checks.json`.

Additional CPU removal checks passed: unit context matches the unary path
exactly in FP64/BF16 activation tests, and the FP64 complete model matches an
independently computed unary FFN reference. Tanh's unit gate uses +infinity only
in the context preactivation and produces finite outputs. Forward hooks leave
all recorded weights unchanged. These implementation checks do not prove a
trained-model inference benefit or GPU intervention results. Evidence:
`records/signed-context-v1-removal-checks.json`.

Language validation is scored; test loss remains unopened. Its predecessor also
failed both corpora. Originality remains unverified. Language evidence is in
`records/signed-context-language-v1-comparison.json`. Hardware evidence:
`records/signed-context-v1-checks.json`, `records/signed-context-v1-profiles.json`,
and the frozen source/protocol under `dump/signed-context-v1/`.
