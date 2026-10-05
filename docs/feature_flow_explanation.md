# What the completed feature-flow experiment shows

All eleven variants on both corpora and all 22 removals completed. The fixed
recipe is retired. Full measurements are in the
[model report](retired_models/feature_flow_transformer/result.md); the
[terminal audit](../records/feature-flow-language-v1-terminal-audit.json)
verifies budgets, full validation and frozen source/recipe/checkpoint hashes.

WikiText's primary and all six dense references have now completed. Primary
loss 4.300017 costs 1.47% against full native SwiGLU 4.237625 and gains only
0.0045% against compact SwiGLU 4.300213. Both quality margins fail there too.
The variance-calibrated compact control is weaker on this corpus (4.309997),
so it is not selected as the comparison. Retrained one-step loss 4.296649 is
better than the primary. Diagonal loss 4.293004 and no-update loss 4.296419
are also better. Cross-neuron exchange therefore fails its retrained comparison
on both corpora, not just TinyStories.

At the WikiText primary checkpoint, zero M gives 4.303129, diagonal M gives
4.301878, removing even internal responses gives 4.301398, and using one step
gives 4.300059. Training-mean FFN replacement gives 9.804952. The small
one-step change agrees with TinyStories's weak sensitivity to extra recurrent
steps; the full FFN remains essential in both. This is one development seed,
not a general claim that hidden feature interactions cannot help. The
[partial WikiText audit](../records/feature-flow-v1-wikitext-primary-decision.json)
verifies completed budgets, full validation, dataset identity, frozen source
and recipe hashes, and absence of test-evaluation records.

P learns 512 combinations of the residual features. Within each 32-coordinate
group, M mixes bounded nonlinear responses at three internal steps, then the
final scalar response enters the full Q projection. This can create nonlinear
compositions, but it adds no independent input detectors. Its feature bank
still begins from the same 512 projected values. Additional steps and responses
do not guarantee additional useful information.

The primary TinyStories loss is 2.606496. Full native SwiGLU reaches
2.538486; the strongest equally sized ordinary dense control, variance-calibrated
compact SwiGLU, reaches 2.614896. The primary costs 2.68% against full and gains
only 0.32% against compact, failing both required quality margins.

Retrained one-step loss is 2.607807; diagonal three-step loss is 2.604415; no-update
loss is 2.604789. Thus the primary beats one step slightly but loses to diagonal
and no-update controls on this seed/corpus. This is not evidence that cross-neuron
exchange helps here. Controls have honestly recorded smaller counts where
appropriate, and no alternative control is substituted as a post-hoc candidate.
Ordinary cached execution reaches 2.606072 with the same mathematical function;
the small difference from fused execution does not establish superiority on one
seed. CUDA and ordinary group operations accumulate in different FP32 orders.
WikiText cached loss is 4.291654 versus fused 4.300017. This training-trajectory
difference does not make the cached control a qualified replacement: its loss
still misses the full-dense margin, and its 617.7 MiB allocation fails the memory
target. It is not substituted for the registered primary after seeing results.

At the trained primary checkpoint, zero M gives 2.609012, retaining only its
diagonal gives 2.608527, removing even internal psi responses gives 2.607348,
and using one step gives 2.606477. Original calibration and residual h are
retained. These small changes show little frozen-checkpoint loss sensitivity
to this internal computation in this allocation; they do not prove that M
encodes no information. Mean-FFN replacement gives 8.603021, so the overall
FFN still matters greatly. Frozen ablations change statistics and cannot assign
semantic roles or isolate training dynamics by themselves.

The [analytic expansion audit](feature_flow_expansion.md) explains a relevant
initialization property: at M=0, one and three steps have identical functions
and first gradients; extra depth first differs at second order in M. The tiny
synthetic FP64 audit verified that expansion, independently of all trained
weights. This is a mathematical explanation, not an inference about all learned
trajectories. The retrained controls and removals provide the actual evidence.

The short hardware screen passed at 401.7 MiB allocated, about 26–28% less than
full dense, using exact discrete recomputation and fused GPU execution. That
achievement concerns execution/storage of a compact function; it does not
establish richer inference computation. No causal training-dynamics advantage,
language qualification, repeated-seed gain or originality claim has been shown.
Retire this fixed recipe because both corpora's quality gates and its
cross-neuron comparison fail. All registered allocations/removals are complete;
conditional tuning/confirmation will not run. The next hypothesis remains to
be selected; [readout reuse notes](ffn_readout_prior_art.md) document known
mechanisms and representational constraints without claiming a new architecture.
