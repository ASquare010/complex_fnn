# H041: frozen independent-seed WikiText replication

Written after H039 passes, before seed29 or seed43 trains. Seed17 affine
BlockShuffle reached NLL4.749982 versus unmodified BlockShuffle4.832481,
full GELU4.863755, full SwiGLU4.880372 and calibrated narrow4.894435.
All short- and longer-budget gates pass on that seed. This is still selected,
single-seed evidence; the purpose here is to challenge it with independent runs.

## Fixed cohort

Retain exactly the five H039 architectures, selected rates, initialization,
optimizer groups, BF16 native execution, checkpoint policy,800steps, B16,
context128 and all322,688 validation targets. Change only seeds to29and43,
including initialization and the paired data-sampling seed=10000+seed.
No learning-rate or shape selection on these seeds. Reuse the five verified
seed17/800 checkpoints; train ten additional trials,16,384,000 sampled tokens.
The same seed shares the same token stream across all five recipes.

Use H039 order (full SwiGLU,narrow,BlockShuffle,full GELU,affine) rotated by one
position at seed29 and two positions at seed43. One fresh GPU worker at a time,
2400-second deadline each. Complete both seeds even if one quality result fails;
stop only on infrastructure/numerical failure, retaining all artifacts. A retry
requires an explicit recorded continuation, with no overwritten run or hidden
extra compute. Never select seeds or intermediate checkpoints after seeing loss.

Before training, verify current shared computation hashes against H039 archives,
all prior source/checkpoint/data hashes, final parameter counts, and the frozen
plan/config hashes. Check that every new configuration differs from H039 only
in seed. At tiny CPU scale, verify common non-FFN initial tensors match across
all five recipes for each seed, and affine vs unmodified BlockShuffle logits,
loss and common gradients match exactly at zero correction. All actual runs
must retain finite histories/diagnostics, and affine/base initial full-validation
NLL must agree within1e-7 within each seed. The five final sampling-RNG states
must match within each seed. Distinct seeds must have distinct RNG states.

## Fixed acceptance and statistics

Apply every H039 gate separately to seeds17,29,43: >=70% FFN parameter reduction,
<=1% relative NLL degradation against BOTH full controls, beat calibrated narrow,
>=0.2% improvement over unmodified BlockShuffle, and allocated training peak
<=1.1timesBOTHfull references. Replication passes only if all three seeds pass
all gates. Also report mean NLL, sample standard deviation, every paired NLL
difference and range, mean clipping, memory and descriptive serial timing.

Report a two-sided95% Student-t interval for the mean paired NLL difference,
using df=2 and tcrit=4.302652729911275. It assumes approximately normal paired
seed differences; three seeds cannot verify that assumption. Report an exact
one-sided paired sign-test p-value as a small-sample cross-check. Even three wins
out of three give p=0.125, so do not describe three consistent wins alone as a
conventional95% significance result. The seed17 selection and architecture-search
history also limit confirmatory interpretation. No population-level SOTA claim.

A pass earns TinyStories transfer of the affine recipe and a gain-only/bias-only
training ablation, then stronger convergence/scale/throughput and novelty checks.
A failed gate stays a failed replication; the earlier single-seed win remains
historical. Neither outcome completes the broad original research goal.
