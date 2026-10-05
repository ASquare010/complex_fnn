# Conditional latent FFN tuning and confirmation

Registered while the seed-101 development campaign is live, before the wide
candidate's first completed validation result. This specifies the follow-up
already required by [the acceptance contract](language_screen_v1.md), rather
than changing the current screen. It launches only if the fixed wide recipe
passes all registered comparisons on both corpora. Otherwise retire that recipe.

## Equal validation tuning

Keep the width-512, four-layer backbone, both frozen datasets and all rank/width
allocations fixed. Tune eight variants: the three full dense controls, both
ordinary compact dense controls, wide recompute, thin recompute and thin with
the registered initialization calibration. The cached wide variant remains an
execution-equivalence control; it does not need a separate hyperparameter search.

Use seeds 101, 103, 107 and rates 0.0003, 0.0006, 0.0012, all at 2,000 updates,
200 warmup updates, batch eight, decay 0.1 and BF16. Each variant receives exactly
the same nine recipes per corpus. Reuse an existing matching run only if source,
configuration, dataset, complete token budget and final validation match the
frozen tuning protocol. Report every reused and newly executed run.

Choose each variant's rate separately within each corpus by mean final full
validation NLL over the three paired seeds. Break exact ties by the smaller
rate. Architecture, input/output ranks and hidden width remain a common recipe
across corpora; only the equally tuned optimization rate may differ. Report that
distinction. No best-checkpoint, seed or intermediate-step selection.

Before confirmation, freeze every variant's selected rate, the common backbone,
data fingerprints, source snapshot, evaluation procedure and confirmation
budget. Do not use test loss to select any of these. Development tuning alone
does not satisfy reliability.

## Fresh independent confirmation experiment

Train all eight variants from scratch at seeds 4001, 4003, 4007, 4013, 4019 on
each corpus for 10,000 updates / 20,480,000 supervised targets. Use the frozen
selected rates and 1,000 warmup updates for every variant; the other optimizer
settings and backbone stay fixed. Start new run identities and model/optimizer
states; do not continue or fine-tune development checkpoints. The longer budget
is an independent registered experiment, with fresh seeds and previously unopened
test evaluation. It is not an independent lab replication or proof of scale.

Alternate/reverse variant execution order across seeds, with no concurrent GPU
work. Score complete eligible validation and test windows at the final update.
Open test evaluation only after all recipes are frozen. Report all validation
and test results, including negative ones, and account for any interruption.

For each corpus and split, select the strongest full and strongest ordinary
compact control by their mean NLL across the five fresh seeds. Report paired
per-seed relative NLL differences against each control, plus the selected
strongest references. Require at most 1% mean relative NLL cost against the
strongest full and at least 1% mean gain against the strongest compact on both
corpora. A 95% paired interval for the compact gain must have lower bound above
zero, as already required by the acceptance contract. Use the Student-t interval
on five paired differences with four degrees of freedom (critical value
2.7764451052); expose every difference and interval, including full-baseline cost.
Apply the stated quality margins to both validation and held-out test. Report
selection of the strongest reference and the small five-seed uncertainty; do not
present it as a simultaneous confidence guarantee over all tested hypotheses.

The wide candidate must also beat both thin controls by mean NLL on both corpora,
with paired differences reported. An inference removal penalty does not replace
these retrained allocation comparisons. If an equal tuning or fresh-seed gate
fails, retire the fixed recipe rather than changing margins or choosing seeds.

## Sustained resources and explanation

Record every confirmation run's peak allocated/reserved training memory and
training-only seconds, separate from evaluation/checkpoint wall time. After
confirmation use three alternating-order rounds per corpus, 200 warmup and
1,000 timed training updates, candidate versus all three full dense controls,
same frozen architecture and selected rates. Include cold warmup, mean/range
throughput and uncertainty, not just the fastest measurement. Resource acceptance
remains >=20% lower allocated VRAM OR >=1.2x throughput, with <=5% deterioration
in the other resource; reserved memory must stay below 6 GiB. Confirm savings
against the strongest full quality reference and report all full comparisons.
Keep recomputation FLOPs explicit. Short-screen savings alone are insufficient.

At each fresh candidate checkpoint repeat same-backend unit expanded gates,
even nonlinear detector removal before B, and training-mean FFN output replacement
from 32 batches at mean-sampler seed 2027. Repeat corresponding detector/output
removals for the thin and dense controls. Report per-layer gate/feature statistics
and loss deltas. Compare retrained allocation controls and frozen-weight removal
effects: the former concerns training/architecture, the latter inference
dependence under changed statistics. Neither alone identifies semantic roles.
Only credit a richer inference mechanism when the controlled comparisons support
it; otherwise state that the evidence supports a practical training/parameterization
benefit with an unresolved mechanistic explanation.

Preserve a result.md for tuning and confirmation, compact records, every run's
configuration/source/data provenance, and the decision to continue or retire.
Review closest low-rank FFN, bottleneck GLU and recomputation work again before
claiming originality. Established factorization and activation recomputation
remain prior art even if this allocation meets the practical gates.
