# Conditional confirmation for the basis-readout FFN

Registered before any language training in this iteration. Hardware fixed-basis
passed, learned-basis failed; only fixed-basis may advance. No tuning or test
evaluation is authorized by a failed development recipe. The acceptance contract
is unchanged: >=70% FFN weight reduction, <=1% NLL cost to strongest full and
>=1% NLL gain over strongest ordinary compact on both corpora, and the resource
gate without more than 5% deterioration in the other resource.

Only if the completed registered two-corpus development screen passes both
dense margins and beats the tied-readout control on both corpora: tune fixed-basis,
all three full and all three compact dense references equally. Use seeds
101/103/107 and rates .0003/.0006/.0012, 2000 updates, warmup 200, all remaining
settings/data fixed. Select one learning rate per architecture per corpus by
mean validation NLL, with the smaller rate winning exact ties. Select strongest
full/compact architectures by mean tuned NLL, then freeze those choices before
the fresh-seed comparison. Keep every dense run; no selective budgets.

Fresh confirmation: seeds 4001/4003/4007/4013/4019, 10000 updates / 20.48M targets,
warmup 1000, frozen tuning choices, same paired windows and common backbone.
Retrain tied and cached execution controls using the candidate's selected rate;
those comparisons address mechanism, not an equal-tuned superiority claim.
Mean compact gain must be >=1% and its paired 95% t interval lower bound >0;
mean full cost must be <=1%, with both thresholds on each corpus. Report all
five paired values, intervals and worst seeds. Test is scored only after choices
are frozen and validation gates pass, once for every registered confirmation
run; no changing the architecture/rate after test scores.

Repeat resource measurements in a fresh standalone process, 200 warmup/1000
timed updates, three alternating-order rounds, record clock/power context,
temperature if available, actual precision/TF32, allocated/reserved peaks and
median throughput. Candidate must pass against all three full references on
both corpora under the same resource formula. Its development resource margin
is narrow, so the short screen is insufficient for a practical qualification.

At frozen final checkpoints repeat all registered basis/readout/shape/mean
removals, retaining calibration. Retrained tied/fixed/cached controls and frozen
interventions have different purposes; penalties alone do not identify semantic
roles. A fresh process/seed confirmation is independent of development choices,
not replication by an independent laboratory. Novelty needs a fresh comparison
with closest primary work before any discovery claim.
