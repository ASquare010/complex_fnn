# Conditional feature-flow confirmation

Registered before the one-seed language screen starts. The original
[FFN hypothesis](feature_flow_hypothesis.md) and goal thresholds remain binding.
This document defines work only for a primary that survives all resource,
language and one-step/diagonal comparisons on both corpora. A failed recipe
is retired; its conditional tuning/confirmation will not run.

The development screen is seed 101, eleven variants per corpus, 2,000 updates
and 4,096,000 targets, with the existing fixed backbone/data/optimizer/BF16
settings. Full three, ordinary compact three (including variance calibration),
primary and four controls receive identical sampled windows. Test loss remains
unopened. The passing short hardware screen is not sustained confirmation.

If it survives, equally tune the primary and each of the three full/three
ordinary compact references using learning rates .0003/.0006/.0012 at seed 101,
same 2,000-update budget and validation target coverage. Preserve existing
.0006 runs and execute missing allocations only. Select each architecture's
rate by validation loss, independently per corpus, then compare the strongest
full/compact references and freeze choices. No architectural or removal changes.

Repeat paired seeds 211/307/401 with frozen rates and the same 2,000-update
budget, evaluating full validation. All seven architectures get the same
budgets/windows within each seed. Require both quality margins per seed on both
corpora, with mean losses and paired variation also reported. No replacing a
failed seed or selecting a seed. Record uncertainty without claiming this small
set proves a universal improvement.

An independent fresh confirmation then uses seed 509, frozen rates, 8,000
updates/16,384,000 targets and warmup 800, keeping all other backbone/data
settings unchanged. Run the primary and all six ordinary references, full
validation, then open test exactly once for those frozen completed allocations.
Require the original full/compact margins on validation and test for both
corpora. No further selection or tuning on test results. Preserve disconfirming
results and retire a failure.

Confirm resources separately with three alternating-order whole-model rounds
per corpus, 100 warmup/1,000 measured training updates for the primary and all
three full references. Include casts, copies, reductions and optimizer work;
report maximum allocated/reserved training memory and median throughput plus
every round's timing. Retain >=20% memory reduction or >=1.2x throughput with
<=5% deterioration of the other metric, against every full reference. Repeat on
the available RTX 4070 Laptop; do not claim independent hardware generality.

Repeat the registered frozen removals on confirmation checkpoints and retain
retrained one-step/diagonal/no-update controls at their development settings for
the paired seeds and longer confirmation allocation. Compare primary with
one-step and diagonal on both corpora. Frozen interventions change distributions
and do not establish semantic roles; retrained comparisons and longer training
help distinguish training dynamics from persistent inference computation.
Verify closest prior implementations before claiming originality. Only all of
these completed requirements can qualify the replacement.
