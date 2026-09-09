# Locked moderate-floor transfer to three existing checkpoints

Frozen after the width384 seed17 diagnostic and before these evaluations.
The .01 and .1 floors both pass the original local gates. Lock alpha=.1 because
it gives the stronger condition improvement among the passing treatments:
30134 to 26.53, with .01118% relative NLL degradation. Alpha1 loses 3.81% NLL
and is rejected. No further strengths will be tested in this transfer.

Use exactly the width192/layer4 h1024 native-gate 800-step BlockShuffle runs
with seeds17,29,43. Each gets four cases: unchanged, alpha0 reconstruction,
alpha.1 floor and its fixed seed1026+layer random-direction control. Use the
same FP64 transformation, per-block noise norm matching, fixed 32768 BF16
validation targets, CPU directional audit, and integrity/retention gates as the
[original plan](conditioning_plan.md). The noise direction is shared by rule;
these are three trained checkpoints, not three independently sampled noise
controls. Preserve original checkpoints and report every result.

This transfers a selected post-training intervention across size and seeds;
it is not independent three-seed replication of the width384 quality result.
All earlier training/model-selection history remains unchanged. A useful
transfer requires all three floors to preserve NLL within .1%, improve worst
down-projection condition by >=10x, and pass finite/integrity checks. Failure
in one checkpoint remains a failed all-three gate. No new training is earned
solely by this diagnostic, and no full-network stability claim follows.
