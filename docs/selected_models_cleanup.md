# Selected-model cleanup — 2026-10-07

The user requested removal of legacy code, configs and integrations, retaining
only Branch Sigmoid for Step 1 and the winning span-32 context encoder for Step 2.
This supersedes the earlier active Curve-Wide/dense/Step 3 and shortlist scope.

Removed 439 obsolete source/config/notebook/cache files from the active tree.
Another36 historical planning/result documents were removed from active docs
after their snapshot hashes were verified. The research charter, frozen
leaderboard, Step2 evidence and storage receipts remain accessible.
The FFN experiment implementations, all sweep controllers, dense/Curve-Wide LMs,
rolling-memory/readers/context comparisons, dataset experiment machinery and
root FFN shortlist are no longer active. `src/config` contains exactly:

- `branch_sigmoid.json`: selected architecture and explicit future training recipe.
- `branch_data_identity.json`: independent decoded identities for retained data.
- `position_compressor.json`: selected span-32 encoder and reconstruction endpoint.

Branch Sigmoid is standalone under `src/models/branch_sigmoid`. Its equation,
calibration, weight names, backbone and checkpoint are preserved. The trained
Step 2 encoder requires the original curve FFN internally; that component now
lives under `position_compressor/curve.py`, with no standalone curve LM. Its
architecture-hash compatibility entry covers the exact import-only relocation.

The CLI and two notebooks target only the selected models. No automatic model
registry or Step 3 integration remains. Shared attention/normalization utilities,
verified data loading and selected training/evaluation code remain. Reconstruction
edit distance uses a small dependency-free implementation to avoid an optional
missing package blocking imports. No environment or dependency sync occurred.

The pre-cleanup snapshot covers379 source/document files, all hash-verified in
`dump/selected-cleanup-20261007/before.zip`. Deletion used an explicit file plan,
absolute workspace bounds, reparse-point checks and archived-hash verification.
See [deletion receipt](../dump/selected-cleanup-20261007/deleted.json).

Original winner logits and encoder vectors were saved before cleanup. Regression
verification checks bit-exact equality, checkpoint hashes, parameter counts,
reconstruction and retained training imports. See
[verification log](../dump/selected-cleanup-20261007/verification.log).
All three regression checks passed. Both new inference notebooks executed
successfully. Step1 token targets were exactly preserved after removing unused
history tensors. [Completion receipt](../dump/selected-cleanup-20261007/result.json).

Datasets, checkpoints, result records, audits and historical source snapshots
were not removed by this code cleanup. Old documentation and leaderboards describe
historical runs and may reference retired active paths; use archived sources for
historical replay, not the new code. The research charter remains unchanged.
No training was run. Previous storage-cleanup deletions remain in force.
