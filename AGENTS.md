# RoPE compressor complete (2026-10-07)

All 7000 seed17 updates finished. Full final validation replay: 1000/1000 exact,
NLL0.00227359, token accuracy100%. Encoder7342336 / full10488832 parameters;
379.44/470MiB peak allocated/reserved. Checkpoint/encoder/source hashes and record
verified. Completion receipt dump/rope-compressor-launch/completion.json.
No new training authorized. One seed, previously used validation; no matched
RoPE ablation or superiority claim. Sequence shortening is not byte compression.
Old compressor weights were deleted as requested; Step 1 weights remain.

# Authorized RoPE compressor run (2026-10-07)

Latest user explicitly authorized new encoder training and deleting old compressor
weights. This supersedes prior no-training and compressor-weight retention rules.
One fresh seed17 Branch Sigmoid + RoPE run: 7,000 total updates in five stages,
using the existing 50k paragraph data; no extra experiments. Sources frozen after
launch; do not retry native failures automatically. See docs/branch_rope_training.md.
Step 1 Branch Sigmoid weights, datasets, records and source archives are retained.

# Branch compressor migration (2026-10-07)

User explicitly replaced Step 2 CurveFFN with Branch Sigmoid in encoder AND decoder.
This supersedes historical internal-curve retention and compatible encoder loading.
Both selected packages remain active. Step 1 weights and outputs remain unchanged.
Step 2 is now an UNTRAINED architecture revision: old CurveFFN weights are preserved
but deliberately rejected, and historical reconstruction scores do not apply.
No training is authorized by this code migration. See docs/branch_compressor_migration.md.

Active packages: src/models/branch_sigmoid and src/models/position_compressor.
Keep only their configs plus branch_data_identity.json in src/config. No old
experiment registry, sweeps, alternative LMs, Step 3 integrations or custom kernels.
No training is authorized by cleanup; future training needs an explicit budget.
Use ordinary PyTorch and uv run --no-sync. Do not replace/sync the CUDA environment.
Do not blindly retry native crashes. Earlier DLL/runtime failures remain unexplained.

Preserve selected checkpoints, datasets, historical records/audits and source
snapshots. Do not restore retired source into the active tree to repair old paths.
Original sources/recipes are required for historical optimizer resumption. The
current selected Branch loader verifies the exact winner hash for inference.
Step 2 exports require current architecture/FFN source hashes and Branch v2 format.
Do not modify docs/research_goal.md. Research targets remain unachieved; cleanup
is not a new quality result. Step 2 reconstruction is not next-token LM evidence.

Branch Sigmoid: web3.820069/chat2.411506,8,654,208 total/4,718,592 FFN parameters,
10,000 updates seed461,454.54/480MiB allocated/reserved. All latest branch-next
variants lose to it on both corpora. One-seed/selection limitations still apply.
Historical CurveFFN span32 encoder:4,798,720 parameters; full model7,292,928; bounded
reconstruction success, no universal losslessness or downstream reasoning claim.

No training workers or monitoring automations remain. Archived active-tree snapshot
and deletion receipt: dump/selected-cleanup-20261007/. See README.md and
 docs/selected_models_cleanup.md. Historical evidence remains under dump/records;
keep records ignored/untracked, never force-add. Historical leaderboards are frozen
references; do not silently recompute old runs using migrated sources. Export any
new authorized runs and document their distinct source provenance/results.
