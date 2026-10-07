# Checkpoint storage cleanup — 2026-10-07

The user explicitly requested removal of old model weights and unnecessary files
to recover SSD space. This request supersedes earlier blanket instructions to
retain every historical checkpoint, for the exact paths in the deletion receipt.

Deleted **1,682 files (89,614,288,805 bytes / 83.46 GiB)**: 1,279 obsolete,
duplicate, superseded or unranked weight files and 403 rebuildable cache files.
The drive had **112.72 GiB free** after cleanup, versus approximately 29.26 GiB
before it. No new training was performed during cleanup.

## Retained model endpoints

**54 selected model/encoder checkpoint files (7.57 GiB)** remain at their original
paths, with their original full state and bytes:

- The 10,000-update Branch Sigmoid winner, Dynamic Filter, dense, original
  Curve-Wide, equal-parameter expanded Curve-Wide, Branch Softmax and signed hybrid.
- Evolution's Gate-16 and positive partial-wave memory/quality comparisons.
- Evidence Balance and Attention Energy, representing the closest lower-memory
  result and the best single-corpus score in that batch.
- The six selected historical 8,000-update Curve-Wide/full-dense/compact-dense
  endpoints across TinyStories and WikiText.
- The two-seed squared-ReLU/shifted-squared-ReLU confirmations and dense controls;
  the larger-model squared-ReLU scaling pair; the two-seed signed-hybrid
  confirmations and their controls; and the two Dynamic Filter confirmations.
- Eight final Step 3 pilot/comparison endpoints, including the rolling-memory
  notebook default and matched dense checkpoints.
- Selected 8-, 16- and 32-token compressor checkpoints, the original selected
  8-token source-migration predecessor, and all seven frozen encoder exports.

All prepared datasets, source/configuration files, source snapshots, notebooks,
the configured GPU environment, Git history, result records, audits, logs and
the historical source archive remain. Small diagnostic fixtures and user memory
examples were not treated as obsolete model weights.

## What deletion means for old results

Historical recorded scores and audits were not rewritten. Rankings still refer
to their completed full-validation measurements, but **many retired checkpoints
are no longer available for inference, replay or resumption**. Do not recreate
them, retrain controls or restore old research automatically to repair a link.
Read the retention list before planning a checkpoint-dependent diagnostic.

The two unverified partial Branch refinement weights were deliberately removed;
their source snapshots, logs, previous hashes and failed-attempt records remain.
They were never ranked and cannot be resumed. The selected three 10,000-update
controls required by the pending refinement comparison remain intact.

## Verification and local receipts

Deletion used an exact reviewed file list, absolute workspace-bound path checks,
reparse-point checks and live-job checks. No tracked source file was deleted.
All **929 protected checkpoint/data/record files** matched their pre-cleanup
SHA-256 hashes after deletion.

- [Complete retention and deletion plan](../dump/cleanup-2026-10-07/plan.json)
- [Per-file deletion receipt](../dump/cleanup-2026-10-07/deleted.jsonl)
- [Protected-file hashes](../dump/cleanup-2026-10-07/protected-before.json)
- [Completion receipt](../dump/cleanup-2026-10-07/result.json)

These local receipts and all experiment records remain ignored by Git.
