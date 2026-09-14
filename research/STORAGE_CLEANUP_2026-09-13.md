# Storage cleanup — 2026-09-13

Removed 6,880 obsolete, untracked `.pt` experiment tensors: **152.83 GiB**.
Free space increased from about19.9 GiB to172.7 GiB. No research was resumed.

Preserved all248 tensors in the protected result sets (about17.99 GiB), including
H156-H160 and the H117 seed initializations. Also preserved all three selected
models under `src/experimental/checkpoints`, all H160 completed/partial runs,
and the interrupted native run's step200 checkpoint. All protected result tensor
sizes/timestamps are unchanged. Selected models, the resume checkpoint and all six
required initialization checkpoints passed their saved SHA256 checks.

Source, reports, metrics, configurations, receipts, datasets under `data`, the UV
environment and Git history were retained. Only historical `.pt` files outside
the protected studies/initializations were removed; no tracked files were deleted.
These included old model/optimizer checkpoints, gradient copies and generated
experiment tensor datasets. Older raw-tensor audits now require regeneration;
missing files listed here are intentional cleanup, not unexplained corruption.
Historical receipts retain their original hashes and have not been rewritten.

- `storage_cleanup_2026-09-13.json`: pre-deletion inventory and protected set.
- `storage_cleanup_2026-09-13_deleted.jsonl`: successful deletion journal.
- `storage_cleanup_2026-09-13_result.json`: totals and verification record.

The paused H160 resume instructions remain in `PAUSED_HANDOFF.md`. The latest
five study folders and seed initialization checkpoints were kept in full.
