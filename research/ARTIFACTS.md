# What is kept in Git and what stays local

Git contains maintained source/tests/configurations, readable historical source,
equations, frozen plans, reports, selected figures, compact endpoint results,
protocols, final audit receipts and essential retirement source snapshots.
The active candidate decision is in [CURRENT_STATE](CURRENT_STATE.md).

Datasets, tensor checkpoints, saved gradient tensors, repeated run source ZIPs,
allocator pickles, compiler caches and detailed training histories stay local
and are ignored. They are not deleted. A fresh clone does not contain trained
weights or data, and cannot independently rescore historical checkpoints without
those artifacts. Reports and audit receipts describe measurements already made;
they do not substitute for the missing raw artifacts in a new verification.

The [local artifact inventory](evidence/local_artifacts.csv) records relative
paths, byte sizes and SHA256 hashes at release time. It is an inventory, not a
backup or a download service. Keep the local workspace or a separate backup if
you need the excluded tensors. File links to raw results in historical reports
resolve in that complete workspace; they may be absent from a small Git clone.

Essential pre-retirement source snapshots are in [archive](archive/README.md).
Old experiment drivers intentionally assert historical source hashes. Reproduce
them in their recorded source tree rather than changing those assertions to
accept newer code. Scientific scripts under results are historical, frozen
source; active lint/tests cover src, tests, scripts and main.py.

To generate a new inventory without overwriting an existing one:

    uv run python scripts/artifact_inventory.py --output research/evidence/local_artifacts_next.csv

Before a future commit:

    uv run python scripts/artifact_inventory.py --check-index

The index check rejects dataset/checkpoint/cache files, repeated run ZIPs,
individual files above 5 MiB and a tracked working-tree total above 48 MiB.
The total is uncompressed source/evidence size, not a GitHub quota estimate.
The check applies to the Git index; run it after staging intended files.
