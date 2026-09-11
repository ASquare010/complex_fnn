# H114 source map

Read the [prospective plan](../../../research/fp32_classifier_profile_plan.md)
and [results report](../../../research/fp32_classifier_profile_results.md).
This study contains an execution adapter, not another registered model family.

| File | Role |
|---|---|
| `common.py` | One classifier precision/chunking adapter, identity checks, double qualification, memory ledger, tensor hashing |
| `study.py` | One matched continuation loop for all four methods; warmup and measured updates, full validation, saved model/Adam/gradients |
| `prepare.py` | Verifies H113, source/data/checkpoint hashes; snapshots navigation; freezes H114 |
| `launch.py` | Exclusive scientific child using the existing UV-managed Python and environment |
| `inspect_inputs.py` | Read-only metadata inspection before freezing |
| `audit.py` | Later independent native rescoring, exact initial-gradient replay and all-batch verification |
| `prepare_audit.py` | Freezes that independent audit and completed artifacts before replay |
| `gradient_replay_diagnosis.py` | Six-pass zero-update replay diagnostic after the exact audit stopped |
| `audit_recovery.py`, `prepare_audit_recovery.py` | Explicit numerical replay recovery; also stopped and preserved |
| `completion_audit.py`, `prepare_completion_audit.py` | Native scoring/state/stream audit with zero backwards; gradient advancement stays held |
| `analyze.py` | Applies the original gates; exports all cases and updates, with no outlier removal |
| `plot.py` | All-fixture scientific figure |

The first five Python files and plan are included in the original 104-source
freeze. Later audit/reporting code is separately hashed; do not rewrite the
original protocol to make posthoc files appear prospective. Maintained
Transformer, token loss, dataset, diagnostics and optimizer utilities are reused.

The recorded launcher is invoked with the UV-managed Python executable, e.g.
`python -B results/fp32_classifier_profile_v1/source/launch.py study` from the
repository root, where `python` denotes the absolute executable in `launch.py`.
It avoids the earlier UV dispatcher panic while using the same UV environment.
It refuses existing logs/case directories. **Do not rerun into this completed
root**; reproduce in a new output root with a separate frozen protocol.

Raw `runs/*/initial_gradients.pt`, `final.pt`, histories and detailed metrics
remain local and hashed. `result.json.gz` preserves every phase and update;
`metrics.csv.gz` is the 56-case table; `updates.csv.gz` includes all 2,800
warmup/measured updates. The final receipt checks their source correspondence.

Both original gradient audits failed; read the [recovery record](../../../research/fp32_classifier_profile_audit_recovery.md)
and [completion-audit scope](../../../research/fp32_classifier_profile_completion_audit.md).
No complete gradient replay has passed, and no fresh LM allocation is earned.
The narrower completion audit verifies saved gradient integrity, not independent
reproduction of every gradient. No further tolerance relaxation was made.

GPU events measure stream elapsed time, not pure kernel activity. Allocated
and reserved CUDA memory are separate. Case boundaries clear unused cuBLAS
workspaces, never subtract their bytes during a case. These are short
checkpoint continuations with saved Adam state, not fresh language runs.
