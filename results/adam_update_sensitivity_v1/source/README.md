# H119 source guide

This is a frozen diagnostic using six saved H117/H118 probe gradients. It adds
no maintained model variant or training loop. Read the
[report](../../../research/adam_update_sensitivity_results.md) and
[prospective plan](../../../research/adam_update_sensitivity_plan.md) first.

| Script | Purpose |
|---|---|
| `prepare.py` | Verify H118 evidence, snapshot navigation and freeze inputs/code |
| `study.py` | One common native AdamW step on a disposable model; original CPU stage |
| `prepare_recovery.py`, `recover.py` | Preserve six CPU cases and run only the six remaining CUDA cases using the unchanged step function |
| `analyze.py` | All 45 CPU-FP64 direction pairs, 12 distortions and native/device comparisons |
| `prepare_audit_recovery.py` | Freeze completed outputs without changing the original failure |
| `audit.py` | Independent NumPy formulas, moment/clip checks and all reported metrics |
| `summarize.py`, `plot.py`, `report.py` | Derive compact evidence, figure and readable report |
| `publish.py` | Update navigation after verifying its preserved pre-study contents |
| `launch.py` | Exclusive subprocess through the recorded UV-managed Python, with immutable logs and exit files |

Original `prepare_audit.py` remains frozen but was not run: it requires a
successful original study exit. The separate recovery preparation checks the
actual failed/successful stage sequence instead. No numerical rule changes.

Executed scientific stages: prepare → study (six CPU steps, then guard failure)
→ prepare_recovery → recover (six CUDA steps) → analyze → prepare_audit_recovery
→ audit. The audit process exits normally while reporting **numerical FAIL**:
all six CPU clipping comparisons exceed the fixed tolerance. All CUDA checks
and independent metric comparisons pass. These distinctions are preserved in
the [final verifier](../../verification/adam_update_sensitivity_final_v1.py).

The launcher uses the UV-managed CPython 3.12.9 runtime with the existing
`.venv/Lib/site-packages` on `PYTHONPATH`; it does not resolve or modify the
locked environment. Its absolute interpreter path records this Windows host.
Run modules from the repository root. Existing protocol/output assertions
prevent overwriting evidence: a new execution needs its own prospective output
directory and protocol, not edited hashes in this frozen study.

Raw `.pt` artifacts, detailed JSON and original navigation copies stay local.
Compact `.json.gz`, `.csv.gz` and `.log.gz` preserve metrics and failures for
Git. Hashes identify raw tensors; they cannot replace unavailable tensors in a
fresh clone. Accounting is 12 disposable optimizer steps and zero language
training updates, forwards, backwards, training targets or validation scores.
