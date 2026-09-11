# H115 source and evidence guide

This is the frozen 30-condition decoder diagnosis described in
[the report](../../../research/decoder_gradient_transport_results.md) and
[prospective plan](../../../research/decoder_gradient_transport_plan.md).
It performs no optimizer updates and changes no maintained model implementation.

| Source | Purpose |
|---|---|
| `common.py` | Five decoder modes, encoder traversal, hooks, errors and qualification |
| `study.py` | Six fixtures × five modes, fixed incoming gradients and four repeats |
| `prepare.py` | Verify H114; freeze 109 sources,18 inputs and document snapshots |
| `launch.py` | UV-managed Python worker, import preloads, logs and exit records |
| `prepare_audit.py`, `audit.py` | Separately freeze and run native-forward and saved-arithmetic audit |
| `analyze.py` | All-condition statistics, unchanged gates, four compressed tables |
| `plot.py` | Four-panel figure; no scientific execution |
| `report_values.py` | Compact result values for the report |
| `inspect_progress.py`, `inspect_publication.py` | Read-only local status/schema inspection |
| `publish.py` | Verify pre-study documents before updating current navigation |

`protocol.json` froze the first four Python files and plan before execution.
`audit_protocol.json` later froze the audit code, complete result, qualification
and all60 tensor artifacts before the independent audit. Inspection, analysis,
publication and final-verification sources are post-study and explicitly serve
those narrower purposes. Never edit frozen sources or overwrite this result root
to retry a scientific condition. A follow-up needs a new root and prospective plan.

The established launcher directly uses UV's Python3.12.9 interpreter and the
repository environment's packages. It preloads SymPy/PyTorch/Dynamo, confirms
CUDA is not initialized before configuring the worker, uses four CPU threads,
disables TF32, records faulthandler output and allows one GPU worker at a time.
These preloads are a recorded execution procedure, not a cure for the previously
observed intermittent Windows import failures. This study's scientific and
analysis stages completed without an import failure.

The main ledger contains60 classifier and240 decoder backwards. Four
CPU-double chain-rule cases add12 backwards; eight scalar rounding cases add
four CPU and four CUDA backwards. Total320, with zero optimizer updates.
The independent audit performs30 native forwards and zero backwards. It checks
stored first-gradient arithmetic and forward recapture, not a new gradient replay.

Public compact artifacts are `result.json.gz`, `audit.json.gz`, `summary.json`,
`qualification.json`, protocols, environment, compressed logs, exit codes and
four tables: `metrics.csv.gz` (30), `pairs.csv.gz` (120), `replays.csv.gz` (180),
`traces.csv.gz` (1,200). The report and figure explain aggregation and limits.
Local `conditions/` holds30 pairs of `inputs.pt` and `first_passes.pt`, plus
per-condition JSON. Repetition0 saves tensors for both classifier policies;
later repeats save complete hashes and error records, not all tensor copies.
Raw JSON/logs, all tensors and `before_documents/` remain local and ignored.

The stdlib verifier in
`results/verification/decoder_gradient_transport_final_v1.py` checks hashes,
counts, memory ledgers, table arithmetic, gates and exports without new GPU work.
Run it from the repository root using the UV-managed interpreter; it requires
the local checkpoints, tensors and data absent from a compact Git clone.
The final receipt records the exact scope and the unchanged previous116-test
pass; it does not claim a fresh maintained-suite run or completed research goal.
