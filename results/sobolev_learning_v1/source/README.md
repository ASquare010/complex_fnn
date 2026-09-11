# H109: learning after derivative-aware initialization

This experiment tests H108's qualified calibration component under actual
value-only training on new, disjoint FFN input windows. Read the
[frozen protocol](../../../research/sobolev_learning_plan.md) first. It preserves
the original >=70%-FFN parameter target and does not register a new model.

`model.py` supplies a loss-normalization adapter around existing raw dense FFNs,
smooth teacher target generation and local value/JVP scoring. `qualification.py`
checks loss and gradient scaling, one-step Adam identity, deployment identity,
JVPs and parameter counts in four isolated groups. `study.py` uses the maintained
capture and training helpers; all controls share fresh data, streams and rates.

Four H108 initialization files are reused exactly: value-greedy/value readout,
value-greedy/Sobolev readout, Sobolev-greedy/Sobolev readout and random/value
readout. Only primary widths 448/298 are tested. There are 18 fresh input sets,
144 fits at 600 updates each, 72 selections and 216 endpoint states. The old
initialization data is reused; the fitting/selection/reporting windows are fresh
relative to H105/H107/H108. The teacher training corpus and training seed are not
new. The first 32,768 rows fit, next 4,096 select and last 4,096 report.

## Execution and preserved preflight failure

Use the recorded UV-managed Python 3.12.9 and the existing CUDA package directory
from `.venv/Lib/site-packages`. `launch.py` records the malloc allocator/hashseed
settings and writes exclusive stage logs. The first launch was `launch.py study`.
It failed before the protocol, qualification, capture or any optimizer update:
the maintained hash helper needs a `Path`, and string call sites were supplied.

`repair_preflight.py` archives the original source in `preflight_failure/study.py`,
hashes its source/error/log/exit record, and fixes only those path types. That
one-off repair is already complete; do not run it again. `preflight_recovery.json`
records the change. `study_recovery.py` then invokes the corrected study under a
separate stage name. This is the first scientific execution, not a training retry.

```text
uv run --no-project --python C:/Users/Cuebric/AppData/Roaming/uv/python/cpython-3.12.9-windows-x86_64-none/python.exe python results/sobolev_learning_v1/source/launch.py study_recovery
```

Existing logs and protocols intentionally prevent overwrite. A scientific replay
needs a fresh root and protocol, all recorded local teacher/initialization files,
the WikiText-2 cache and prior collection metadata. It must preserve this run.
The frozen source hash list includes every scientific dependency recorded before
the first data capture; postprocessing files were added afterwards.

## Evaluation and audit

`analyze.py` exports all 216 endpoints in compressed CSV, full mean/median/sample
variance/seed/depth statistics, equal-rate and selected comparisons, gradient and
activation summaries, and the predeclared decisions. No reporting derivative
participates in optimization or endpoint selection. `plot.py` creates the static
figure; the [report](../../../research/sobolev_learning_results.md) interprets it.

`audit.py` independently recaptures every input through the decoder, recomputes
smooth targets from raw state tensors, uses autograd JVPs, checks exact H108
initializations and index streams, verifies 600-step records/initial losses,
rescores all 648 selection/output/derivative metrics and checks all 72 selections.
The teacher target checks use their original batch512. Student rescoring uses
batch257 and explicitly recorded aggregate tolerances. It performs zero updates.

Public gzip records retain complete metadata; plain originals, 18 input/target
datasets, all endpoint tensors, source ZIP and per-update histories stay local.
The repository metadata is not a backup of those tensors. A clean clone lacking
them cannot independently reconstruct the historical score audit.

`inference_only.py` is a posthoc fresh-process resource check with no optimizer
or backward pass. It records 72 selected models and 18 full references separately
from the original inference measurements taken after training; it does not
replace any frozen gate input. `inspect_summary.py` prints a compact human review
of the complete summary without running models. The final standard-library
[verifier](../../verification/sobolev_learning_final_v1.py) checks decisions,
source preservation, summary/CSV consistency and exact lossless packaging.

The maintained tree and tests are unchanged from the previous 116-test pass.
The four new qualification groups and independent audit are separate evidence.
No local fitting result proves NLL, whole-model memory, long-run convergence,
gradient stability, novelty or the broad research objective.
