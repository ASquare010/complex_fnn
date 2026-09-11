# H113: classifier precision diagnostic

Read [the frozen plan](../../../research/classifier_precision_plan.md) and
[the measured result](../../../research/classifier_precision_results.md).
This study changes no maintained model or helper and performs no optimization.

- `common.py` defines the five policies, gradient metrics, conversion-node
  inspection, double checks, gradcheck and the small BF16 accumulation witness.
- `capture.py` saves real normalized decoder rows from all fixed H112 states.
- `study.py` measures the fixed 24-state grid and saves every gradient tensor.
- `prepare.py` verifies prior evidence and freezes 98 source dependencies,
  24 source checkpoints and both corpus manifests before execution.
- `launch.py` uses the same absolute UV-managed runtime, exclusive logs,
  malloc/hash settings and bytecode bypass. The study preloads dependencies
  before CUDA, with no claim that this fixes prior native failures.
- Later `audit.py` independently recaptures through native forward, derives the
  FP64 gradients explicitly and checks every policy rerun. `analyze.py` applies
  the prospective gates and reports the missing whole-process-peak limitation.
- `plot.py` shows every fixture and its corpus median. `report_values.py` is a
  read-only text view; `prepare_publication.py` preserves the prior Git policy.

Raw `captures/*.pt` and `gradients/*.pt` stay local and are ignored. All numeric
records are available in `result.json.gz`, `summary.json`, `captures.json`,
`metrics.csv.gz`, `qualification.json` and `audit.json`. Gzip exports decompress
exactly to the unchanged original records. Full artifacts remain hashed for
local audit. Historical output roots must not be reused or overwritten.

The final CPU verifier is `results/verification/classifier_precision_final_v1.py`.
It checks existing evidence and exports, without retraining or changing scientific
sources. New reproductions require a fresh root and prospective manifest. No
extra timing repeats or language training are launched by rerunning analysis.
The maintained repository remains two model folders, five variants and eight
recipes, unchanged from its previous 116-test pass.
