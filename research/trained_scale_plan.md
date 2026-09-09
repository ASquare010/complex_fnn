# Trained scale check: width 384, eight layers

Frozen before these runs. Previous goal turns made progress: five quadratic
settings and stronger controls were implemented, trained, audited and retained;
none earned promotion. This round addresses a different missing requirement:
actual trained width/depth evidence for the strongest established recipes.
It is not a novelty claim or a new architecture sweep.

## Fixed comparison

Double width from 192 to 384 and depth from four to eight. Keep six heads,
so head dimension becomes 64; keep context128, vocab4096, the frozen TinyStories
cache, batch16, BF16, base LR6e-4, AdamW(.9,.95), clip1, warmup/cosine schedule,
seed17 and 32768 fixed validation targets. The initial screen is 200 steps
(409600 training tokens) for every model. This is a short trained scale probe,
not convergence or evidence that the data budget is sufficient for 15M weights.

| Recipe | Hidden | Unique FFN weights | Total weights | FFN matrix forward FLOPs/token |
|---|---:|---:|---:|---:|
| Full SwiGLU | 1024 | 9437184 | 15735168 | 18874368 |
| Full GELU | 1536 | 9437184 | 15735168 | 18874368 |
| Calibrated narrow SwiGLU | 304 | 2801664 | 9099648 | 5603328 |
| BlockShuffle SwiGLU, G8 | 2048 | 2801664 | 9099648 | 5603328 |

The compressed recipes preserve 70.3125% FFN weight/matrix-FLOP reduction.
Total weights fall by about 42.17%. The gain in total-model reduction follows
from the deeper/wider model's parameter composition, not a new compression rule.
Both attention geometry and depth change; this is not a width-only scaling law.

Transfer the locked recipes without tuning: narrow down initialization and LR
use h_reference/h=1024/304=64/19 with pure-decay compensation. BlockShuffle uses
its semiorthogonal factor initialization, factor fan-in LR and native gate
recomputation with parameter decay. Full dense models retain their original
initialization and optimizer. Residual output initialization uses the existing
1/sqrt(2L) factor. Matrix and sampling seeds remain name-local and common.
This does not establish muP transfer or equalize historical selection effort.

## Gates and bounded budget

Run correctness/count/common-initialization checks at the new dimensions,
then exactly four 200-step runs, one GPU job at a time. Nonfinite training stops
that model and blocks promotion; retained failure information is never erased.
If peak training allocation exceeds 6GiB, stop before another larger run and
reassess all models under one shared memory protocol. Do not silently change
batch size, precision or context for only one model.

A full four-model 800-step seed17 cohort is earned only if BlockShuffle is
within 2% NLL of BOTH full dense controls, beats calibrated narrow SwiGLU by
at least .5% NLL, retains its exact matched weight/FLOP budget, remains finite,
and has training peak <=110% of full SwiGLU. These are predeclared scale-screen
continuation thresholds, not final acceptance criteria. All four longer runs
start fresh under the 800-step schedule; extending a decayed 200-step checkpoint
would change the optimizer protocol. No failed quadratic or paired candidate
is included in this comparison.

Measure fixed-shape serving of the BlockShuffle and calibrated narrow
checkpoints against full SwiGLU using existing rotating-order eager and shared-
stream V2 CUDA graph audits. Label packed factors and dense caches separately.
The final serving floor remains .8 times full throughput; no timing result
from another date/shape is substituted. A wider GEMM may change kernel utility,
but this is a hypothesis, not an assumed performance gain. Preserve raw rounds,
cache bytes, allocated peaks, exact validation and fresh-input correctness.

If continuation gates fail, retain the measurements and inspect the concrete
quality/optimization/runtime limitation before another architecture choice.
If the longer cohort succeeds, lock seeds29/43 and then evaluate a broader
corpus. Full gold acceptance still needs <=1% degradation against both dense
controls, strong narrow controls, speed/memory, multiple seeds, trained scale,
convergence and broader data. No single run can establish a breakthrough.

## Artifacts and execution

Use configs/scale384_{full_swiglu,full_gelu,calibrated_narrow,blockshuffle}_screen.json
with the existing src.core.cli. Runs are named scale384_{recipe}_s17_200.
All ordinary trainer provenance, checkpoint, history, saturation/gradient,
validation and timing artifacts are required. Analytic counts are checked
against actual initialized parameters before training. Scaling and serving
results must be reported separately from the earlier width192 cohort.
