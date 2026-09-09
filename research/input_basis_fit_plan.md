# H094: quickly falsify direct-input nonlinear expansion

H092 collected25 tests but its second child failed during Windows shm.dll
initialization: no numerical tests ran. H093 invokes the unchanged25 functions
in one interpreter. All pass; independent saved-tensor inspection verifies157
pairs, including56 exact checkpoint pairs. DLL root cause remains undetermined.
No tolerance, equation, initialization or precision policy changed.

## Fixed screen

Test H092's rational, Hermite and trigonometric bases; retain duplicate-even,
linear-pair, antipodal-GELU, direct-path GELU and core-linear mechanism controls.
Also rerun full GELU/SwiGLU, calibrated narrow GELU/ReLU/SwiGLU and four-shift GELU.
Fourteen forms, four tasks, seeds17/29/43, rates0.001/0.003:336 runs.
Each has300 updates, batch256:100,800 updates and25,804,800 presentations total.
This is a quick fixed-budget screen, not a convergence experiment.

Use fresh input seed9844, existing analytic smooth/oscillatory/multiplicative/
piecewise definitions, permutation9283 and output rotation9282. Data has73,728
rows of384 inputs:65,536 train,4,096 rate-selection,4,096 reporting. Normalize
outputs by training-only standard deviation without centering. All forms share
inputs, targets and seed-specific batch streams. This is fresh synthetic data
from previously studied families, not an unrelated real benchmark. No test
corpus or language quality is measured. The Gaussian identity bound is not a
claim about these uniform-input nonlinear targets.

One GPU, UV, four CPU threads, FP32 native CUDA, TF32off. AdamW betas(0.9,0.95),
epsilon1e-8, decay0, pre-update gradient clipping1, constant base rate. Conventional
references retain the existing calibrated down-projection rates(1536/h, or
1024/h for narrow SwiGLU). All direct-input forms use the base rate for every
weight. Their shared initial function and pair-wise same-shape U/D initialization
is qualified; conventional FFNs keep their original random initialization.
Thus conventional comparisons test the whole recipe. Pair controls isolate the
basis more closely; do not attribute every difference solely to expressivity.
Zero nonlinear readout initially makes U's first-step gradient zero by design.

Rotate execution order across task/seed. Select rate only on the selection split,
then evaluate both rates on reporting. Record all checkpoints, source/data hashes,
loss histories, preclip norms/clipping, activation/gradient diagnostics, synchronized
forward/backward/update timings after50 warmup steps, and peak allocated GPU memory.
Timing is native training at batch256; it is not inference throughput. No hidden
automatic retries or extensions. Stop on nonfinite state; preserve partial evidence.

## Decisions fixed before fitting

For each of the three candidates, require all of:

- Finite complete grid; full GELU beats zero predictor by2% on at least two tasks.
- At least70% fewer weights than full GELU and SwiGLU.
- Geometric-mean reporting MSE ratio<=0.98 against each of narrow GELU/ReLU/SwiGLU,
  four-shift GELU, duplicate-even, linear-pair, antipodal-GELU and direct-path GELU.
- Better aggregate reporting MSE than each of those controls on every seed.
- Per-task mean MSE<=1.05 times the strongest listed control on that task.
- Aggregate ratio<=1.01 against both full references (a proxy gate, not NLL).
- Mean selected median native-update time<=1.25 times narrow GELU.

Failure closes this fixed recipe at this budget; it does not prove the whole
function family incapable. A pass only earns a subsequent convergence/real-data
experiment. Gold success still requires language NLL, compute, scale and broader
data. Core-linear is a diagnostic, not a competitor omitted to manufacture wins.

Independently regenerate data, verify checkpoint counts and every rate selection,
and rescore every selection/report split without optimizer updates. Publish
mean/median/sample variance/SD per task, paired aggregate ratios and all decisions.
Raw cells and tensors stay local; publish compact summaries and an artifact manifest.
