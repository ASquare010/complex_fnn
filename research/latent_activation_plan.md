# H082 - Learnable curves inside BlockShuffle factors

Freeze before implementation or numerical execution. The previous goal turn made
PROGRESS by completing and independently verifying H081. Its three failed fixed
recipes remain closed. This stage returns to learnable activations at a different
location: the 384-wide intermediate representation inside each BlockShuffle
projection, before the fixed middle permutation. H034/H035/H068 changed the outer
activation/value branch; their failed recipes are not repeated or reopened.

## Primitive and controls

For each of the up, gate and down projections, replace
P_out^-1 B2 P_mid B1 x with P_out^-1 B2 P_mid phi(B1 x). Keep both factor tensors,
their names, the outer SwiGLU and all existing initialization. At zero controls
phi is exactly the identity. Share two learned scalars within each of eight
intermediate groups; use independent controls for each projection and FFN.

With a = 0.5 tanh(theta_a), s = exp(log(4) tanh(theta_b)), test:

- latent_tanh: phi(z) = z + a s tanh(z/s)^2.
- latent_sine: phi(z) = z + a s sin(z/s)^2, a constrained Snake-related control.
- latent_affine: phi(z) = z + 0.5 tanh(theta_a) z + 0.5 tanh(theta_b).

All theta values start at zero. The affine form has the same 48 extra weights as
either curve and tests whether learned gain/offset alone explains an improvement.
The second curve parameter has zero gradient at identity until amplitude learns;
record this expected non-identifiability rather than hiding it. Do not use the
failed H081 optimizer calibration. Retain the established factor learning rates.

The [theory note](latent_activation_theory.md) gives conditional scalar proofs:
tanh slopes lie within 1 +/- 2/(3 sqrt(3)), sine within [0.5,1.5], affine within
[0.5,1.5]. Both curved functions are bijective and bi-Lipschitz on the real line.
These statements do not hold as injectivity claims after floating-point rounding,
do not bound a whole network's Jacobian, and do not prevent saturated theta
gradients or poor optimization. The original linear projection is contained at
identity, but no strict full-FFN family separation or learning advantage is claimed.

## Fixed full-width fitting allocation

Use d384/G8 to retain the full-size block connectivity. Eight forms, in order:

| Form | Hidden width | Standalone weights |
|---|---:|---:|
| plain | 2048 | 350,208 |
| latent_affine | 2048 | 350,256 |
| latent_tanh | 2048 | 350,256 |
| latent_sine | 2048 | 350,256 |
| narrow_swiglu | 304 | 350,208 |
| narrow_gelu | 456 | 350,208 |
| full_swiglu | 1024 | 1,179,648 |
| full_gelu | 1536 | 1,179,648 |

Curves and affine exactly match each other. Plain/narrows have 48 fewer weights
(0.0137% of the smaller model); do not call that an exact parameter match. Each
new form retains more than 70% FFN reduction. Eight such FFNs would have 2,802,048
weights, adding 384 to the retained Transformer, but this stage does not construct
or train a Transformer. Pointwise work increases; matrix MACs stay unchanged.

Four generic nonlinear tasks x eight forms x seeds17/29/43 x rates0.001/0.003:
192 fresh cells, each 600 updates at batch256. Total115,200 optimizer updates,
29,491,200 example presentations. There are no language targets or full-model
resource runs. The 600-step budget is frozen before seeing this experiment; every
control receives the same budget. No extra rate, width, curve, task or step repair.

Use the H073 fitting loop, FP32 CUDA, TF32 off, four CPU threads, native eager
Torch, no compiler or training checkpointing, one bounded GPU worker at a time,
UV's existing compile/data extras. Preserve AdamW betas(0.9,0.95), epsilon1e-8,
zero decay, clip1 and constant base rates. Factors retain fan-in multipliers;
both narrows retain down LR correction. Shape parameters use multiplier1.
All common factor tensors and initial outputs match plain at each seed. Dense
and structured projections use H073's unit mean-row-energy initialization.

## Data, selection and task orientation

CPU seed9812 creates73,728 rows of384 Uniform[-sqrt(3),sqrt(3)] coordinates.
Split first65,536 train, next4,096 rate selection, last4,096 held-out reporting.
Seed9283 creates one fixed random permutation pi of input coordinates. Evaluate
each H073 nonlinear target on u=x[:,pi], then apply its fixed output rotation
from QR seed9282. This breaks the original contiguous input-group alignment
without changing the sparse target formulas or using model outputs as labels.
There are no architecture-derived teachers or easy linear task in the aggregate.

With cyclic a=u_i, b=u_(i+1), c=u_(i+2), d=u_(i+3), tasks in order:

1. smooth: sin(2a) + b^2 + exp(0.5c) - d.
2. oscillatory: sin(3a) cos(2b) + 0.5 sin(4c+d).
3. multiplicative: ab + 2bcd + a^2d.
4. piecewise: ReLU(a+b) - 0.7 abs(c-d) + where(a>0,b,c).

Divide each output coordinate by its training population standard deviation,
without centering. Report variance-scaled MSE, not NLL or centered NMSE. Save the
permutation and measure how often neighboring target coordinates cross input
groups; that is descriptive, not a selected permutation or a gate. These tasks
remain synthetic and sparse. Three optimization seeds share one dataset and do
not constitute independent dataset replications or arbitrary-data evidence.

Save CPU streams from seed20000+training_seed, shape600x256, indices in[0,65536).
The same stream is used for every task/form/rate. Selection and reporting rows
never enter gradients, target scaling or shape initialization. Score final
training/selection errors, choose lower final selection MSE per task/form/seed
(ties lower rate), and persist the choice before scoring either rate on held-out
rows. Keep both checkpoints and reporting scores; no intermediate selection.

## Qualification and measurements

Eight preflight checks: analytic scalar slopes/bounds/Bezier identity; three
parameterized FP64 finite-difference checks (affine/tanh/sine); initial identity
and common-factor gradient agreement plus nonzero-shape eager/checkpoint fidelity
on CPU FP32 and CUDA BF16; counts/optimizer coverage; train-only scaling and
sampler/permutation reproducibility; and selection/gate failure behavior. Finite
differences use eps1e-6, atol1e-5, rtol1e-3, fast mode, with actual forward-call
counts saved. Analytic FP64 comparisons use rtol1e-10, atol1e-11. Require exact
identity/checkpoint outputs and gradients. Qualification performs zero optimizer
updates; its scratch tensors never initialize a fitting run.

Pointwise work uses FP64 when supplied, otherwise FP32, and casts corrections
back before adding to the original input. Test finite values on the declared
input/control grids; do not claim all representable floats are safe. Shape
domains remain |a|<=0.5 and 0.25<=s<=4. Record sampled slopes, learned controls,
input/output RMS, correction RMS, control saturation and final shape gradients.

Preserve every update loss/preclip norm/time, initialization and final state
hashes, full weights/moments/RNG, clipping, configurations/groups and source/data
hashes. Synchronize every update; exclude first50 timings; reset allocated and
reserved peaks after warmup. These are standalone fitting resources, not full
Transformer memory or serving throughput. Save completed artifacts with fsync,
ZIP validation and semantic readback; retain all partial failures.

After each selected affine/curve checkpoint is fixed, score one extra held-out
ablation: zero both affine controls, or only curved amplitude, with all projection
weights unchanged. This measures dependence after coadaptation, not a causal
explanation of the training trajectory. There are36 such ablations, no updates.

## Frozen gates and completion

The assay is usable only if selected full GELU beats the zero-output predictor by
at least2% (three-seed geometric MSE ratio) on at least two of the four tasks.
If not, label INCONCLUSIVE_ASSAY_FAILURE and promote nothing, while preserving
the whole fixed grid. For EACH curved form require all of:

1. Complete finite grid and passing preflight/positive-control assay.
2. At least2% lower paired geometric held-out MSE than plain over all12 task/seed
   endpoints, and an improvement in each seed's four-task aggregate.
3. At least1% lower aggregate MSE than the equal-count latent affine control.
4. Aggregate MSE below both calibrated narrow controls.
5. Each task's three-seed error ratio <=1.05 versus plain and the zero predictor.
6. At least70% FFN parameter reduction.

Report comparison against both full controls and between the two curves without
misusing the language NLL threshold as an MSE threshold. Affine is a diagnostic
control, not an additional promoted architecture. A curved pass earns separately
frozen full-model resource qualification, then a language screen. A failure closes
that tested placement/curve/recipe/budget. No automatic extension or repair.

Pin the H081 final/result/source anchors, its121 sources, four new isolated
scientific files (125 total), this plan, theory, git state and environment.
Hidden durable coordinator records actual PID/UTC/log/return code. Stop on
runtime/nonfinite/fidelity failure, preserve evidence, and inspect live handles
before recovery. No automatic retry. Independent analysis audits every saved
checkpoint/choice, regenerates data and streams, rescoring selected held-out
predictions and ablations without updates, and derives all gates separately.
Update the report, current state, overview and ledger. No active-model addition;
three active model folders, six variants, nine recipes remain. The gold research
goal still requires language, multi-seed duration, convergence, scale, broader
data, practical resources and convincing published comparisons.

References: [latest failed language screen](blast_learning_screen_results.md),
[prior full-width fitting](ungated_fit_results.md),
[learned-activation evidence](learnable_activation_domain.md),
[Snake](https://arxiv.org/abs/2006.08195),
[architecture and gradient rank](https://arxiv.org/abs/2402.06751).
