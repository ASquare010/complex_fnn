# H107: can an explicit affine path replace nonlinear width on real FFNs?

H106's capacity gates qualify GELU h256 and SwiGLU h170 independently; h128/h85
remain closed. Passing a matrix bound does not establish that neurons can learn
the residual. This study tests that missing step with fresh captured windows.
The broader parameter, VRAM, language-quality and scale goal stays intact.

## Models and hypothesis

Candidates: Wx+b+D phi(Ux+a), using a full d-by-d affine path, and its SwiGLU
correction counterpart. d384, GELU h256: 344,704 parameters; SwiGLU h170: 344,020.
These save 70.78%/70.84% versus the 1,179,648-weight teacher FFNs. The explicit
affine path leaves more nonlinear capacity available for the residual function.
It also adds a matrix launch: measure whether that tradeoff is worthwhile.

Controls: ordinary GELU, ReLU, LeakyReLU (slope .01), PReLU (one learned slope,
initial .25) and SiLU at h448; ordinary SwiGLU at h298. Biases are included:
344,896/344,897 parameters for the ungated forms; 344,276 for SwiGLU. Every
control has essentially the candidate parameter budget. The original full
GELU/SwiGLU teachers define the desired function; they are not counted as fitted
compressed models. Known activations and affine bypasses are not claimed novel.

## Fair, explicit initialization

All models get the same pretrained teacher and training-only statistics. Inputs
are centered and scaled per coordinate (SD floor 1e-3); outputs are centered and
scaled by one global training RMS, with floor 1e-6. Normalization is folded into
weights/biases for deployment; no d-by-d normalization matrix or extra buffer
is hidden from parameter counts.

A deterministic permutation of teacher up-projection rows initializes all hidden
banks; smaller banks are prefixes of larger banks. Gated students use paired
teacher gate rows when available. With a GELU teacher, their gate starts at one
(zero weights, unit bias). This is post-training compression with access to
teacher weights, not evidence of learning from scratch.
The row permutation seed is 18000 plus the dataset seed; repository SwiGLU
convention is SiLU(up(x)) times gate(x).

Fit each model's readout by the same CPU FP64 ridge procedure on calibration
labels: center/standardize all design columns, ridge 1e-4, unpenalized intercept.
Design SD has floor 1e-6.
The candidate design includes input columns as well as hidden features; ordinary
controls use hidden features only. Both are jointly fitted rather than granting
a closed-form warm start only to the candidate. All weights train afterward.
The full affine-only fit is reported as a diagnostic baseline.

## Fresh data and fixed budget

Use the same two audited 3,200-step seed-17 teachers, depths 0/3/7. Exclude every
window used by H105 before drawing new data. One fixed permutation (seed 17001)
assigns disjoint 320-window groups to calibration/optimization seeds 71/83/97.
Per dataset: first 256 windows = 32,768 training rows; next 32 = 4,096 selection;
last 32 = 4,096 reporting. Windows contain 128 tokens. All come from the original
WikiText-2 training cache; the teachers trained on this corpus. This is fresh
local compression data, not unseen-domain or official language-test evidence.

Teacher capture uses BF16 at batch 8. Stored BF16 teacher outputs are labels.
Students train in FP32, TF32 off, batch 256, 600 fixed updates, AdamW (.9,.95),
eps 1e-8, decay zero, clip 1. Rates .001/.003 apply uniformly to every parameter;
no per-form rate advantage. Same sample stream for all forms/rates in a dataset.
The CPU randint stream seed is 19000 plus the dataset seed. Form order rotates
by dataset seed index to distribute warmup effects; rates remain ascending.
Use the shared function_fitting trainer, four CPU threads, one GPU at a time.

Eight forms x two teachers x three layers x three seeds x two rates = 288 fits,
172,800 updates. Evaluate the common initialization and both final checkpoints;
choose among these three using selection loss only, breaking ties in favor of
initialization and then the smaller rate. Report every endpoint and all three
seed results, not just the selected or luckiest score. No early stopping.

Before fitting, qualify actual counts, ridge stationarity, shared teacher-row
initialization, exact normalization folding, analytic gradients and finite-
difference input gradients. Retain numerical and scientific failures.

## Costs and gates

Record training forward/backward/update times, peak allocated/reserved VRAM,
parameter/optimizer bytes, losses, preclip norms, clipping, activation/gradient
distributions and selected raw-input inference latency. Timing uses CUDA sync
and warmup. Account separately for teacher capture and CPU initialization.
The pipeline peak includes teacher capture, not just the small student's fit.
Teacher checkpoints and all saved student states are organized and hashed.

Full-reference resource profiles use the normalized exact teacher at the same
batch and precision, 30 optimizer updates at LR zero (10 timing warmups) to
allocate Adam state without changing the function. There are 18 such profiles.
They measure resources, not a matched learning comparison. Inference benchmarks
use folded raw-input models; initialization can win quality selection but still
has the measured fitting cost charged when reporting this complete search.

Each candidate earns a separate language-insertion test only if, for BOTH
teachers: mean selected reporting MSE <=.05 (in training-variance units), paired
geometric-mean error at least 5% below every conventional control, each sampling
seed wins its aggregate against every control, and no depth's mean exceeds the
best conventional control by over 5%. It must also remain finite, use >=70%
fewer FFN parameters, save >=25% mean training allocation versus the full profile,
and cost <=1.25 times ordinary GELU for mean update and raw-input inference.

Local MSE gates allocate tests; they are not substitutes for actual NLL, full-
model VRAM, longer training or broader-data validation. If neither passes,
close these fixed recipes without extra width/rate/step/kernel tuning. A useful
control result may motivate a separately specified follow-up, not a retroactive
candidate promotion. Failed arithmetic guards or runtime problems are distinct
from a scientific failure and must be preserved rather than silently rerun.

## Prior work

The affine-plus-nonlinear family has longstanding precedents including
[Wide & Deep](https://arxiv.org/abs/1606.07792). The directly relevant
[FFN linear-recoverability study](https://arxiv.org/abs/2606.19379) uses closed-
form affine recovery, residual probing and language insertion controls. H107
does not claim to invent this decomposition or transfer that paper's results.
Its proposed contribution is controlled evidence for or against a particular
parameter/VRAM tradeoff on this repository's trained teachers.
