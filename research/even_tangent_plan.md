# H153: does an explicit even tangent add useful local capacity?

Previous turn: progress. H152 found negligible benefit from learned odd shapes,
and its product benchmark was inconclusive. Diagnose, do not repeat that training.
At zero core biases, the softsign-residual stack and linear readout (zero bias)
are odd. Theta and readout-weight tangent features are odd; core-bias tangent
features are even. An even target is orthogonal to odd features under a symmetric
input distribution. This is an initialization/local statement: trainable biases
can break symmetry, so it is NOT a global impossibility result for H152.

Proposed scalar: phi(x)=x+rho*tanh(theta+sign(x)*eta)*x/(1+abs(x)), rho=.5.
At eta0 it equals the old map. dphi/deta=rho*sech(theta)^2*abs(x)/(1+abs(x)),
an even direction. Each half-axis derivative is in[1-rho,1+rho], so the map is
monotone/bijective, with a derivative kink at zero when eta!=0. Its inverse uses
the existing scalar inverse with coefficient selected by sign(y); signs agree.
The readout model adds128 eta parameters (1312 ->1440), no duplicated activations.
Do not claim novelty: odd/even activations and H092/H099 parity mixtures pre-exist.
This is a two-sided invertible control in the current reconstructed stack.

CPU-only, no GPU training: initial H152 learned-model construction at fresh
seeds307/317/331, d32/L4. Three scalar output probes0/11/23 with targets x_i*x_(i+1).
These probes use separate coefficients: a relaxed local diagnostic, not joint
32-output learning. Base Gaussian samples2560 per seed, train1024/val512/report1024.
Pair each x with -x to compute even/odd parts exactly. Training-only feature/target
centering and column RMS scaling. Analytic derivatives checked by central finite
differences (epsilon1e-6,rtol1e-4/atol1e-7) for random theta/bias/eta directions.
Scalar inverse tested FP64 on signed logspace1e-12..1e12,relative-scaled<=1e-12.
Parity checks require unwanted component relative norm<=1e-12 at initialization.

Even tangent ridge controls: bias-only, duplicated bias, bias+eta. Duplication is
scaled by1/sqrt(2), so its Gram kernel and L2-ridge function equal bias-only;
verify predictions within1e-9. Normalize bias and eta columns using train stats,
scale each block in concatenations by1/sqrt(2). Rates/ridge penalties1e-6/.001/1,
objective trainMSE+lambda*||coef||^2; select by validation only.81 ridge solves.
Report held-out MSE, rank/novel feature energy, coefficient magnitudes, mean/median/
variance and paired improvements. Baseline output intercept handles target mean.

The tangent solve may require nonlocal updates. Therefore replay the selected
coefficients in the ACTUAL nonlinear model at fixed fractions.01/.1/1, including
output-bias intercept, and score the even output component. No fraction selection.
This is not full-output MSE: odd residual output remains and is explicitly excluded.
Diagnostic earns a full-model trial only if bias+eta beats bias-only reporting MSE
by>=5% for every probe/seed AND fixed.1 replay improves the zero-even predictor
by>=5% for every probe/seed. Otherwise no promotion; do not infer full learning
from an unconstrained tangent oracle. No production memory/runtime claim.

Freeze sources first, save features/targets/coefficients/replay outputs. Independent
NumPy audit verifies selected ridge normal equations, saved-score MSE and duplicate
predictions; no GPU/autograd/optimizer updates in audit. Preserve H152 evidence.
