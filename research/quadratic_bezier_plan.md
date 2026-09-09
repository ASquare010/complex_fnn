# H022-H024: the requested single quadratic and three small branches

Frozen before function or language-model training. Implementation stays in
src/bezier_ffn; data, training, diagnostics and reporting stay in src/core.

## Equation and exact scope of the proofs

Use the requested B(t)=2(1-t)t*c1+t^2*c2, with P0=0 and t=sigmoid(z).
No X-curve inversion or second coordinate polynomial is needed. This is a
quadratic Bernstein basis with learned group-shared controls, not a new basis.

H022, direct activation: c1=-1+.5*tanh(theta1), c2=2+.5*tanh(theta2).
At initialization q(z)=4*sigmoid(z)^2-2*sigmoid(z)=u+u^2,
u=tanh(z/2). Therefore q(0)=0, q'(0)=.5 and q''(0)=.5,
matching SiLU through second order at zero. Their tails differ substantially.
Bernstein weights are nonnegative and sum to one after including P0, so
|q|<=2.5. The derivative is
q'=2[(1-t)c1+t(c2-c1)]t(1-t), hence |q'|<=2.
Both tails have zero limiting derivative. No nonvanishing-gradient claim follows.

H023, three branches: reshape Ux to three h/3-channel branches and apply q_i.
Output feature i is r_i=q_i+lambda_i*q_j*q_k, with distinct i,j,k and
lambda_i=.5*tanh(eta_i), initially zero. D concatenates/readouts these features.
This is three small projected feature networks with pair products before a
common output projection. A purely linear mixture could be absorbed into D;
these products introduce cross-branch interactions instead. In branch
preactivation coordinates, d^2 r_i/(dz_j dz_k)=lambda_i*q'_j*q'_k,
which equals lambda_i/4 initially for z=0 at fixed nonzero lambda. This is
not a function-class separation from a dense FFN with learned input mixing.
The zero-lambda family exactly contains the unmixed control, but finite-step
optimization need not find its optimum. |r_i|<=5.625. Each row and column of
its 3x3 feature Jacobian has absolute sum <=2+2*(.5*2.5*2)=7, so
||J_r||_2<=sqrt(||J_r||_1||J_r||_infinity)<=7. Consequently the FFN
Jacobian norm is <=7||D||_2||U||_2. Trained matrices remain unconstrained;
this upper bound gives no global stability or lower singular-value guarantee.

H024, residual activation: phi(z)=SiLU(z)+B(sigmoid(z);delta1,delta2),
delta_j=.5*tanh(theta_j), initialized at zero. Convexity gives |B|<=.5;
the same derivative expression gives |B'|<=.5. The complete FFN is
D[phi(Ux)*Vx]. It starts exactly at the calibrated narrow SwiGLU, including
common matrix initialization, forward dtype and first common-weight gradients.
The value projection still makes the whole gated map unbounded. Compare one
curve for the whole layer (G=1) with eight independently learned groups.

The implementation evaluates curve arithmetic in FP32 under BF16 autocast,
casts the correction before adding to SiLU, and preserves FP64 for derivative
tests. Extra scalar coefficients use the base LR with no weight decay.
Nonlinear arithmetic and saved tensors are real costs excluded from the matrix
FLOP counter; synchronized runtime and measured memory decide utility.

## Fixed screens and controls

Use the established 200-step, seed-17 TinyStories protocol and data hashes.

| Screen/config suffix | Hidden | Groups | All-layer FFN weights | Recipe/control |
|---|---:|---:|---:|---|
| direct | 228 | 3 | 350232 | Original dense recipe; GELU h228 |
| mixed | 228 | 3 | 350244 | Same as direct; direct is the product ablation |
| residual_shared | 152 | 1 | 350216 | Width-calibrated SwiGLU h152 |
| residual_grouped | 152 | 8 | 350272 | Same calibrated recipe |

Configs: configs/quadratic_{suffix}_screen.json. All have 700416 FFN matrix
forward FLOPs/token, versus 2359296 for full SwiGLU. Extra learned coefficients
are counted, not rounded away. Every candidate retains >70% FFN reduction.
Direct/mixed use uniform matrix LR and original initialization, exactly like
GELU h228. Both residuals use the locked width initialization/LR calibration
and product-decay correction. This deliberate recipe difference is disclosed;
compare each architectural mechanism with its own matched control first.

Before LM training: exact counts, FP64 first/second derivatives, the scoped
bounds above, zero-interaction and zero-residual equivalences (including CUDA
BF16 common-weight gradients), finite backward and an actual tiny overfit.
Then run the existing three interaction tasks for 300 steps, seed17, using
quadratic direct/mixed h6 G3, residual h4 G1, GELU h6 and SwiGLU h4.
Counts including common scalar bias: 55,58,51,49,49 respectively. Tiny-model
coefficient overhead is material, so these diagnose mechanisms rather than
establishing parameter superiority. No architecture-specific toy trainer.

Promote at most one candidate to 800 steps, selected by lowest final NLL, only
if all hold: >=.5% lower NLL than calibrated SwiGLU h152; >=.5% lower than
matched GELU h228; >70% FFN reduction; finite outputs/gradients; training peak
<=110% of full SwiGLU. The mixed candidate must additionally beat direct by
>=.2% NLL to justify its interaction. Thresholds concern this cheap screen,
not the final research target. Speed is measured and disclosed; any promoted
candidate must earn the established >=.8 relative serving throughput gate
under equally applied timing before wider replication. Unsuccessful screens
are retained; no late threshold changes or automatic long-budget promotion.
A survivor at 800 steps must improve on calibrated narrow control and earn
three seeds, full controls, trained scale and a broader corpus. Single-seed
screening cannot establish novelty, convergence or broad capability.

## Prior art and interpretation

Learned bases and activation coefficients have substantial prior art. See
[learnable polynomial/trigonometric activations](https://arxiv.org/abs/2502.01247),
which study initialization and larger vision/language settings, and
[GLU variants](https://arxiv.org/abs/2002.05202) for gated FFNs. Bernstein networks
also predate this experiment, e.g.
[adaptive surface reconstruction](https://doi.org/10.1016/S0952-1976(01)00037-9).
Our limited literature search does not establish novelty of the exact mixture.
These elementary algebraic bounds are useful implementation constraints, not
proof that a new layer universally exceeds current state of the art.

## Follow-up amendment after the four frozen screens

All four initial screens failed promotion. Shared and grouped residual NLLs
are 4.162753 and 4.164174, versus calibrated SwiGLU 4.166703: gains of only
.095% and .061%. Direct/mixed NLLs are 4.265256/4.273424, worse than GELU
h228 (4.192404). Zero sampled sigmoid tails and small learned controls do not
support blaming tail saturation for this result. Branch products worsened LM
NLL despite helping the selected function tasks. Do not promote those settings.

H021 established a conventional width-calibration improvement, but the direct
quadratic and its GELU control have so far used the original recipe. Freeze
exactly two further 200-step seed17 runs: direct quadratic h228 G3 and GELU
h228, BOTH using fan-in initialization, fan-in down LR and product decay.
The ratio is 768/228=64/19. This tests transferring the same calibration rule;
it does not change the activation or mix branch products into another model.
Promote calibrated direct only if it meets the original quality/memory/count
thresholds, now versus BOTH calibrated SwiGLU and calibrated GELU controls.
The earlier runs and their failed gates remain visible.

Also repeat the unchanged calibrated SwiGLU 200-step recipe in a separate
reproducibility folder. New absolute throughput was several times historical
throughput despite identical recorded hardware/software. This repeat checks
numerical reproducibility and timing context, not another tuning choice.
Compare synchronized serving in rotating order if any candidate is promoted;
never infer relative speed from these temporally separated training runs.

## Stronger GELU control: bounded longer-budget check

The calibrated direct curve reaches 4.145671, but equally calibrated GELU
reaches 4.100158. The direct curve fails its GELU gate, so no quadratic setting
is promoted. Calibrated GELU improves 2.200% on original matched GELU and is
only .581% above full SwiGLU at 200 steps. This is an existing-activation
baseline result, not a new primitive. Freeze one seed17, 800-step run of
calibrated GELU h228 to test whether this early conventional-control advantage
persists. Use exactly the 200-step recipe with the established 800-step
schedule; compare against existing 800-step full GELU, full/calibrated SwiGLU
and BlockShuffle. No extra candidate search is implied by this control run.
