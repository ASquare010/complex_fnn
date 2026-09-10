# H100: learned three-way interactions on hidden Gaussian directions

Previous goal turn: progress. H099 completed and independently audited 264 runs,
rejected its shared scalar-mixture recipe, and exposed poor cubic learning.
The gold goal remains unchanged and unmet. This is a new neuron structure,
not longer training or kernel refinement of the rejected scalar mixture.

## Mechanism and scoped mathematical claims

Learn three affine projections u=Ux+b, v=Vx+c, w=Wx+e, then use one readout:

    cp3:         phi(u,v,w) = u*v*w
    bounded_cp3: phi(u,v,w) = 4*u*v*w / (1 + u^2 + v^2 + w^2)

Each uses h228 features, 350,208 matrix parameters and 1,068 biases: 351,276
trained parameters. Full GELU/SwiGLU have 1,181,568/1,182,080 including biases,
so both clear 70% reduction. There is no scalar-shape parameter, raw input
readout or teacher-derived initialization. The factor four makes the two feature
values agree at u=v=w=1; it is fixed before fitting.

For r^2=u^2+v^2+w^2, |uvw| <= r^3/(3*sqrt(3)). Hence the bounded feature grows
at most 4*r/(3*sqrt(3)). Its u derivative is

    4*v*w*(1-u^2+v^2+w^2)/(1+r^2)^2.

The absolute numerator factor in parentheses is <=1+r^2, and
|vw| <= (v^2+w^2)/2, so each partial derivative has magnitude <=2 and the
feature-gradient norm <=sqrt(12). This is not a bound on learned projection
matrices, a whole network, or a lower gradient bound. Products can still have
zero derivatives; raw cp3 has unbounded derivatives and cubic growth.

Raw cp3 can represent the first three task families using just sixteen features:
quadratic via (q-1)(q+1)*1; cubic via q*(q-sqrt(3))*(q+sqrt(3)); product via
q_i*q_(i+1)*1. Count the learned readout and projections in the actual prototype.
Verify this construction only as a privileged representability check, never as
training initialization. The bounded form cannot represent global cubic growth.
A two-factor affine product has degree <=2 and is population-orthogonal to a
third Hermite target under Gaussian input. Biases do not remove that limit.
These are single-layer statements, not language-model or deep-network theorems.

## Fresh rotated data

Use 73,728 Gaussian inputs, dimension384, input seed9900. Use the first16 columns
of a full FP64 QR orthogonal matrix (seed9901) as an unknown input basis Q.
Targets depend on q=xQ, not a selected coordinate subset. An independent output
QR rotation (seed9902, first16 rows) maps the sixteen latent targets to384 outputs.
Use H099's four normalized latent functions: (q^2-1)/sqrt(2),
q*(q^2-3)/sqrt(6), q_i*q_(i+1), and the population-affine-free squared-ReLU
residual. Scale outputs by training-only population SD, without empirical centering.
Compute targets in FP64 and store FP32. The first65,536 rows train, next4,096
select rates, final4,096 report. No fitting sees Q or reporting labels.

The known-feature readout is explicitly privileged: its fixed Q buffer gives
the exact features and it learns only6,144 output weights. Report its fixed
6,144 basis entries separately. It must reduce error by95% on every task/seed.
It checks the feature/readout pipeline, not task-blind feature discovery.
All neural forms use zero readout/output-bias initialization and name-local
N(0,1/384) projection weights and Uniform[-1,1] projection biases as in H099.
The raw/bounded candidates have exactly the same initial weights.

## Controls and fixed screen

Thirteen forms: full GELU h1536, full SwiGLU h1024, full squared-ReLU h1536;
narrow GELU/squared-ReLU/StarReLU/fixed-rational-mixture/cubic-ridge h456;
narrow SwiGLU h304; two-factor product cp2 h304; raw and bounded cp3 h228;
and the privileged known-feature readout. Cubic-ridge uses (z^3-3z)/sqrt(6).
It tests whether a plain known polynomial is sufficient. StarReLU and the fixed
H099 mixture remain strong controls, not new inventions. Retired H018/H019
two-projection products are not reopened for language promotion; cp2 is a control.

Three seeds17/29/43, two rates .001/.003, batch256, 300 updates:312 runs,
93,600 optimizer updates and23,961,600 presentations. Draw each seed's stream
with seed30000+seed, identical for all forms. Rotate execution order.
FP32 native CUDA, TF32 off, four CPU threads, one RTX4070 Laptop GPU.
AdamW (.9,.95), eps1e-8, weight decay0, clip1, constant rate. Narrow down-weight
LR multiplier is reference width/actual width (1024 for SwiGLU,1536 otherwise).
Feature-aware control uses base LR; other parameters use base LR.

Fit twelve affine least-squares controls on exactly the sampled training
multisets using the audited GPU-statistics/CPU-small-solve method. These use a
different solver and are reported separately from neural updates/timings.

## Freeze and qualification

Freeze source, plan, environment and H099 audited anchors before numerical work.
Exactly30 checks:13 counts/zero-readouts,4 FP64 formula/gradchecks (cubic ridge,
cp2,cp3,bounded cp3),1 bounded-feature analytic derivative/growth test,
4 known-feature oracle checks,3 raw-cp3 representability checks,
4 whole-FFN CUDA checks of the new forms,1 common-zero-function check.

Formula/derivative comparisons atol/rtol1e-11; representation comparisons1e-10;
FP32 stored target oracle tolerance5e-6. Gradcheck eps1e-6,atol1e-5,rtol1e-3.
GPU uses nonzero readouts, transposed2x3x384 input/cotangents, FP32 atol/rtol1e-5
and BF16 .02; all nonlinear feature arithmetic uses an FP32 region with one cast,
preserving FP64 when requested. Non-reentrant checkpoint gradients match exactly.
Save paired tensors and independently recheck all pairs before fitting.
Stop on any failure; do not silently retry, edit equations or loosen tolerances.

## Independent decisions for raw and bounded products

Require finite complete evidence, the known-feature positive control and one
full conventional form halving zero error on at least three tasks. Then each
candidate must: clear70% total FFN reduction; beat every narrow/fixed/polynomial/
cp2/affine control by>=2% in geometric mean paired MSE; win each seed aggregate
against every control; stay within5% of the best control's mean on every task;
stay within1% of both full GELU/SwiGLU in aggregate; and cost<=1.25 times narrow
GELU's mean median update time. In addition it must halve cubic zero-predictor
error in every seed, since learning cubic interactions motivates this mechanism.
Candidate-vs-candidate ratios are reported, not added as contradictory mutual gates.

Assay learnability failure means INCONCLUSIVE_ASSAY. Otherwise a failed candidate
is REJECTED_AT_THIS_BUDGET. A survivor only earns independent confirmation;
it does not achieve the language, compute, convergence or broader-data goal.
Audit every initialization, checkpoint score, rate selection, sample stream,
affine fit, summary and gate. Record activation/gradient distributions, all loss
and timing histories, clipping and peak allocated VRAM. Report matrix-only FLOPs
as such. No automatic language run, active model integration or kernel work.

## Prior work and interpretation

[Pi-Nets](https://arxiv.org/abs/2006.13026) already uses tensor-factorized
polynomial networks, including shared factors; this direction is not untouched.
[Polynomial-net complexity/Lipschitz work](https://arxiv.org/abs/2202.05068)
provides prior analysis of regularization and bounds, not validation of our formula.
[Gaussian multi-index gradient-flow research](https://proceedings.mlr.press/v258/simsek25a.html)
studies difficulty discovering directions without low-order Hermite components.
Its correlation-loss assumptions differ from this finite-data AdamW/MSE study;
it motivates investigation but does not prove the cause of H099's failure.
No novelty or general superiority claim follows from this synthetic screen.
