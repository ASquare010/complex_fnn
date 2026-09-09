# H091: reuse projections without forcing an input bottleneck

H090 made progress: its complete three-seed grid and independent audit rejected
four learned geometry recipes. Commit b14810b preserves that evidence. The gold
goal remains unchanged and unmet. This round qualifies a different allocation
of the same weight budget before assigning training.

## The discarded full-expansion allocation

Let U have m rows and let each projected coordinate emit q nonlinear features.
An unrestricted dense readout costs q*d*m weights, in addition to d*m for U.
With q=2 and our 350,208-weight budget at d=384, m=304. Every such function
f(x)=D*Phi(U*x+b) is constant along ker(U), regardless of Phi. It loses at least
80 input directions. At the general 70% reduction boundary against 8*d^2 full
FFN weights, m <= 2.4*d/(q+1), so q>=2 forces m<d.

For X~N(0,I_d) and target X, even an arbitrary predictor of U*X has normalized
population MSE at least (d-rank(U))/d. Conditional expectation is P_U*X, and
the independent orthogonal residual has covariance I-P_U. At m=304 the bound
is 80/384=20.8333%. This does not bound our reused uniform nonlinear targets,
residual Transformers, preceding attention or normalization. It is a scoped
reason to avoid forcing a linear input bottleneck in the new isolated FFN.

## Partial feature expansion

Keep m=d=384 projected coordinates, and expand only the first k=144:

    z = U*x
    f(x) = D * concat(GELU(z), psi(z[:144])).

U has 384*384 weights; D has 384*528. Total: 350,208, exactly the narrow
GELU h456 count and 70.3125% below a full h1536 GELU FFN. The construction has
528 output features from 384 directions. Its square U can have full rank;
this is neither a bound on its conditioning nor a guarantee that the nonlinear
map/readout preserves information. No full-network gradient claim follows.

Test four secondary bases:

- rational: z*softsign(z) = z^2/(1+abs(z)), implemented z*(z/(1+abs(z)));
- Hermite degree 2: (z^2-1)/sqrt(2);
- Hermite degree 3: (z^3-3*z)/sqrt(6);
- sine: sin(2*z).

The rational base is even, grows at most linearly and has |psi'|<=1.
Its derivative is sign(z)*(1-(1+abs(z))^-2), continuously zero at z=0.
Hermite bases have unit variance under a standard Gaussian but unbounded
derivatives; that Gaussian premise is not a claim about trained activations.
Sine has bounded value and |psi'|<=2. These are fixed bases whose independent
outgoing vectors adapt during learning; there are no extra scalar shape weights.
This deliberately reallocates projection weights to output-specific nonlinear
coefficients, rather than assuming one shared scalar curve is sufficiently rich.

Include two equally sized controls: duplicate GELU(z[:144]), and identity
z[:144]. A duplicate folds into the first 144 columns of the base readout and
adds no function class. It changes optimizer coordinates. The identity control
separates a new basis from a simple independently weighted linear component.
A smaller core GELU h384 has 294,912 weights and is the initialization control.
Future training must also retain full/narrow GELU, ReLU, SwiGLU and the strong
four-shift GELU control from H090 with equal new tuning effort.

Initialize U and the first 384 columns of D from the existing seed/name-derived
Gaussian generators, standard deviation 1/sqrt(384); set the extra D columns
to zero. All six partial forms initially represent the same core GELU function.
The wider GEMM can round differently from the separate core GEMM, so CPU FP64
comparison uses atol/rtol 1e-11, not an unsupported bitwise cross-shape promise.
Nonzero extra weights must be tested before qualification is granted.

## Literature and novelty boundary

[CReLU](https://proceedings.mlr.press/v48/shang16.html) already expands projected
features into complementary phases. [FAN](https://arxiv.org/html/2410.02675v3)
combines learned projections with sine/cosine and conventional nonlinearities.
A [FAN mechanism study](https://arxiv.org/html/2512.14873v3) explicitly investigates
which activation components contribute. Polynomial bases also have substantial
prior work, including [ICLR 2026](https://openreview.net/pdf?id=QywpTFx86x).
[Squareplus](https://arxiv.org/abs/2112.11687) is an established algebraic smooth
rectifier; the rational basis here is simply z times known softsign. None of
these simple ingredients, the bottleneck bound, or this combination has verified
novelty. The empirical question is the weight allocation under the gold budget.

H018/H019 used two projected gate/value branches and their products; H023 used
three sigmoid-quadratic branches and products. Those failed recipes remain
closed. This proposal uses no value projection or branch products, and keeps
the base GELU pathway with zero-initialized independent secondary readout.

## Frozen qualification, no training allocation yet

Use seven forms (six partial plus core), seeds 17/29/43 where stated, four CPU
threads, native PyTorch, TF32 off and one GPU process. Freeze source before any
numerical checks. Do not modify prior evidence or relax a failed tolerance.

Collect and run exactly 23 tests once:

1. Seven actual parameter-count and initial-state checks.
2. Six FP64 independent basis/formula/first-derivative and numerical gradchecks,
   including zero and nonzero inputs. Analytic tolerance 1e-11; finite differences
   epsilon 1e-6, atol 1e-5, rtol 1e-3, fast mode; record forward counts.
3. One initial-function test across all six forms against core, FP64 1e-11.
4. One duplicate-folding test with nonzero secondary readout, FP64 1e-11.
5. One input-bottleneck/rank test: independently constructed orthonormal rows,
   explicit nonzero kernel vector and projector trace 80; all six full-lift
   feature families remain constant along that vector, tolerance 1e-10. Check
   the three partial U matrices have numerical rank 384 and record singular values.
6. One analytic rational-slope check on 4,097 FP64 values in [-8,8], including
   zero; compare derivative at 1e-11 and enforce its unit bound.
7. Six whole-FFN CUDA tests with nonzero secondary readout: FP32/BF16 output and
   input/up/down gradients versus an independent reference, atol/rtol 1e-5 and
   0.02 respectively. Non-reentrant checkpoint results must be bit-exact.

For the partial activation, retain native GELU for the main pathway. Compute
secondary basis arithmetic in FP32 (FP64 when requested), cast once, concatenate.
The reference uses independently written formulas and the same precision policy.
Qualification GPU probes use seed 9842, transposed 2x3x384 input, and a fixed
nonzero secondary readout generated at seed 9843 with standard deviation 0.02.
Persist paired outputs/gradients with hash and exact tensor readback.

Stop on any failed check. Independently audit the saved observations and source
preservation before a separate fitting protocol. No dataset, optimizer update,
speed claim or new active model folder is authorized by mathematical checks
alone. Preserve raw tensors locally and compact evidence/source in Git.
