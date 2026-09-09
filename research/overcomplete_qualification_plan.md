# H053: Overcomplete structured-headwise representation and calibration

Frozen before constructing the candidate or scoring its qualification. H052's
product-decay control failed and is closed. This new direction changes the
interaction structure, not another activation or optimizer coefficient.

## Candidate and purpose

Implement an unregistered standalone module in src/multihead_ffn/overcomplete.py:
d=384 -> m=512 through a two-factor BlockShuffle map, eight heads of width 64,
one private SwiGLU of width 200/head, then a two-factor map back to d. Both
mixers use G=16, with intermediate width d. No router, extra activation,
bias, sharing across layers or change to any registered model/trainer.

Exactly 2*d*(d+m)/G + 3*m*200 = 350,208 parameters/layer, matching H047 and
BlockShuffle, 70.3125% below full SwiGLU. The structure is a combination of
established structured matrices and headwise subnetworks, not verified novelty.

The immediate requirement is an explicit representation of the earlier
three-output quadratic Q, using actual constrained input AND output factors.
Merely escaping the square-mixer theorem's assumption is insufficient. Construct
weights, do not fit the witness with an optimizer. Keep the exact recipe in the
research note/script, not as a production-only special case.

## Frozen checks before any language-model budget

- Full-shape FP64 forward witness on 31 seeded Gaussian rows, relative L2 error
  <=1e-11. Three Hessian-vector products for each of eight seeded directions at
  a nonzero point must match I, diag(1..384) and ones*ones^T to relative 1e-11.
- Independent dense-factor/head-loop FP64 forward and every input/parameter
  derivative at small dimensions, plus input gradcheck. Invalid dimensions
  must fail. Exact count, deterministic initialization and old source preserved.
- Isometric input initializer A^T A=I and output B B^T=c^2 I, c=1/sqrt(16),
  at FP64 tolerance 1e-11. No maintained orthogonality or gradient lower bound.
- Derived private U/V std .02*sqrt(H); private D std .02*sqrt(1024/200).
  Explain why rectangular-head covariance prevents an exact distribution match.
  Record each head's covariance trace, effective rank and gate variance.
- On 4096 independent standard-Gaussian inputs, seeds 17/29/43, each initialized
  FFN output RMS ratio to a same-seed full SwiGLU reference must lie in [0.8,1.25].
  This only qualifies scale; it neither proves stable training nor equal NTKs.
- After CPU checks pass, one native CUDA BF16 qualification at B=16/context=128,
  compared against its own FP32 output and gradients on the same input/probe.
  Relative L2 limits: forward <=.02, input/all-parameter gradients <=.08.
  Ten AdamW steps on one fixed synthetic target, LR .0012, ordinary parameter
  decay .1, clip 1; all gradients/weights finite, all parameters receive nonzero
  gradients before the first update. No requirement to beat a synthetic loss.
- Record isolated-FFN allocated forward/backward peaks for this candidate and
  full SwiGLU under the same eager execution. This is not a full-model memory
  gate or a speed result. No gate recomputation or custom kernel added.

CPU numerical checks have a 300-second ceiling; CUDA qualification a separate
180-second ceiling and only starts after CPU qualification. Full regression
suite runs when no GPU experiment is live. One candidate, no seed/rate sweep,
no language training or validation scoring. Retain source, plan, outputs and
all failures. Native/import failures stop for diagnosis; no automatic retries.

Only all required checks passing can earn a separately frozen integration and
balanced short language-model screen. A constructive witness proves one function
is present, not class containment, novelty, easy learning, LM quality or universal
optimality. Existing negative headwise, learnable-activation and decay studies
remain. The full research goal stays unchanged and unmet.
