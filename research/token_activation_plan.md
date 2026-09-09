# H067: token-conditioned residual activation qualification

Frozen before implementation or measurements. Previous turn H065/H066 made
PROGRESS, qualified numerical behavior and rejected the staged memory repair.
Do not extend that checkpoint search. Keep the active shortlist unchanged until
a new hypothesis earns promotion.

## One candidate and its necessary simpler control

Let u=U(x), v=V(x) be the existing BlockShuffle projections. Define

    alpha(x) = 0.25*tanh(w^T*x + b)
    F(x) = D[SiLU(u) * (v + alpha(x)*SiLU(v))].

Use one scalar router per token per layer, shared across hidden channels. The
static control sets w absent and learns only b. Initialize every new parameter
at zero, recovering native SwiGLU. Preserve existing projections, channel mixing
and down-projection scaling. This is a residual, baseline-preserving adaptation
of one-sided token-dependent activation mixing; it is not an original MoA claim.
No other activation, router count, amplitude, normalization or factorization search.

Routing uses FP32 (FP64 for double inputs), then casts alpha to the projected
activation dtype before hidden-width arithmetic. The hidden SiLU/product remains
native BF16 on CUDA. Standard non-reentrant product checkpointing is applied
equally to the candidate, static control and local plain reference. This local
reference uses the checkpoint gate policy, not an asserted reproduction of the
historical custom native backward's complete training trajectory.

At d384/h2048/G8/L8, dynamic adds 385 parameters/layer, 3,080 in total; static
adds one/layer, eight total. FFN totals are 2,804,744 and 2,801,672 versus baseline
2,801,664. Both must remain at least 70% below the full 9,437,184 FFN reference.
No active factory variant, recipe, default or source change is needed for this
qualification: put the prototype beside its frozen experiment artifacts.

## Mathematical deliverables and limits

Derive baseline/static containment, the 0.75..1.25 multiplicative value envelope,
and a positive partial derivative bound with the router treated as independent.
Explain why none is a whole-network nonvanishing-gradient or learning guarantee.
Derive the router's generally nonzero learning signal at zero initialization.

Construct one structured FFN output exactly equal in real arithmetic to
SiLU(x1)*x2 + 0.25*tanh(x3)*SiLU(x1)*SiLU(x2), with the actual d384/h2048/G8
projection paths. On x1=x2=1 this becomes a nonzero constant plus a nonzero
multiple of tanh(x3). Audit a meromorphic-residue argument separating this exact
function from any finite single SwiGLU FFN, even allowing affine projections,
and from finite exact-GELU FFNs. State all assumptions. This is not a uniform
approximation lower bound, a parameter-efficiency learning theorem, a result for
stacked Transformers or a priority claim. Do not assert strict separation from
the static control without a separate proof.

## Fixed 21-check qualification

- 12 zero-router paired cases: static/dynamic x CPU FP32 or CUDA BF16 x seeds
  17/29/43. CPU shape [2,7,24], hidden48/groups3; GPU shape [16,128,384],
  hidden2048/groups8. Name-derived existing initialization, identical projections,
  independently seeded synthetic inputs and incoming gradients. Compare output,
  input gradient and every shared projection gradient bitwise. New router
  gradients must be finite and nonzero; do not demand a nonzero signal for every
  possible parameter/input configuration.
- 4 nonzero-router cases: static/dynamic x CPU/CUDA, seed17. Set b=.4 and dynamic
  w to linspace(-.02,.02,d). Compare eager versus product-checkpoint outputs and
  all gradients bitwise with unchanged coefficients. No optimizer updates.
- 5 mathematical/software checks: exact parameter counts; independent FP64
  gradcheck of inputs/router; sampled envelope and partial derivative bound;
  the actual full-size structured witness and its router-coordinate derivative;
  numerical residue-identity spot checks supporting, not replacing, the proof.

Use four CPU threads, one GPU process, UV with existing extras. Save source and
input/state hashes, per-case equality/errors/nonzero-gradient evidence, and process
records. Complete all 21 checks unless a runtime failure makes continuing unsafe.
No automatic retry or source change after observed scientific failures. Preserve
all failures. Existing active source/test/configuration bytes must match H066.

## Decision

Passing earns only a separately frozen function-fitting/resource comparison of
plain, static and dynamic forms. No full language run is authorized by this local
qualification. A practical candidate still needs actual memory/runtime, matched
learning controls, both full quality limits, calibrated narrow, replicated longer
training, convergence, scaling, broader data and a novelty audit. Mathematical
noncontainment alone does not satisfy the research objective. Any failure closes
this tested definition without automatically varying the amplitude or dictionary.

Primary prior art: [MoA/LA](https://arxiv.org/html/2605.26647v1) and its one-sided
mixtures; [conservation-law pole analysis](https://arxiv.org/html/2606.17816v1),
Appendix C.1, as precedent for treating SiLU meromorphically. Existing static
rational/affine results and failed FlashMHF/headwise screens remain authoritative.
