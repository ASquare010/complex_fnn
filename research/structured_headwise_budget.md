# Structured mixers can fund wider headwise nonlinearities

**Status: analytical proposal only. No registered model or training result.**
H047's router-free headwise recipe saved memory but failed quality. Its two
dense mixers spend 84.2% of its FFN weights before the nonlinear subnetworks.
Replacing them with structured maps would move weights into those subnetworks.
This is a budget tradeoff to test after the frozen longer-duration controls;
it is not a conclusion that mixer cost caused H047's loss.

## Mechanism and exact count

Let q = A*x, split q into H heads of width dh=d/H, and define

    g_h(q_h) = D_h [SiLU(U_h*q_h) * (V_h*q_h)]
    F(x) = B * concat_h(g_h(q_h)).

Each private U/V has shape de by dh; D has shape dh by de. Two square
BlockShuffle mixers each use two grouped d by d factors with G groups.
A fixed channel permutation has no learned weights. Counts per layer are

    dense-mixer headwise:       P = 2*d^2 + 3*d*de
    structured-mixer headwise:  P = 4*d^2/G + 3*d*de.

For d=384, G=8, H=6, dh=64 and de=240, the structured proposal uses
73,728 mixer weights plus 276,480 private weights: **350,208 per layer**.
Across eight layers this is **2,801,664 FFN weights**, exactly matching plain
BlockShuffle and calibrated narrow, and **70.3125% below full SwiGLU**.
Total Transformer weights would be 9,099,648 with the frozen non-FFN dimensions.

| Recipe | Mixer weights/layer | Private weights/layer | Head input width | Private hidden/head | Total hidden features |
|---|---:|---:|---:|---:|---:|
| H047 dense mixers | 294,912 | 55,296 | 16 | 48 | 1,152 |
| Proposed structured mixers | 73,728 | 276,480 | 64 | 240 | 1,440 |

At fixed d/de, head count does not change the private parameter total 3*d*de,
but it changes both head input dimension and H*de intermediate features. Thus
parameter count alone cannot predict feature capacity, memory or kernel speed.
The proposed stages use four grouped mixer contractions plus three private
contractions; H047 uses two dense plus three private contractions. Logical
multiply-add work remains 2*P per token, but actual memory and speed are unmeasured.

At the same budget, G=4 would permit de=176 and G=16 would permit de=272.
Those are arithmetic alternatives, not an authorized sweep or selected winners.
G=8/de=240 is a single understandable hypothesis; adding routers and activation
banks at the same time would obscure what any quality change means.

## What is and is not proven about expressivity

Each head now sees 64 learned coordinates instead of 16 and gets a wider
private nonlinear map. The input/output matrices, however, belong to a
restricted structured family instead of the dense family. These simultaneous
changes do not establish containment of either complete FFN family in the other.
Neither hidden-feature count nor parameter count proves expressivity dominance.

For fixed q coordinates, every cross-head mixed second derivative of the
additive head map is still zero. Arbitrarily flexible learned scalar activations
inside each head do not change that fact. Learned A changes coordinates and
stacked layers can compose interactions; the statement is not a full-network
impossibility. See the [existing proof](archive/retired/src/multihead_ffn/headwise.md).

There is no automatic gradient guarantee either. For one head, with q restricted
to ||q||_2 <= R and matrix spectral norms, SiLU gives the sufficient upper bound

    ||J_g(q)||_2 <= (2 + 1/e) * R * ||D||_2 * ||U||_2 * ||V||_2.

To derive it, differentiate both factors of SiLU(Uq)*(Vq), use
||diag(v)||_2 <= ||v||_2, |SiLU(z)| <= |z| and
sup |SiLU'(z)| <= 1+1/e. For ||x||_2 <= R_x, the chain rule gives

    ||J_F(x)||_2 <= (2 + 1/e) * R_x * ||B||_2 * ||A||_2^2
                    * max_h(||D_h||_2 * ||U_h||_2 * ||V_h||_2).

The second factor of ||A|| comes from ||q_h|| <= ||A||*||x||. This is only an
upper bound on a bounded input set; weights can grow, and base J_g(0)=0.
It gives no positive lower bound and does not prove easy optimization.

The [affine correction](archive/retired/src/learnable_activation_ffn/model.md) can add a
nonzero origin Jacobian with few weights, but did not earn a material same-rate
three-seed gain. A learnable-activation extension of this proposal would need
a separate simpler-versus-richer control after the base itself earns promotion.

## Initialization and optimizer questions

For square factors, the existing BlockShuffle initializer can make A exactly
orthogonal and B a scaled orthogonal map in real arithmetic. Its product gain
is 0.02*sqrt(d)*residual_scale. Choosing residual_scale=1/(0.02*sqrt(d)) for A
and c/(0.02*sqrt(d)) for B gives norms 1 and c=1/sqrt(2*layers).
This is exact initialization algebra, not an implemented or maintained constraint.
It supplies no trained spectral or whole-network Jacobian guarantee.

The existing headwise Gaussian calibration would use private U/V standard
deviation 0.02*sqrt(H), private D standard deviation 0.02*sqrt(h_full/de),
and B norm c. For H=6/de=240/h_full=1024 these are about 0.048990 and 0.041312.
Isotropic Gaussian q is invariant under orthogonal A; this preserves the
assumptions of the prior initialization calculation. Output RMS still needs
an independent numerical check. Factor-specific learning rates and decay change
the represented-map update, so the optimizer treatment must be frozen and
explicitly controlled before any training comparison.

## Prior art and decision boundary

Products of block-diagonal matrices with permutations are established in
[Monarch](https://arxiv.org/abs/2204.00595); this is not a new structured linear
primitive. [Monarch Mixer](https://arxiv.org/abs/2310.12109) already studies
structured architecture replacements across language and vision models.
[Flash Multi-Head FFN](https://arxiv.org/html/2512.06989v1) supplies prior art for
headwise nonlinear subnetworks, parallel aggregation and specialized execution.
Combining these components does not by itself establish novelty. The exact
small-budget allocation above is a local hypothesis, not a literature priority claim.

Before a future screen: implement only the base in the existing architecture
folder; independently check counts, FP64 forward/all derivatives, initialization
norms, native BF16 behavior and GPU memory; preserve old-model snapshots. Then
freeze equal new tuning effort against appropriate full/narrow/BlockShuffle and
headwise controls. A failed base should not be repaired by stacking activation
complexity without diagnosing the failure. H051 and H052 subsequently completed and failed their promotion gates.


The subsequent [square-mixer Hessian obstruction](headwise_hessian_obstruction.md)
gives a specific vector quadratic that remains outside this single-module
family even with learned dense mixers and arbitrary head functions. This
sharpens the expressivity limit; it does not diagnose the observed NLL. That
note also derives an untrained, equally parameter-matched overcomplete option
outside the proof's square-mixer assumption. No architecture is registered or
promoted by either calculation.

The rectangular alternative now has a [standalone implementation and explicit
constrained-factor witness](archive/retired/src/multihead_ffn/overcomplete.md). H053 qualifies
its local numerics and initialization scale, with a 20.37% isolated memory cost
over full SwiGLU. It has not been registered or tested for language quality.
