# H058 proposal: additive block-diagonal/global low-rank SwiGLU

**Historical proposal; the tested recipe is now retired.**
[H061 language results](additive_block_lowrank_screen_results.md) fail all quality
gates despite H060 numerical/resource qualification. The text below preserves
the original reasoning, not the current allocation decision.


Original H058 hypothesis. H060 implemented and qualified this mechanism;
H061 subsequently rejected its tested language recipe. It is a known-family
comparator following H056's headwise failure and H057's inconclusive causal
diagnosis. The proposal itself allocated no language training and claimed no
verified novel primitive. The derivation below is historical.

## One concrete mechanism at the exact budget

Keep ordinary full-width SwiGLU, h=8d/3, and replace each of its gate, value and
down projections by

    W = S + L R,
    S: G=8 block-diagonal map,
    L: output_width by r, R: r by input_width, r=d/8.

Evaluate the block path and two dense low-rank maps separately, then add them.
The FFN remains W_down[SiLU(W_gate x) * (W_value x)]. There is no private
headwise FFN, router, new activation shape or learned intermediate expansion
shared by all projections. Each of the three projections has its own factors.
The global low-rank contribution enters before gating and can couple features
across the local blocks. The block path can retain full input/output rank when
its blocks have full rank; neither full rank nor conditioning is guaranteed
through training. Global information is not forced through a low-rank-only path.

For either direction d<->h, each projection has

    dh/G + r(d+h)

parameters. All three therefore have

    P = 3dh/8 + 3(d/8)(d+h) = 19d^2/8.
    P / (8d^2) = 19/64; FFN reduction = 45/64 = 70.3125%.

For d=384, h=1024, G=8, r=48, each projection has 49,152 block weights plus
67,584 factor weights. P=350,208 per layer, exactly matching BlockShuffle,
square headwise, overcomplete headwise and the h=304 narrow control. Eight layers
use 2,801,664 FFN weights and 9,099,648 total weights in the unchanged decoder.
Logical matrix forward FLOPs are 2P/layer/token, excluding additions, activation,
layout and actual GPU execution costs. h stays 1024, versus BlockShuffle's 2048.
Fewer activations are a hypothesis about utility, not a measured memory/speed gain.

The geometry generalizes to widths divisible by 24; r=d/8 and h=8d/3. For d=24,
r=3 and h=64, the same 70.3125% ratio holds. Initialization and optimizer groups
are deliberately not specified by this proposal: they need their own derived,
reviewed qualification before a language screen, not choices made after scoring.

## Analytic quadratic construction to check in the implementation

This family can represent the earlier three-output quadratic using the actual
allowed block and low-rank supports. The following is a constructive algebraic
proposal; numerical forward/Hessian checks are still required after implementation.
Use d a positive multiple of 24. Let local input width a=d/8 and hidden block
width b=h/8=d/3. All indices below are zero-based.

For each block g and local coordinate j<a, set rows gb+2j and gb+2j+1 of BOTH
sparse gate/value maps to +e_(ga+j)^T and -e_(ga+j)^T. These pairs consume 2a
rows per block. Their combined gated feature is x_i^2, using
SiLU(z)z + SiLU(-z)(-z) = z^2.

Because b-2a=d/12>=2, rows 2a and 2a+1 in block 0 are free. Set column 0 of
each gate/value L to +1 and -1 in these rows and row 0 of its R to all ones.
Other low-rank entries are zero. These two rows supply the signed pair for
(sum_i x_i)^2, without disturbing the coordinate pairs.

Set the sparse down map to zero. Use its first three L columns as output basis
vectors e_0,e_1,e_2. Row 0 of its R is .5 on both entries of every coordinate
pair; row 1 is .5*(i+1) on the coordinate-i pair; row 2 is .5 on the two global
sum entries. Other entries are zero. Since r=d/8>=3, this is admissible.
Outputs become .5*sum(x_i^2), .5*sum((i+1)*x_i^2), .5*(sum x_i)^2 and zeros.
It needs 2(d+1) active features, which fit h=8d/3 for d>=24.

This construction is outside the earlier square-mixer additive-headwise class
by the retained obstruction theorem. It does not establish class containment,
universal approximation at this fixed budget, novelty, learnability or NLL
advantage. H056 already demonstrates why that distinction matters.

## Prior art and alternatives

[SLTrain, Han et al. (2024)](https://arxiv.org/abs/2406.02214) uses sparse-plus-low-rank
weights for pretraining, with uniformly random fixed sparse support. Our proposed
support is block diagonal and our intervention is FFN-only. Those differences
must be explicit; this is not a faithful SLTrain reproduction. The paper's
reported systems results involve its own implementation and settings and cannot
be transferred to this proposal.

[Low-rank passthrough networks (2016)](https://arxiv.org/abs/1603.03116) already
studies low-rank-plus-diagonal parameterizations. The [2026 DLoR theory](https://arxiv.org/abs/2605.05659)
also studies expressivity with diagonal additions. These references rule out
claiming that adding a sparse direct path to low-rank factors is new. Their
width/depth approximation results do not certify this fixed-budget FFN's quality.

[Wei et al. (2024)](https://arxiv.org/abs/2406.16450), already in this repository's
literature audit, compares LowRank, BlockShuffle and BlockDense and introduces
self-guided training. Its BlockDense uses composition; the present proposal uses
addition. Temporary dense guidance remains a different training-budget comparison
and must include transient parameters/memory if investigated.

| Direction | Why consider it | Current decision |
|---|---|---|
| Additive block/global projection | Exact existing budget, ordinary SwiGLU width, direct local and global paths | Next mechanism to qualify |
| Random-support SLTrain comparator | Strong published sparse/global pretraining reference | Audit its implementation; account for indices/kernels and intervention scope |
| Temporary dense self-guidance | Published response to difficult structured training | Defer; extra training parameters/memory need a distinct fair protocol |
| Overcomplete norm/gauge repair | H057 observes scale effects but no sufficient cause | No repair or extra training earned |

## Next required work

Audit the primary method/code and possible equivalent block-support variants.
Choose one fixed initialization/optimizer policy before any quality measurements.
Implement the simple module in its own descriptive folder, prove actual counts
and the construction, verify independent dense forward/backward and numerical
calibration, then measure real full-model memory. Reuse existing native gate
recomputation only if justified and verified. No dense W should be secretly
materialized in training while claiming structured memory savings.

Only a passing qualification earns a separately frozen balanced language screen.
Retain both full controls, narrow and the strongest existing structured control;
add published-family comparisons where feasible. Multiple seeds, longer training,
scale, a broader corpus and novel-mechanism evidence remain later requirements.
Do not infer improvement from this proposal's arithmetic or its witness.

Implementation: [additive_block_lowrank_ffn](archive/h061_retired/src/additive_block_lowrank_ffn/README.md).
The analytic construction and initialization/update policy now have the scoped
numerical evidence in H060; the original proposal text above records the reasoning
before qualification. No empirical language or novelty conclusion follows.
