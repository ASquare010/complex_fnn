# Hypothesis: expanded nonlinear responses with a learned structured readout

Registered on 2026-10-04 before implementation/measurement. The coupled-curve
campaign continues unchanged. On TinyStories its candidate scores 2.626786,
self-gate control 2.598287, fixed shapes 2.628363, and plain SiLU 2.673912.
Restoring initial shapes at frozen weights changes the candidate to 2.628235
and self-gate to 2.599811. This suggests little inference dependence on the
learned slopes/offsets in these runs, not proof that training them is useless.
WikiText candidate 4.298236 also fails the dense quality margins; its remaining
controls are pending. No new GPU campaign or core-source changes overlap it.

The next question concerns the readout: do forcing several nonlinear responses
into one sum per coordinate and sharing their output direction discard useful
feature distinctions? Test a larger basis bank with separate learned readout
weights. The full input/output matrices remain; do not reintroduce rank-128
bottlenecks or claim the basis creates missing input information.

## Fixed computation and budgets

Width 512, independently in each of four Transformer layers:

```text
z = P x                                      # P: 512 by 512
s[k,i] = SiLU(a[k,i]*z[i] + b[k,i])           # k = 0,1,2
phi[i] = [z[i]*s[0,i], z[i]*s[1,i], z[i]*s[2,i],
          s[0,i],      s[1,i],      s[2,i]]   # six responses per coordinate
r = block_diagonal(B[0],...,B[15]) * phi    # 16 groups, each 32 by 192
FFN(x) = Q * (scale * (r - initial_mean))    # Q: 512 by 512
```

Each group reads all six responses from its 32 input coordinates. Six basis
responses have different readout weights, and the learned B mixes coordinates
within groups before global Q. Grouping concerns learned feature coordinates,
not tokens; P already sees the full residual input. No attention, token mixing,
external memory or routing is added. Readout is still structured: each group's
192 responses use only 32 intermediate output directions. Full global rank 512
is allowed but not guaranteed, and the groups have no known semantic locality.

Primary candidate fixes a=[0.75,1,1.25], b=[-0.5,0,0.5] across coordinates.
Secondary candidate learns those a/b arrays, initialized identically. Both train
P, B and Q from scratch. No pretrained/SVD basis reuse. Fixed shapes do not mean
a fixed activation transformation: B learns different combinations of the six
responses. These are smooth basis functions, not a new activation primitive.

Primary FFN weights: 4*(2*512^2 + 16*32*192) = 2,490,368, 70.54% fewer than full
SwiGLU. Learning a/b adds 12,288 weights: 2,502,656, about 70.40% fewer. Forward
projection work for both is 4,980,736 FLOPs/token; training projection work
14,942,208, plus scalar basis/reconstruction/reduction work. This is more work
than the tied curve recipe. FP32 grouped GEMMs below have different physical
costs from BF16 dense GEMMs; FLOP counts are not throughput or memory evidence.

## Initialization and controls

Initialize P std .02 and Q std .02/sqrt(2*layers) with the common named up/down
generators. Each B starts sparse: output coordinate i reads its own three
z*SiLU responses with weight 1/sqrt(3); all other entries start zero. Use the
self-gate curve's fixed initial_mean and scale from the same 128-node CPU FP64
Gaussian calibration (variance 512*.02^2), verified against 256 nodes. Under
that approximation it matches the expected full SwiGLU residual-update variance.
Correlations, distributions and logits are not generally matched to dense models.

The tied-readout mechanism control learns six coefficients per input coordinate
but sends those six responses only into that coordinate before Q. Represent its
B by actual sparse local coefficients, not a dense parameter full of masked
unused entries. Initialize product coefficients 1/sqrt(3), linear coefficients
zero. Same fixed templates/P/Q/calibration as the primary; count its smaller
2,109,440 trainable FFN weights. It tests the benefit of separate/grouped readout
against a learned per-coordinate basis mixture. A gain might reflect more
readout weights; equally sized ordinary dense references remain necessary.

Include an ordinary-autograd cached primary as execution control: same function,
weights, dtypes, grouping and initialization, storing the expanded bank. Do not
require the recomputed primary to beat its identical cached function's NLL.
Secondary-versus-primary comparisons concern template learning; do not credit
it merely because a frozen-template removal changes statistics.

Full references: native SwiGLU 1376, fused-kernel SwiGLU 1376, GELU 2064.
Compact references: fused-kernel SwiGLU 408 (2,506,752 weights), GELU 608
(2,490,368), GELU 611 (2,502,656). Select strongest ordinary compact across these
nearby matched budgets for either candidate; report the small size differences.
No per-corpus switching of architectures: report each candidate's independent
pass/fail on both corpora. If both survive, select the smaller mean normalized
NLL relative to strongest full across both corpora, before equal tuning.

## Execution, checks and sequence

Compute basis arithmetic and grouped B forward/backward GEMMs in FP32 on CUDA
BF16 runs, with autocast disabled only inside this operation. P/Q remain normal
BF16-autocast linears. Return the centered/scaled readout as BF16 once before Q.
Ordinary cached references use exactly those same dtype/layout choices. CPU
FP64 references use FP64 throughout this inner operation. No TF32 or lower
precision substitution selected from validation; record actual runtime settings.

The custom inner operation saves z, templates and B references; reconstruct phi
for dB, discard it before allocating dphi=B.T*incoming_readout_gradient, then
reconstruct scalar responses for dz and optional da/db. Use grouped layouts
directly where possible to avoid a second expanded-bank copy. For each branch,
with incoming product/linear basis derivatives tp,tl:

```text
dz += tp*(SiLU(u) + z*a*SiLU'(u)) + tl*a*SiLU'(u)
da += z*(tp*z + tl)*SiLU'(u)
db += (tp*z + tl)*SiLU'(u)
```

Include scale in the incoming readout derivative and correct group/layout
indexing. Return master matrix/template gradients in their original dtype.
This is exact recomputation up to the recorded arithmetic, not approximate or
straight-through differentiation. No higher-order-gradient claim.

Before profiling, verify independent CPU FP64 forward/all gradients and finite
differences, cached/recomputed and tied-readout equivalence at initialization,
complete-model causality/locality/counts/save-load/exact optimizer recovery.
Verify CUDA FP32/BF16 references, all matrix/template/input adjoints, incomplete
token tiles, BF16 rounding and interventions. Use Torch-owned allocations and
the calling Torch stream/context; validate shapes/dtypes before raw launches.

After the coupled-curve campaign is terminal, profile seven variants (three full,
both candidates, cached primary, tied control), three alternating rounds on each
corpus, 20 warmup/100 timed updates. Same common backbone/data/hardware as the
preceding screen. Record cold compilation, max allocated, median throughput and
reserved memory. Each candidate independently needs >=20% memory savings OR
>=1.2x throughput against every full, <=5% deterioration in the other resource,
and <6 GiB reserved. Retire resource failures before language; cached/tied are
controls, not substitutes that can qualify.

If any candidate passes, promote one readable complete Transformer, verify
production equivalence/shared-Trainer CPU recovery, then run both corpora at
seed 101, 2,000 updates / 4,096,000 targets, rate .0006, warmup 200, decay .1,
same windows and full eligible validation. Test loss stays unopened. Maximum
ten variants/corpus: full three, compact three, hardware-surviving candidates,
cached primary and tied control. Failed candidate recipes receive no language
run; cached primary remains explicitly an execution/control recipe. Both dense
quality margins and better NLL than tied control are required on both corpora.

At frozen checkpoints: replace learned B by its initial diagonal readout; zero
even input-coordinate response bundles (all six) before B; keep only product or
only linear responses; restore initial templates for the learned candidate;
replace FFN outputs by means from 32 training batches at seed 2027. Retain the
original calibration. Implement bank removals inside the inner operation, not
by zeroing the 512 readout latents. Report changed statistics and retrained
controls; destructive penalties alone do not prove semantic roles or generalization.
Survivors need equal three-seed/three-rate tuning, five fresh longer-budget seeds,
held-out test and sustained-resource confirmation under the unchanged contract.

## Prior work and originality

[Learned activations](https://arxiv.org/abs/1412.6830),
[KAN](https://arxiv.org/abs/2404.19756) and
[FastKAN/RBF bases](https://arxiv.org/abs/2405.06721) already use learned scalar
functions or nonlinear basis representations. The recent
[KAN small-language-model study](https://arxiv.org/abs/2607.15525) also tests
grouped bases and reports no consistent advantage over strong MLPs across its
tested quality/benchmark/latency comparisons. Those results neither validate
nor rule out this allocation; they are a warning against declaring a discovery
from a basis bank or a small validation gain. This uses shifted SiLU/product
bases, full linear P/Q and a block readout rather than the cited spline/RBF
designs, but that difference alone does not establish originality. Fixed bases,
grouped matrices, structured readouts and recomputation are established. Verify
closest architectures again before claiming novelty; earn a practical result
through controlled repeated comparisons and independent confirmation.
