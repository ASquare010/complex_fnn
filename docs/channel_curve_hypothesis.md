# Hypothesis: full-width mixing with coupled learned activation branches

Registered before implementation/measurement on 2026-10-04. The latent campaign
continues unchanged; no new GPU experiment or core-source edits overlap it.
TinyStories wide recomputed/cached NLL is 2.726152, thin is 2.725223, and
initial-scale-calibrated thin is 2.701581. All miss the dense quality margins.
These observations motivate removing the rank-128 restrictions; they do not
prove those restrictions caused the deficit. WikiText and removal evidence will
still be preserved when that campaign finishes.

## Question and computation

Can a compact FFN spend almost all its weights on full-width input/output mixing,
while using several cheap learned nonlinear interactions per coordinate to
compensate for the absence of a large, independently projected hidden bank?

At width d=512, three branches per coordinate:

```text
z = P x                              # P: 512 by 512
u[k,i] = a[k,i] z[i] + b[k,i]
v[k,i] = c[k,i] z[(i + shift[k]) mod 512] + e[k,i]
r[i] = sum_k SiLU(u[k,i]) * v[k,i] / sqrt(3)
FFN(x) = Q * (scale * (r - initial_mean))  # Q: 512 by 512
```

Shifts are fixed at [1, 17, 129], not chosen from validation. They affect feature
coordinates within one token, never sequence positions. P learns the feature
basis; Q mixes the resulting nonlinear coordinates into residual updates. All
four coefficient arrays a,b,c,e have shape [3,512] and are learned. Branches can
learn different thresholds/slopes, while their gates depend on different basis
coordinates. No softmax, attention, dynamic routing, token mixing or external
memory is added. Attention and embeddings stay the existing common backbone.

P/Q have no imposed low-rank factorization; that does not guarantee learned full
rank or retention of useful information. Three nonlinear branches share each
readout direction, limiting expressivity compared with independently projected
detectors. Fixed shifts are a sparse interaction graph, not an optimal learned
graph or a new mathematical primitive. Features may be redundant or unstable.

Four-layer weights: 4*(2*512^2 + 4*3*512) = 2,121,728, 74.90% fewer than
8,454,144 full SwiGLU FFN weights. P/Q forward projection FLOPs total 4,194,304
per token; scalar activation/coefficient/reduction work is additional. Ordinary
projection backward gives 12,582,912 total training projection FLOPs/token.
Recomputing scalar branches adds scalar work, not extra projection GEMMs. Report
all measured resource costs rather than treating those projection counts as speed.

## Initialization and controls

P has independent named normal initialization std 0.02; Q std
0.02/sqrt(2*layers), matching the common dense input/residual initializer.
Initialize a branch slopes [0.75,1,1.25], offsets [-0.5,0,0.5], c=1 and e=0,
broadcast across coordinates. These are fixed choices before measurement.

Fix initial_mean and scale from deterministic 128-point Gauss-Hermite integration
in CPU FP64, with z variance 512*0.02^2=0.2048. Coupled initialization has mean
zero under independent Gaussian coordinates and second moment
variance(z)/3 * sum_k E[SiLU(a_k z+b_k)^2]. The self-gate control below uses the
mean and centered variance of z*sum_k SiLU(a_k z+b_k)/sqrt(3). Set each mode's
scale so its centered output-coordinate variance matches
(1376/512)*variance(z)*E[SiLU(z)^2], the full SwiGLU feature-bank variance per
512 readout directions under that same approximation. These fixed constants
are not trained and are never estimated from validation. Check quadrature
convergence against 256 nodes before profiling. Record constants and sampled
initial moments. Real P outputs are correlated; this calibration does not match
all distributions, gradients or initial logits, or prove equal training dynamics.

Three mechanism controls:

* **Self-gate learned curves:** use z[i] for every gate instead of shifted z.
  Same learned coefficient/matrix budgets, but no feature exchange inside the
  activation. Its separate fixed initial-mean/variance calibration is above.
* **Fixed activation shapes:** coupled shifts, a/b fixed to the same initial
  templates while c/e remain learned. Same initial function and scale as the
  candidate, fewer trainable scalar weights. This isolates learning of branch
  slopes/thresholds, without asserting a perfectly matched training distribution.
* **Plain unary:** ordinary bias-free SiLU dense FFN, hidden 512, common named
  initialization. Include it in strongest ordinary compact language selection.

Ordinary compact controls: kernel SwiGLU hidden 346 has 2,125,824 weights
(0.193% more); GELU hidden 518 exactly matches 2,121,728. Full controls are
native SwiGLU 1376, fused-kernel SwiGLU 1376 and GELU 2064. The candidate must
pass both dense quality margins and beat self-gate/fixed-shape controls on both
corpora. A useful gate must not be inferred merely from a destructive removal.

## Implementation and registered sequence

Start with an independent ordinary-autograd CPU reference and staged complete
Transformer. A custom activation saves z and coefficient references, reconstructs
the three scalar branches backward, and accumulates local and inverse-shifted
neighbor gradients. CPU FP64 finite differences must verify z and all a/b/c/e
gradients; verify self-gate accumulation includes both paths into z.

For incoming derivative t for a single unnormalized branch, s=SiLU(u):

```text
dz_local = t * a * SiLU'(u) * v
dz_neighbor = t * s * c
da = t * z * SiLU'(u) * v
db = t * SiLU'(u) * v
dc = t * s * z_neighbor
de = t * s
```

Include scale/sqrt(3), inverse permutation for neighbor derivatives and token
reductions for coefficients. CUDA kernels perform activation and derivative
arithmetic in FP32 for BF16 projections; cast summed input derivatives once,
keep master coefficient gradients FP32. Do not silently use approximate gradients.
Verify CUDA FP32/BF16 equations/gradients with recorded tolerances and errors;
complete-model causality/locality, counts, save/load and exact optimizer recovery.
No higher-order gradient support is claimed. No core-source changes while a
previous GPU campaign is live. CPU preparation may proceed separately.

After the latent campaign is terminal, profile seven variants (three full, candidate,
self-gate, fixed-shape, plain): three alternating-order rounds on each corpus,
20 warmup and 100 timed updates, common width 512/four layers/16 heads/context
256/vocabulary 4096/batch eight/BF16/four threads/RTX 4070 Laptop. Record cold
warmup, maximum allocated peak, median throughput and reserved memory. Require
>=20% less allocated VRAM OR >=1.2x throughput against every full control, with
<=5% deterioration in the other resource and <6 GiB reserved. Retire the fixed
candidate before language if it fails; controls cannot qualify in its place.

If hardware passes, promote one readable complete Transformer, verify production
equivalence and shared-Trainer CPU recovery, then run nine language variants on
both prepared corpora: three full, two matched compact, candidate, self-gate,
fixed-shape and plain. Seed 101, 2,000 updates / 4,096,000 targets, rate 0.0006,
warmup 200, decay 0.1; identical windows and full eligible validation. Test loss
stays unopened. Require <=1% loss cost versus strongest full and >=1% gain versus
strongest ordinary compact, plus beating both mechanism controls, on both corpora.

At frozen trained weights, change shifts to zero while retaining the candidate's
original calibration (do not substitute the self-control's normalization),
restore a/b initial shapes while retaining trained c/e, zero even branch-summed
coordinates before Q, and replace FFN outputs by means from 32 fixed training
batches at seed 2027. Report statistics and loss penalties without interpreting
them alone as semantic roles. Surviving recipes require equal three-seed/three-rate
tuning, five fresh confirmation seeds and independent longer-budget validation/test
confirmation with sustained resources, as in the existing acceptance contract.
Different branches/shifts/calibrations require a new registration before measurement.

## Prior work and originality

[Learned activations](https://arxiv.org/abs/1412.6830) already learn neuron-specific
nonlinearities; [KAN](https://arxiv.org/abs/2404.19756) uses learned spline functions
on edges. This proposal retains linear mixing and uses neither the KAN spline
architecture nor its theorem as a quality guarantee. [GLUs](https://arxiv.org/abs/2002.05202)
already combine features multiplicatively; [gMLP](https://arxiv.org/abs/2105.08050)
uses gating in a different spatial mixing design. Those results do not establish
this token-local FFN's performance. Full mixing, nonlinear bases, multiplicative
interactions and activation recomputation are established ideas. Originality of
this particular controlled combination remains unverified; a successful practical
result must be supported by repeated comparisons, not a new name.
