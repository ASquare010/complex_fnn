# Down-projection conditioning intervention

Frozen before evaluation. The width384/layer8 BlockShuffle seed17 800-step
checkpoint has worst projection condition number 30134 in layer6.down, while
full SwiGLU's maximum is 17.28. This is a projection-conditioning concern;
finite measured loss-direction gradients do not demonstrate a training failure.
Before training a constrained architecture, isolate the role of weak directions.

## Fixed treatments and controls

Use the original checkpoint without updating it. In each of the eight down
projections, the second factor is eight square 48x48 blocks. Keep every other
weight bitwise unchanged. For each layer independently, decompose these blocks
in CPU FP64, replace each singular value s by max(s, alpha*s_max), then apply
one common scalar so the combined factor retains its original Frobenius norm.
Here s_max is the largest singular value across ALL eight blocks in that layer.
Test only alpha=.01, .1 and 1. The last is a scaled orthogonal factor.
Cast transformed factors back to their original FP32 storage before inference.

Evaluate eight cases: unchanged checkpoint, alpha=0 SVD reconstruction,
three floors, and three additive random-direction controls. Each random control
has the SAME per-block Frobenius perturbation norm as its floor treatment;
use fixed seed1026+layer index and reuse directions across severities. Random
controls do not also preserve final factor norm, so they isolate perturbation
magnitude only. This is one noise direction, not statistical random-control
replication. No hyperparameter selection or checkpoint fine-tuning is allowed.

Every case starts from the original weights. Record per-factor spectra, change
norms, full down-projection spectra, and one CPU FP32 fixed-batch loss-direction
backward audit. Evaluate all 32768 fixed validation targets with unchanged
batch16/context128 CUDA BF16 and the frozen data/tokenizer. The unchanged case
must reproduce archived NLL within .0002; reconstruction must differ from it by
<=.0001 NLL and <=1e-6 in weight elements. Record finite outputs and gradients.
Archive source/config/checkpoint/data hashes and the transformed factors.
No timing claim is made by this diagnostic; the factor shapes and inference
operations are identical, but unchanged FLOPs alone do not prove equal speed.

A floor earns further consideration only if all values are finite, worst full
down-projection condition number improves >=10x, and NLL degrades <=.1% relative
to the unchanged checkpoint. Report all cases regardless of this decision.
This threshold is local diagnostic evidence, not the overall research gate.
A successful post-training edit does not show that training with this constraint
improves optimization or that the complete FFN has a nonvanishing Jacobian.

## Scoped proof

Let A be the block-diagonal square factor, R the rectangular first factor,
and W=P_out^-1 A P_mid R. In exact arithmetic, flooring and common rescaling
produce A_alpha with condition number <=1/alpha for alpha>0, provided
the original factor has nonzero Frobenius norm; common rescaling
cancels from the ratio and preserves the Frobenius norm. For alpha=1 all
singular values equal their original aggregate RMS, so A_1=cQ with Q orthogonal.
Consequently the nonzero singular values of W_1 equal c times those of R.
For full-row-rank R, sigma_min(W_alpha)>=sigma_min(A_alpha)*sigma_min(R)
and sigma_max(W_alpha)<=sigma_max(A_alpha)*sigma_max(R). Thus its nonzero
condition number is bounded by cond(R)/alpha. Permutations are orthogonal.

These statements concern real-valued matrices before FP32/BF16 rounding.
The finite-precision reconstruction and inequalities are tested numerically.
The rectangular factor can still be ill-conditioned; the gated nonlinear FFN
and full residual network have no corresponding global lower Jacobian bound.
Clamping eigen-directions is established linear algebra, not a novelty claim.
Structured orthogonal parametrizations already appear in
[BOFT](https://arxiv.org/abs/2311.06243) and
[Group and Shuffle](https://arxiv.org/abs/2406.10019), so a later orthogonal
BlockShuffle variant would require a specific additional novelty argument.

Interpretive clarification after the initial diagnostic: rectangular down
projections necessarily have an input nullspace. All condition numbers and
lower singular-value bounds above refer to the nonzero spectrum/full-row-rank
output space, not injectivity on every input direction. The archived original
plan remains in the initial diagnostic source.zip.
