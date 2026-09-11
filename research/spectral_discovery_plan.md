# H103: can training-label energy reveal missing nonlinear directions?

Previous goal turn: progress. H101/H102 produced audited memory evidence and a
reproduced numerical failure. The full VRAM/parameter/quality objective is active.
This next experiment addresses H100's missing cubic feature discovery directly.

## Hypothesis and local proof

Let X be standard Gaussian, Q have orthonormal columns, and Y=C f(Q^T X), with
square-integrable label energy and the moments needed below finite. Put
e=||Y||^2/E||Y||^2. Consider centered weights w=e-Ee (raw), or
w=e/(1+e)-E[e/(1+e)] (bounded). The population matrix is

    M = E[w (X X^T - I)] = Q E[w (Z Z^T - I)] Q^T.

Proof: decompose X=QZ+U, with independent centered Gaussian U in Q's orthogonal
complement. Cross terms vanish by independence. E[w UU^T]=E[w]E[UU^T]=0.
This proves only that M's range is contained in the relevant subspace. It does
not prove rank, a positive eigengap, finite-sample recovery or trainability.

For independent normalized Hermite features of degree k>=2, f_i=He_k(Z_i)/sqrt(k!),
the raw matrix is Q diag(2k A_ii/tr(A)) Q^T, A=C^T C. Orthogonality and
Z He_k = He_(k+1)+k He_(k-1) give E[f_i^2 (Z_i^2-1)]=2k; off-diagonal terms
vanish because E[Z He_k(Z)]=0. Thus the cubic target has a nonzero second-order
label-energy signal even though E[YX]=0. This is a population calculation under
stated Gaussian assumptions, not a novelty or finite-sample theorem.

The bounded weight has magnitude <=1 after centering. That controls weights,
not the unbounded Gaussian XX^T factor. It changes the eigenspectrum and need
not preserve the raw eigengap. Test it rather than assume robustness helps.

## Limits that matter to the overall goal

Constant-norm labels (including ordinary one-hot next-token labels) give e=1
and M=0 exactly. This cannot directly initialize an LM FFN from label energy.
If successful, its possible uses are regression or teacher-output/residual
distillation; neither is validated by this diagnostic. Any subsequent teacher
cost, preprocessing storage and label accesses must be accounted for. No hidden
teacher basis, latent coordinates, held-out labels or task name enter inference
of the subspace. The true basis is used solely for post-estimation diagnosis.

## Frozen cheap screen

Three independent Gaussian datasets: dimensions384, hidden rank16, N65536;
input/basis/output seeds (10300+s,10400+s,10500+s), s in17/29/43. Reuse only the
four target equations from H100: quadratic, cubic, product, piecewise. Use
training-only per-output scaling. No neural updates and no validation selection.

Estimate a rank32 input subspace using raw and bounded label-energy moments.
Controls: input PCA, an independent random rank32 subspace and bounded energies
permuted across training rows. Rank32 is fixed before results, not selected using
the true rank. CPU double eigensolve; CUDA double streaming moment accumulation
with chunks2048; one GPU, four CPU threads. Log wall time and peak CUDA allocation
including preprocessing. Store matrix/eigenvalue/basis hashes and source/config.

Before the screen, verify exact Gaussian quadrature moments for degrees2/3;
streamed versus dense centered moments on a tiny double case; invariance under
output orthogonal rotations and rescaling; constant-norm-label zero signal;
orthonormal estimated basis and invalid input rejection. All tolerances1e-10
absolute/relative, except eigenvector orientation is compared via projectors.

Score recovery as ||Q^T B||_F^2/16. Random expected recovery is32/384=0.08333.
Also report all principal-angle cosines and the smallest squared cosine, so
recovering only a few easy directions cannot masquerade as full recovery.

A method earns a fitting screen only if cubic recovery>=0.30 on all three
datasets, exceeds the permuted-label recovery by>=0.15 each, and its smallest
squared cosine is>=0.01 each. The shuffled-label control must stay<=0.15.
These are diagnostic allocation gates, not claims that0.30 recovery is sufficient
for a good model. Do not select rank or weight transform by reporting-set scores.
If neither qualifies, close these fixed moment recipes without neural training.

## Prior work checked

- [Second-order Stein multi-index estimation, NeurIPS2017](https://papers.neurips.cc/paper/7190-estimating-high-dimensional-non-gaussian-multiple-index-models-via-steins-lemma.pdf)
  establishes response/score moments and truncation precedents.
- [Gaussian multi-index gradient flow, AISTATS2025](https://proceedings.mlr.press/v258/simsek25a.html)
  studies directional difficulty without low-order Hermite components.
- [The Generative Leap](https://arxiv.org/abs/2506.05500) and
  [Neural Networks Learn Generic Multi-Index Models](https://arxiv.org/abs/2511.15120)
  give stronger spectral/layerwise-learning precedents under their assumptions.
  This simple label-energy estimator does not claim their guarantees or novelty.
