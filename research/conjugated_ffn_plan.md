# H143: conjugated shared residual FFN — bounded falsification screen

Previous turn: progress (H142 batch-scaling qualification). Return to structural
parameter efficiency. H006 shared complete FFNs across existing Transformer layers;
H106/H107 added an affine path and lost against narrow controls; H100/H104's cubic
products and spectral starts failed promotion. Do not reopen those recipes.

## Hypothesis and exact scope

For row vectors, let R(z)=GELU(z U^T+a) D^T+b, eta=1/sqrt(2).
Set z0=x; z1=z0+eta R(z0); z2=z1+eta R(z1 P) P^T; F(x)=z2-x.
P is a fixed permutation, sampled independently of targets. Share U,D,a,b.
Compare P=I (ordinary recurrence), independent second-step weights (untied
permuted), narrow one-step GELU, wide GELU and near-MAC-matched SwiGLU.

A centered matrix of F outputs has rank at most rank([D,P D])<=2h, whereas
same-basis recurrence remains bounded by h. Bias contributions are fixed vectors
and disappear on centering. The column-vector convention uses P^T D instead;
both state the same orbit-of-output-subspace bound. For d=2h, coordinate-half
swap, U selecting the first half and D inserting into that half, F applies
eta*GELU coordinatewise to both halves: full output rank can occur. This is an
elementary rank witness, not an approximation/learning theorem or strict class
containment. Recurrence cannot by itself evade the shared output-span limit.

If L bounds R's Jacobian norm, each residual step has singular values between
1-eta*L and1+eta*L when eta*L<1. The two-step state Jacobian inherits product bounds.
We do not constrain L here, and F=z2-x does NOT inherit a positive lower bound.
No nonvanishing-gradient guarantee. Tests check gradients rather than assuming it.

## Prior art (verified primary sources)

[Universal Transformers](https://arxiv.org/abs/1807.03819) use recurrence over depth.
[ShaResNet](https://arxiv.org/abs/1702.08782) shares residual-network weights.
[Shuffling RNNs](https://arxiv.org/abs/2007.07324) use fixed permutations in recurrence.
[Permuted-feature compression](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cdt2.12060)
also links permutations and compressed fully connected layers. These precedents
preclude claiming weight sharing or permutation mixing is new. The exact proposed
combination's novelty is unresolved; this screen is about utility, not a novelty claim.

## Frozen experiment

Dimension384; full GELU h384, full SwiGLU h256, all narrow branches h192.
Include biases and count every trainable scalar and fixed permutation buffer.
Shared two-step matrix MACs equal wide GELU (4dh versus2d^2); untied has the same
MACs and approximately wide parameter count. Permutation traffic and extra launches
still cost time. Exact counts and observed allocation must be reported.

Three independent dataset/model seeds17,29,43. Each has8192 IID normal inputs:
4096 training,2048 validation,2048 reporting. Three independent target families:
random orthogonal linear map; random wide GELU teacher; cyclic cubic products
then an independent orthogonal output mix. Fit input/target statistics using only
training rows; all arms use exactly the same data and minibatch-index stream.
Targets/teachers and permutations use distinct RNG streams. No teacher parameters
or reporting labels initialize students. All six recipes use one fixed LR1e-3,
AdamW betas(.9,.95),zero decay,clip1,batch128,300 updates,FP32/TF32off. This is a
single-rate recipe screen, not a conclusion about each architecture's optimum.

Budget6 arms x3 tasks x3 seeds x300=16,200 optimizer updates. Store datasets,
models, per-step losses/gradient norms, initial/final scores, finite flags, exact
parameter counts, whole-job CUDA peaks including scoring, training timer samples
and readout-subspace ranks. CPU-resident datasets; CUDA use is sequential; release
all models/Adam state between runs. No fresh language-model training or tests.

Preflight: FP64 finite differences of input and tied weights, independent explicit
matrix formulation, equal-MAC/count checks, full-rank swap witness and same-basis
rank obstruction. Stop before fits on correctness failure. Baseline learned quality
must beat zero-predictor reporting MSE by20% on every task/seed before a candidate
can be promoted; otherwise that task is an inconclusive learning probe. Candidate
must use at least40% fewer parameters than both wide controls, beat narrow and
same-basis tied controls on every task/seed, and have reporting MSE<=1.05 times
both wide controls, CUDA peak<=0.90 of wide GELU and median training update<=1.15
of wide GELU. Every seed counts. No LR search, best checkpoint or discarded runs.
These stringent gates allocate stronger tests only; failing closes this fixed
recipe, not all possible conjugated networks. Scalar activations ReLU/LeakyReLU/
PReLU/SiLU and additional modern baselines remain mandatory for any later broad
claim; this mechanism screen cannot support such a claim by itself.

Run order is frozen in code: rotate the six-arm list by dataset index modulo six.
Training timing includes CPU batch gathering, transfer, gradient clearing and the
complete synchronized optimizer update; the first20 timings are warmups. Reporting
scoring and matrix-rank diagnostics are excluded from timing but included in CUDA
peak accounting. No higher-scale latency inference follows from this local test.
