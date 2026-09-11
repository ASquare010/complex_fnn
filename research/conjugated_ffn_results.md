# H143: shared residual FFN with permutation conjugation

**REJECT fixed shared-permuted recipe.** Independent audit passed: **True**.
All 54 fits and 16,200 updates are retained. No active model/default changed.

## The hypothesis and what was proved

Use one narrow GELU branch twice inside a residual transformation, permuting the
coordinates before its second use and undoing that permutation on its output.
The two calls share every weight and bias. Compare same-basis sharing, independent
second-step weights, narrow GELU, wide GELU and wide SwiGLU.

For hidden width h, centered outputs of same-basis reuse stay in one h-dimensional
readout subspace. Permutation conjugation permits the sum of two readout subspaces,
with rank at most 2h. A coordinate-half-swap construction attains full rank at
h=d/2. Independent explicit-matrix evaluation and FP64 finite differences of both
inputs and tied weights pass. This elementary result removes one obstruction;
it proves neither efficient learnability nor a superset of every baseline family.
FP32 fitted-output ranks below are numerical diagnostics, not proofs.

A conditional residual-state Jacobian bound requires a sufficiently small branch
Jacobian norm. No such constraint is enforced here, and subtracting the input to
produce the FFN output removes the residual state's lower-bound guarantee. There
is no claim of eliminating vanishing or exploding gradients.

| Recipe | Trainable scalars | Matrix MACs per token |
|---|---:|---:|
| full_gelu | 295,680 | 294,912 |
| full_swiglu | 295,808 | 294,912 |
| narrow_gelu | 148,032 | 147,456 |
| shared_same | 148,032 | 294,912 |
| shared_permuted | 148,032 | 294,912 |
| untied_permuted | 296,064 | 294,912 |

The candidate uses about half the wide controls' parameters at equal matrix MACs.
Both fixed permutation buffers are included in resource accounting (6,144 bytes
per model, including controls). Extra operations, memory traffic and kernel
launches are not represented by matrix MACs.

## Every task and seed

Ratios compare candidate to the named control; lower is better. Memory and time
ratios use wide GELU. Every candidate-to-SwiGLU comparison and both sharing
controls are included in the machine-readable gates.

| Task | Seed | Candidate reporting MSE | / Wide GELU | / Narrow GELU | GPU peak ratio | Update-time ratio | Failed gates |
|---|---:|---:|---:|---:|---:|---:|---|
| linear | 17 | 0.475775 | 1.755 | 0.773 | 0.940 | 1.714 | positive_control, wide_quality, memory, runtime |
| linear | 29 | 0.475702 | 1.746 | 0.773 | 0.940 | 1.658 | positive_control, wide_quality, memory, runtime |
| linear | 43 | 0.475338 | 1.751 | 0.775 | 0.940 | 1.615 | positive_control, wide_quality, memory, runtime |
| gelu_teacher | 17 | 0.610218 | 1.397 | 1.087 | 0.940 | 1.623 | positive_control, wide_quality, narrow_quality, sharing_ablation, memory, runtime |
| gelu_teacher | 29 | 0.612140 | 1.399 | 1.092 | 0.940 | 1.609 | positive_control, wide_quality, narrow_quality, sharing_ablation, memory, runtime |
| gelu_teacher | 43 | 0.608451 | 1.396 | 1.087 | 0.940 | 1.571 | positive_control, wide_quality, narrow_quality, sharing_ablation, memory, runtime |
| cubic | 17 | 1.063768 | 0.959 | 0.996 | 0.940 | 1.641 | positive_control, memory, runtime |
| cubic | 29 | 1.067931 | 0.956 | 0.996 | 0.940 | 1.673 | positive_control, memory, runtime |
| cubic | 43 | 1.078157 | 0.958 | 0.996 | 0.940 | 1.623 | positive_control, memory, runtime |

The positive-control gate requires both wide recipes to improve over the zero
predictor by at least 20%. A task failing this condition is an inconclusive
learning probe, never a candidate success. Gates additionally require at least
40% parameter savings, error within 5% of both wide controls, strict gains over
narrow GELU and same-basis sharing, GPU peak at most 90% and update time at most
115% of wide GELU. No averages rescue failed seeds.

## All recipes, three independent seeds

| Task | Recipe | Mean reporting MSE | Median | Sample variance | Mean CUDA peak MiB | Mean update ms | Numerical output ranks |
|---|---|---:|---:|---:|---:|---:|---|
| linear | full_gelu | 0.271667 | 0.271445 | 4.020e-07 | 25.082 | 2.308 | [384, 384, 384] |
| linear | full_swiglu | 0.839575 | 0.839229 | 1.823e-05 | 25.083 | 2.697 | [256, 256, 256] |
| linear | narrow_gelu | 0.614681 | 0.615236 | 1.319e-06 | 22.830 | 2.307 | [192, 192, 192] |
| linear | shared_same | 0.581250 | 0.580501 | 1.202e-05 | 23.579 | 3.802 | [192, 192, 192] |
| linear | shared_permuted | 0.475605 | 0.475702 | 5.490e-08 | 23.579 | 3.835 | [384, 384, 384] |
| linear | untied_permuted | 0.213233 | 0.212748 | 1.031e-06 | 25.838 | 3.841 | [384, 384, 384] |
| gelu_teacher | full_gelu | 0.436684 | 0.436668 | 7.423e-07 | 25.082 | 2.388 | [384, 384, 384] |
| gelu_teacher | full_swiglu | 0.851358 | 0.850790 | 1.767e-06 | 25.083 | 2.733 | [256, 256, 256] |
| gelu_teacher | narrow_gelu | 0.560699 | 0.560805 | 8.867e-07 | 22.830 | 2.397 | [192, 192, 192] |
| gelu_teacher | shared_same | 0.539254 | 0.538663 | 1.116e-06 | 23.579 | 3.828 | [192, 192, 192] |
| gelu_teacher | shared_permuted | 0.610270 | 0.610218 | 3.406e-06 | 23.579 | 3.822 | [384, 384, 384] |
| gelu_teacher | untied_permuted | 0.405014 | 0.404778 | 8.799e-07 | 25.838 | 3.848 | [384, 384, 384] |
| cubic | full_gelu | 1.117165 | 1.116671 | 6.528e-05 | 25.082 | 2.322 | [384, 383, 384] |
| cubic | full_swiglu | 1.145557 | 1.143623 | 6.883e-05 | 25.083 | 2.757 | [256, 256, 256] |
| cubic | narrow_gelu | 1.074146 | 1.071923 | 6.091e-05 | 22.830 | 2.394 | [192, 192, 192] |
| cubic | shared_same | 1.083100 | 1.080427 | 6.101e-05 | 23.579 | 3.838 | [192, 192, 192] |
| cubic | shared_permuted | 1.069952 | 1.067931 | 5.483e-05 | 23.579 | 3.821 | [384, 384, 384] |
| cubic | untied_permuted | 1.102920 | 1.100205 | 6.513e-05 | 25.838 | 3.885 | [384, 384, 384] |

Each of nine independent datasets has 4,096 training, 2,048 validation and 2,048
reporting input vectors of dimension 384. Targets are an independent orthogonal
linear map, a random wide GELU teacher, or mixed cyclic cubic products. Input
means/SDs and output centering/global scale use training rows only. No teacher
weights or held-out targets initialize students. Every arm uses identical
minibatch indices, FP32/TF32 off, AdamW at 0.001, zero decay and clipping at 1.
All 300 updates count; initial validation and final validation/reporting scores
are recorded. The final model is used, with no rate search or checkpoint selection.
This single-rate experiment closes or advances a recipe, not an entire function
family. Synthetic generalization is not language-model quality or convergence.

Update timing includes CPU gathering, transfer, gradient clearing and complete
synchronized optimization; the first 20 timings are excluded. Whole-job CUDA
peaks include scoring and diagnostics but exclude driver/context/other processes.
Datasets and their normalization remain CPU-resident. This is local FFN resource
accounting, not a full model's VRAM measurement or a cross-machine latency claim.

## Reproducibility and prior art

The independent audit regenerates all nine datasets bitwise and replays 108
validation/reporting scores using explicit linear/nonlinear operations without
Model.forward or the training scorer, a different batch size (257), and FP64
error accumulation. Every relative score discrepancy is at most 1e-5. All 54
exported states, 300-step histories, batch-stream hashes, parameter counts,
finite flags and 110 zero allocator boundaries verify. There are 162 study scores
and 108 audit scores, 2,073,600 sampled training vectors and no audit backwards.
Preflight finite-difference evaluations are separate CPU correctness work.

Weight sharing and shuffling are established: [Universal Transformers](https://arxiv.org/abs/1807.03819),
[ShaResNet](https://arxiv.org/abs/1702.08782),
[Shuffling RNNs](https://arxiv.org/abs/2007.07324) and
[permuted-feature compression](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cdt2.12060).
The precise combination's novelty is unresolved. Nothing here is a novelty,
SOTA, general approximation or broad-goal completion claim. H006, H100/H104 and
H106/H107 remain recorded; this result does not reopen their rejected recipes.
The remaining activation baselines are required before any broader claim.

## Next decision

Do not insert this recipe into language models, tune kernels or claim a better FFN. The rank witness survived; the fixed learning/resource recipe did not. Use the linear and teacher-task controls to distinguish optimization or tying restrictions from the rank obstruction. Cubic nonlearning is inconclusive when wide controls also fail. A future change must identify a distinct mechanism or a specific failure diagnosis, not silently extend this failed budget.

[Prospective plan and equations](conjugated_ffn_plan.md),
[summary](../results/conjugated_ffn_v1/summary.json),
[evidence receipt](../results/conjugated_ffn_v1/receipt.json).
