# Prior-work audit (begun 2026-09-06)

## H114 numerical accuracy and CUDA timing

[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
warns that equivalent floating-point computations need not agree bitwise.
[CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) and the
[event API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html)
support separate synchronized wall/event timing and allocator accounting.
[H114](fp32_classifier_profile_results.md) measures stable warmed full-model timing
but fails its full-gradient gate and two replay checks. The documentation does
not identify the cause of these particular discrepancies. All original gates
and failures remain explicit; no novelty or deterministic-backward claim follows.

## September 10: shared BF16 casts and classifier gradients (H113)

[PyTorch's AMP API](https://docs.pytorch.org/docs/2.14/amp.html) exposes the weight
cache setting. Its [implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/autocast_mode.cpp)
caches eligible leaf-weight conversions, and its
[accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
warn about operation ordering. H113 tests the actual resulting graph and
gradient dtype; it does not claim caching or mixed-precision analysis as new.

[Higham, The Accuracy of Floating Point Summation (1993)](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf),
section 2, supplies the standard summation error bound and conditioning caveat.
The local 256-versus-260 counterexample and derivative partition are elementary
applications. They explain a measured local gradient difference, without proving
that it causes H112's final loss differences or a general training pathology.

[H113](classifier_precision_results.md) verifies the mechanism on 24 saved states,
rejects uncaching alone and qualifies FP32 chunks for further resource screening.
It retains every timing outlier and explicitly limits its memory accounting.
FP32 computation and loss partitioning remain established techniques; no new
activation, kernel, architecture, SOTA or numerical-analysis theorem is claimed.

## September 10: evaluation and whole-job memory (H111/H112)

[Cut Your Losses](https://arxiv.org/abs/2411.09009) identifies the large classifier
logit allocation and uses specialized cross-entropy kernels. H111/H112 measure
ordinary PyTorch token chunks on this project's decoder; they do not reproduce
CCE kernels or establish an improvement over that system. Valid-target-weighted
loss partitioning is an elementary identity, without a novelty claim. Preserving
an objective in real arithmetic does not ensure an identical BF16 trajectory.

[PyTorch 2.14 CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cublas-workspaces)
documents persistent cuBLAS workspaces and the explicit clearing operation.
H112's two-repetition probe attributes its post-qualification residual allocation
to those workspaces. Clearing them only between trials restores strict empty
tensor-storage boundaries while charging their normal cost during measurement.
This does not diagnose the separate intermittent native Python import failures.

The same documentation distinguishes tensor allocation from allocator reservation;
both differ from whole-process VRAM. H111's fixtures and H112's actual trial
peaks are reported separately. The [H112 result](whole_job_memory_results.md)
qualifies a scoped TinyStories memory component and rejects the fixed two-corpus
claim after WikiText quality failures. It supplies no new activation, SOTA
comparison, scale result or general gradient-stability proof. Primary sources
were checked during this continuation.

## September 10: learning from derivative-aware initialization (H109)

H109 follows the H108 precedents below, particularly
[Sobolev Training](https://arxiv.org/abs/1706.04859) and
[GRAIL](https://proceedings.mlr.press/v328/tang26a.html). It tests whether derivative
information supplied only through initialization survives subsequent output-only
optimization. It does not reproduce continual Sobolev training or claim it fails.

The local result is negative: most of the initializer's derivative advantage is
lost after value-only learning, even as output error improves. An elementary
small-amplitude/high-frequency sine counterexample explains why small value error
alone cannot guarantee small derivative error. This is a general non-implication,
not a new theorem, proof of the measured mechanism or deep-network stability
guarantee. Ambient directional probes differ from upstream decoder perturbations.
[H109 evidence](sobolev_learning_results.md).

## September 10: derivative-aware neuron selection and readout (H108)

[Czarnecki et al., Sobolev Training (2017)](https://arxiv.org/abs/1706.04859)
establishes learning from target values and derivatives. H108 applies that idea
to a ridge readout of selected trained FFN features, with sampled directional
derivatives. It does not claim derivative distillation as new.

[Chen, Cowan and Grant, Orthogonal Least Squares Learning Algorithm for Radial
Basis Function Networks (1991)](https://doi.org/10.1109/72.80341) supplies the
greedy feature-selection precedent. The regularized multi-output marginal gain
used here follows directly from a Schur complement; it is not a new theorem.
[GRAIL (2026)](https://proceedings.mlr.press/v328/tang26a.html) is a recent precedent
for Gram/ridge reconstruction after model compression. These precedents prevent
claiming novelty for selecting neurons and refitting an output projection.

[FLAP (2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28960) and its
[official implementation](https://github.com/CASIA-LMC-Lab/FLAP) motivate the
fluctuation-importance and bias-compensation control. The local H108 control is
not a reproduction of FLAP's complete language-model compression pipeline.

The measured contribution is a controlled eight-method ablation on trained
GELU/SwiGLU FFNs: derivative-aware calibration improves both local errors, with
most derivative gain coming from the readout objective. Complete quality/VRAM
gates still fail. This is a promising component result, not a new activation,
SOTA comparison, priority claim or proof of deep-network gradient stability.
[H108 report](sobolev_selection_results.md).

## September 10: explicit affine residuals on trained FFNs (H106/H107)

[Whipp, How Linear Is a Transformer Feed-Forward Block?](https://arxiv.org/abs/2606.19379)
(June 2026) is directly relevant: closed-form affine recovery of trained FFNs,
heterogeneity across blocks/models, bilinear residual probes and downstream
insertion controls. The paper distinguishes local reconstruction from language
perplexity and motivates a healed-original control when comparing compression
followed by training. H106/H107 do not claim novelty for an affine decomposition,
least-squares warm start or residual distillation. Their experiments test a
specific parameter and measured-memory tradeoff in this repository.

[Wide & Deep](https://arxiv.org/abs/1606.07792) is a longstanding precedent for
combining linear and nonlinear predictors, on a different recommendation task.
H106's residual-rank identity follows from orthogonal projection plus
[Eckart-Young](https://doi.org/10.1007/BF02288367). A finite-data capacity bound
allows arbitrary low-rank correction matrices; passing does not certify that
actual neurons learn them. Reporting-data oracle fits stay diagnostic and are
never used to initialize the separately sampled H107 students.

## September 10: real-FFN projections and output-rank limits (H105)

[ASVD](https://arxiv.org/abs/2312.05821) incorporates activation statistics into
post-training low-rank compression. [SVD-LLM](https://arxiv.org/abs/2403.07378)
uses truncation-aware whitening and sequential parameter updates;
[IO-SVD](https://arxiv.org/abs/2605.15626) adds output sensitivity and adaptive
rank allocation. H105's stated linear-response control is not a reproduction
of these complete systems, and no SOTA claim is based on it.

[Eckart and Young, 1936](https://doi.org/10.1007/BF02288367) supplies the matrix
rank-approximation precedent. H105 applies its singular-value-tail result to
centered finite teacher outputs, then uses the reverse triangle inequality to
account for BF16/FP32 target drift. This is a local capacity diagnosis using
established mathematics. It does not transfer to whole-model NLL or prove that
any architecture with fewer parameters must fail. [Measured limits](real_subspace_results.md).

## September 10: supervised feature discovery, H103/H104

- [Second-order Stein multi-index estimation, NeurIPS 2017](https://papers.neurips.cc/paper/7190-estimating-high-dimensional-non-gaussian-multiple-index-models-via-steins-lemma.pdf)
  provides response/score-moment and truncation precedents. A label-energy
  transformation followed by a second moment is not claimed as a new paradigm.
- [Gaussian multi-index gradient flow, AISTATS 2025](https://proceedings.mlr.press/v258/simsek25a.html)
  studies feature-learning difficulty when useful low-degree Hermite components
  are absent. This motivates diagnosis; it does not prove our optimizer converges.
- [The Generative Leap](https://arxiv.org/abs/2506.05500) and
  [Neural Networks Learn Generic Multi-Index Models](https://arxiv.org/abs/2511.15120)
  supply further spectral/layerwise feature-learning precedents with their own
  assumptions. We do not transfer their guarantees to the present estimator.

H103 derives a scoped Gaussian population identity and tests bounded energy
weights. H104 combines that initializer with established matrix factorization
and cubic/GELU features. All fair dense controls receive the same initializer.
Its synthetic gains and failed promotion gates are [reported together](spectral_fitting_results.md).
No new activation, finite-sample theorem, general nonvanishing-gradient guarantee
or language-model improvement is claimed. Constant-norm language labels give no
signal; teacher-output use would require a separate costed experiment.

## September 10: memory-first execution branch

- [Reformer, section 3](https://arxiv.org/abs/2001.04451) explicitly splits FFN
  computation across tokens to reduce intermediate memory. Its reversible
  architecture and attention modifications are separate mechanisms. H101 uses
  ordinary existing attention/residuals and native checkpointing only.
- [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009)
  avoids full classifier-logit materialization with specialized kernels. H101/H102
  instead use straightforward token chunks and PyTorch CE; each chunk still
  materializes C-by-V logits and no gradient filtering is used. It is not a CCE
  implementation or a demonstrated improvement over it.
- [PyTorch 2.14 checkpoint documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html)
  supplies the non-reentrant recomputation mechanism. [The H101/H102 report](token_memory_results.md)
  measures the combined cost and observes that real-arithmetic identities do not
  guarantee BF16 optimizer-trajectory fidelity. This is our scoped experimental
  observation, not a new theorem or an exhaustive novelty finding.

- H110 extends the fixed narrow-GELU loss-chunk recipe from12 to800 updates,
  retaining the same real-arithmetic CE objective and an independent unchunked
  evaluation path. The [duration study](token_memory_duration_results.md) explicitly
  separates training memory from whole-job memory, because validation may become
  the peak after training intermediates are reduced. This is a measurement and
  duration qualification of an established execution method, not a new activation
  or an implementation of CCE. The checkpoint and CCE primary sources above were
  checked again for this continuation.

## Architectural sources

Primary sources checked; this is not exhaustive novelty certification.
Author-reported results are not reproduced here. Follow-up ideas are our inferences.

| Source | Mechanism / evidence | Cost, limitation and consequence |
|---|---|---|
| [GLU variants](https://arxiv.org/abs/2002.05202) | Multiplicative projected features improve Transformer results | Three matrices; use 2/3 width for parameter matching |
| [KAN](https://arxiv.org/abs/2404.19756) | Learned edge splines; small function-fitting/scientific examples | Edge functions cost memory and compute; toy wins are not LM proof |
| [KAT / Group KAN](https://arxiv.org/abs/2409.10594) | Shared rational functions, CUDA evaluation, variance-aware initialization | Group sharing already exists; vision results do not prove our LM target |
| [PadÃ© Activation Units](https://arxiv.org/abs/1907.06732) | Trainable rational activations | Denominator safety/evaluation costs; strong future activation control |
| [Adaptive piecewise-linear activations](https://arxiv.org/abs/1412.6830) | Learned hinge combinations | More knots cost work; learned shape itself is prior art |
| [Learnable polynomial/trigonometric/tropical activations](https://arxiv.org/abs/2502.01247) | Learned bases tested on ImageNet/OpenWebText | Initialization and basis evaluation matter; basis mixtures already studied |
| [Learning polynomial activation functions](https://arxiv.org/abs/2510.03682) | Polynomial optimization treatment | Specialized optimization is not evidence of fast GPU LM training |
| [Low-rank Transformer training](https://arxiv.org/abs/2407.09835) | From-scratch low-rank models up to 1.3B | Rank and throughput tradeoffs; indispensable control for future low-rank claims |
| [Structured feedforward layers](https://openreview.net/pdf?id=WxLVYZbIew) | Low-rank/block-diagonal FFNs from scratch | Some comparisons change attention too; our attention stays fixed |
| [Monarch](https://arxiv.org/abs/2204.00595) | Products of structured block matrices | Restricted structure and kernel costs; alternative to input bottlenecks |
| [ALBERT](https://arxiv.org/abs/1909.11942) | Parameter sharing and embedding factorization | Fewer weights do not imply less compute; time recurrence |
| [BitNet b1.58](https://arxiv.org/abs/2402.17764) | Ternary weights | Storage reduction differs from fewer learned parameters |
| [Sparse MoE](https://arxiv.org/abs/1701.06538) | Conditional specialist execution | Active compute can be low while total parameter count is high |
| [TinyStories](https://arxiv.org/abs/2305.07759) | Small-model language learning | Narrow synthetic distribution; screening only |
| [FineWeb](https://arxiv.org/abs/2406.17557) | Curated general web corpus | Broader, more expensive evaluation for finalists |

No verified source here establishes our exact bounded cubic residual at 75% FFN
reduction as useful. Absence in a search is not proof of novelty. Expand BÃ©zier,
Bernstein-network and coordinate-dependent basis searches before publication,
and reproduce practical rational/structured baselines. Our bank-collapse and
bottleneck proofs are elementary structural observations without priority claims.


## Additional comparisons checked during branching

- [Structured FFNs, Wei et al.](https://arxiv.org/abs/2406.16450) and their
  [official code](https://github.com/CLAIRE-Labo/StructuredFFN): block-structured
  projections, self-guided training and separate decoding optimizations. Our
  simple grouped sandwich is a related control, not their exact reproduction.
- [Flash Multi-Head FFN](https://arxiv.org/abs/2512.06989): a current relevant
  efficient-FFN comparator. Its reported kernel/memory gains must not be
  attributed to our eager grouped implementation; the completed local adaptations are recorded below.
- [KANs for Small Language Models, Alves and Vicente](https://arxiv.org/abs/2607.15525):
  reports inconsistent downstream/scaling benefits of the tested learned-function
  replacements against strong MLP/SwiGLU baselines. This supports caution, not a
  claim that our specific negative result proves all KANs fail.
- [One Wide Feedforward Is All You Need](https://arxiv.org/abs/2309.01826): FFN
  redundancy and sharing are direct prior art. Their encoder/decoder changes
  differ from our fixed decoder-only attention experiment.
- [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672): layer-wise
  LoRA relaxes weight tying. It is a required comparator if we claim a new benefit
  from depth-specific adaptation of shared transformations.
- [FiLM](https://arxiv.org/abs/1709.07871): feature-wise affine conditioning is
  established. Any proposed per-layer scale/shift adaptation must acknowledge it.
- [MAXIM](https://arxiv.org/abs/2201.02973): cross-gating of feature streams exists
  in vision. A cross-group gate is not new merely because it uses that phrase.

Novelty remains unproven. We are establishing serious baselines and identifying
which mechanisms deserve original follow-up, rather than labeling prior art a
breakthrough.


## Follow-up structured-layer implementation audit
[Compute Better Spent](https://arxiv.org/abs/2406.06248) introduces Block Tensor-Train
layers and emphasizes architecture-dependent initialization and learning-rate
scaling. This is an additional required prior-work control for future structured
claims. The [StructuredFFN official code](https://github.com/CLAIRE-Labo/StructuredFFN)
distinguishes BlockShuffle (two block-diagonal factors) from BlockDense and
LowRank. The initial single-factor grouped controls were not reproductions of
those stronger structured families and are now retired. Their self-guided training also changes the
training computation and must be budgeted explicitly if implemented here.


The new `src/blockshuffle_ffn` now implements the two-factor BlockShuffle
control, with semi-orthogonal factor initialization and the declared fan-in
optimizer treatment. It compresses all FFNs in our fixed micro-decoder; it
does not reproduce the papers' scales, data, dense first FFN or optional
self-guided training. Read [the measured three-seed comparison](blockshuffle_results.md).


The [current comparator audit](current_comparator_audit.md) now records an
equation-level reading of FlashMHF and a derived near-matched parameter budget.
The local H046/H047 empirical comparisons are complete and failed their tested
quality gates; those implementations are archived. Published-scale and flash-kernel
reproduction remain distinct outstanding questions.


## Token-conditioned residual qualification

[H067](token_activation_results.md) qualifies a baseline-preserving, one-sided
residual adaptation of [MoA/LA](https://arxiv.org/html/2605.26647v1), alongside its
static control. Its [scoped proof](token_activation_theory.md) supplies a smooth
exact-function witness using the meromorphic viewpoint also found in
[Tran et al.](https://arxiv.org/html/2606.17816v1). These are established mechanism
and proof-method precedents. The unregistered prototype has no learned fitting,
language, memory or runtime result yet; no novelty claim follows.

## Internal factor rescaling and optimizer geometry (H069)

[Path-SGD](https://papers.neurips.cc/paper_files/paper/2015/file/eaa32c96f620053cf442ad32258076b9-Paper.pdf)
addresses optimization under node rescaling. [Du, Hu and Lee](https://arxiv.org/abs/1806.00900)
prove balancing properties for particular homogeneous gradient-flow/GD regimes.
Their assumptions do not establish an AdamW guarantee for this gated model.
[Singh, The Loss Does Not See the Basis, but Adam Does](https://arxiv.org/html/2608.05136v1)
studies orthogonal factor-gauge equivariance and distinguishes it from balancedness
and sufficient conditions for low-rank recovery. Orthogonal rotation invariance
is not diagonal scale invariance; this repository has not reproduced its training
experiments or used the paper's results as evidence of a BlockShuffle repair.

[H069](factor_balance_results.md) applies an elementary internal diagonal gauge
and weighted-energy minimization to existing structured factors. Exact local
function/gradient preservation passes, but measured imbalance misses the frozen
motivation gates. No optimizer is proposed on this diagnosis, and neither the
gauge identity nor power-of-two rounding bound is a novelty-priority claim.

## Minimal internal orthogonal mixing (H070)

[Group and Shuffle, section 3](https://arxiv.org/html/2406.10019v1#S3) characterizes
fixed-permutation products by low-rank canonical blocks and develops more general
structured orthogonal products. [Kaleidoscope](https://arxiv.org/abs/2012.14966)
learns structured linear maps, and [ButterflyQuant](https://arxiv.org/abs/2509.09679)
uses learnable Givens butterfly angles for quantization. These precedents prevent
claiming a new transform family from a Givens stage inside BlockShuffle.

[H070](rotated_shuffle_results.md) distinguishes a same-origin rotation that can
be absorbed into an existing factor from a cross-origin rotation with a rank 7
projection witness. Local correctness is qualified, but fitted quality, resources,
full-FFN separation and novelty remain unestablished. The rank-count audit also
limits interpretation of H068's smaller synthetic setting; its recorded rejection
is preserved rather than converted into full-size evidence.

## Ungated structured control and parity (H072)

The definitions in [GELU](https://arxiv.org/abs/1606.08415) and
[GLU variants](https://arxiv.org/abs/2002.05202) provide the conventional activation
families. [Wei et al.](https://arxiv.org/html/2406.16450v1#S2.SS1) study structured
FFN parameterizations including the two-factor BlockShuffle map. The present
ungated path already existed in this repository. Reallocating its gate budget
to width is a matched conventional control, not a new structured-transform family.

[H072's analysis](ungated_blockshuffle_theory.md) derives the bias-free odd/even
identities and a population obstruction for one polynomial target. Those are
explicit elementary consequences of the definitions, not claims of proof priority,
whole-network separation or improved learning. Local derivative/initialization
checks cannot substitute for controlled quality and resource experiments.


## H115: finite-precision decoder transport and SDPA controls

[PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
documents backend-dependent numerical behavior and FP32 intermediates in math
attention for BF16 inputs. [Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
and [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
describe floating-point and reproducibility limits. These are general facts,
not evidence identifying a particular operator in the current failed replays.

[H115](decoder_gradient_transport_results.md) distinguishes repeat variability
from paired perturbation transport by holding classifier dH fixed across decoder
replays. Default attention exposes EfficientAttentionBackward, not a measured
FlashAttention node. Math BF16 removes observed repeat variability but still
amplifies paired differences; both FP32 modes pass the numerical gates. Forward
values change under precision/backend interventions. The exact 32,768-fold
scalar rounding witness is elementary arithmetic, not a new theorem or a
condition-number measurement. Chunked classifier memory has established prior
work, including [Cut Your Losses](https://arxiv.org/abs/2411.09009). H115 makes
no novelty or training-quality claim and earns only resource screening.


## H116: precision and classifier memory as a measured joint recipe

[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
supports synchronized timing and distinct allocated/reserved accounting.
[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
documents backend choices and numerical differences. [Cut Your Losses](https://arxiv.org/abs/2411.09009)
is established classifier-memory precedent; chunking and FP32 are not novel here.

[H116](fp32_decoder_resource_results.md) measures default FP32 chunks at about
27% lower full-job allocation and comparable BF16-relative update time on narrow
T512 fixtures. Same-precision native controls isolate the chunking cost. Forced
math attention fails the practical memory/time gate, and full T128 controls
fail memory. Twenty-four independent FP32 backward replays pass; this does not
establish global determinism. The short, correlated fixture scope earns broader
replication/duration only, with no language-quality or architectural novelty claim.

## H117: local gradient agreement does not establish training equivalence

[H117](fp32_training_replication_results.md) extends H116 to 18 fresh runs,
three seeds per corpus and 800 updates. It retains memory savings while the
fixed all-seed quality requirement is assessed separately. No new activation,
FFN or classifier-memory algorithm is claimed. Native checkpoint scoring and
initial FP32 gradient replays are verified independently; full training
trajectory reproducibility is not tested. A failed endpoint cannot be uniquely
attributed to chunking without within-policy trajectory controls.

Previously reviewed primary sources remain applicable:
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html),
[reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) and
[Cut Your Losses](https://arxiv.org/abs/2411.09009).
They explain general numerical constraints and established classifier-memory
work; they do not prove the cause of this study's specific NLL differences.
This is an empirical replication with scoped decisions, not a new literature
priority or theoretical convergence claim.

## H118: compare repeat variability before attributing a training gap

[H118](training_variability_results.md) measures four additional executions of
the selected failed WikiText fixture. Initial gradients differ by tiny amounts,
but trained states and final NLLs differ between unchanged-policy runs. All
observed chunked endpoints remain worse; the predeclared1% separation criterion
is not established. This selected-seed result neither proves equivalence nor
isolates a kernel or optimizer cause.

The report gives the elementary update-map perturbation recurrence: controlling
initial gradient error alone cannot bound final-state error without assumptions
about every update's sensitivity. No new theorem is claimed. Primary context:
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
and [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
These sources establish general numerical limitations, not a causal explanation
of our specific result. Adam's response to saved near-zero gradient coordinates
is a separate, unallocated diagnostic hypothesis.

## H119: first-step Adam sensitivity is a known equation, not a new optimizer

[H119](adam_update_sensitivity_results.md) derives g/(|g|+epsilon) at the first
Adam step and its derivative epsilon/(|g|+epsilon)^2 directly from
[Adam](https://arxiv.org/abs/1412.6980) and
[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
Those primary sources were checked again on 2026-09-10; installed Adam/AdamW,
clipping and optimizer sources are pinned separately. The measured 61–64x
amplification and failed epsilon-distortion criteria are local observations,
not priority claims or a proof of long-training causality. CPU clipping fails
the preset FP64 tolerance; the six CUDA checks pass. Existing
[numerical-accuracy guidance](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
does not guarantee cross-device equality. AdamW's documented foreach temporary
storage motivates a separate memory profile, not an assumed saving.

## H120: optimizer temporaries are distinct from complete-job peaks

[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
documents foreach temporary storage and its native per-tensor/fused alternatives.
The documentation was checked on 2026-09-10 and installed optimizer sources
were pinned before profiling. [H120](optimizer_memory_results.md) measures a
35–36 MiB fused optimizer-phase reduction but no complete-job reduction:
backward is larger in every fixture. This is a scoped systems finding, not a
new optimization algorithm. Independent NumPy updates and native scoring pass.
Further memory work must measure backward tensor lifetimes and residency;
a temporary-storage reduction in a smaller phase cannot lower an unchanged
larger phase. Existing long-run quality failures remain failed.

## H121: existing saved-tensor hooks for checkpoint-input offload

[PyTorch save_on_cpu](https://docs.pytorch.org/docs/2.14/autograd.html#torch.autograd.graph.save_on_cpu),
[checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html) and the official
[checkpoint/offload wrapper](https://github.com/pytorch/pytorch/blob/main/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py)
were checked on 2026-09-10. Installed checkpoint, autograd, CUDA-memory and
wrapper source hashes are frozen. This runtime saves checkpoint inputs before
entering its internal hook, so an outer CPU hook captures the eight boundary
inputs. H121 verifies their values and lifetimes experimentally. The technique
is established; the measured saving is about 12% with native classifier loss
and 5% with chunked loss, failing the preset all-four-fixture memory gate.
No novel algorithm is claimed.
