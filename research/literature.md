# Prior-work audit (begun 2026-09-06)

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
