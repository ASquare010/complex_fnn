# Expanded basis with structured readout: retired after language screen

[Registered hypothesis](basis_readout_hypothesis.md). CPU and CUDA numerical
checks passed. The 42-profile resource screen completed: fixed templates pass,
learned templates fail the slowdown limit. Retire the learned-template execution
before language; no language accuracy claim for it. The fixed candidate's registered
18-run language screen and removals completed. Retire this recipe: it misses
both dense quality margins on both corpora. No tuning/confirmation will launch.
Sustained resource benefit and originality remain unproven.

| Corpus | Fixed basis NLL | Strongest full | Strongest compact | Full loss cost | Compact gain | Tied control NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyStories | 2.604161 | 2.538486 | 2.606365 | 2.59% | 0.08% | 2.596579 |
| WikiText | 4.283474 | 4.237625 | 4.288883 | 1.08% | 0.13% | 4.298645 |

The tied control beats the larger grouped candidate on TinyStories; their ordering
reverses on WikiText. Thus independent grouped readout directions are not shown
to improve both corpora under this recipe. Cached scores 2.610949/4.276874 differ
slightly from fused recomputation at the same mathematical function, as their
floating-point execution differs. Single-seed differences are not a reliable
quality advantage. Cached execution also failed resources at 607.5 MiB.

Frozen-weight readout restoration worsens TinyStories/WikiText to 2.746682/4.656733.
Keeping only product responses gives 2.642003/4.365289, and keeping only unmultiplied
SiLU responses gives 6.874086/9.067687. The latter are called "linear" in artifact
keys but are nonlinear activations, not an affine path. Removing all six responses
of even input coordinates gives 3.504561/5.100576. Training-mean FFN replacement
gives 8.283653/9.866708. These changes establish frozen inference dependence, not
an advantage over dense models or semantic roles. Template restoration is unchanged
for fixed templates. All original calibration is retained.

| Corpus | Candidate | Full reference | Allocated MiB | Memory reduction | Throughput ratio | Pass |
| --- | --- | --- | ---: | ---: | ---: | --- |
| TinyStories | Fixed | GELU | 405.5 | 24.83% | 0.954x | True |
| TinyStories | Fixed | Native SwiGLU | 405.5 | 27.69% | 0.962x | True |
| TinyStories | Fixed | Kernel SwiGLU | 405.5 | 24.98% | 0.974x | True |
| TinyStories | Learned | GELU | 405.5 | 24.81% | 0.922x | False |
| TinyStories | Learned | Native SwiGLU | 405.5 | 27.67% | 0.930x | False |
| TinyStories | Learned | Kernel SwiGLU | 405.5 | 24.96% | 0.941x | False |
| WikiText | Fixed | GELU | 405.5 | 24.83% | 0.970x | True |
| WikiText | Fixed | Native SwiGLU | 405.5 | 27.69% | 0.954x | True |
| WikiText | Fixed | Kernel SwiGLU | 405.5 | 24.98% | 0.954x | True |
| WikiText | Learned | GELU | 405.5 | 24.81% | 0.947x | False |
| WikiText | Learned | Native SwiGLU | 405.5 | 27.67% | 0.932x | False |
| WikiText | Learned | Kernel SwiGLU | 405.5 | 24.96% | 0.932x | False |

Seven variants, three alternating-order rounds per corpus, 20 warmup/100 timed
updates; maximum allocated peak and median throughput. Same common backbone,
data, batch, precision and optimizer as registration. All updates finite and
reserved memory below 6 GiB. Fixed candidate's margin is narrow; sustained
confirmation is required. Cached fixed execution peaks at about 607.5 MiB.

Six responses per coordinate (three shifted SiLU functions and their products
with that coordinate) enter a learned block readout. Two full-width input/output
matrices retain global mixing. This tests separate readout directions for basis
responses; it does not create input information or guarantee its retention.

Four-layer FFN weights: fixed bases 2,490,368; learned bases 2,502,656; cached fixed
execution 2,490,368; tied per-coordinate readout 2,109,440. The candidates cut
full SwiGLU weights by 70.54%/70.40%. Projection work 4,980,736 forward FLOPs/token
and 14,942,208 training FLOPs/token, plus basis/reconstruction/reduction work.
The smaller tied control intentionally uses the same physical grouped operation
with a differentiable sparse matrix assembled from actual local coefficients.

CPU FP64 independent scalar equations, all input/template/matrix gradients,
finite differences and basis removals passed. All four complete models passed
causality, token locality, save/load and exact AdamW checkpoint recovery. Their
initialized functions match exactly; Gaussian calibration agrees at 128/256 nodes.
CUDA FP32/BF16 independent ordinary-autograd forward/all adjoints passed, including
widths 512, 37 and 1 and incomplete 32-token tiles. Complete models and interventions
agree with cached references at recorded tolerances; tested BF16 rounding bits
match Torch casts. FP32 grouped GEMMs have TF32 disabled in this runtime.

The fused bank and its exact adjoint use the existing Torch context/stream,
Torch-owned memory and bundled NVRTC. Runtime preparation reuses a staged copy
of the previously checked channel-curve loader; its source is fingerprinted.
Forward materializes one FP32 bank. Backward reconstructs it for matrix gradients,
frees it before allocating basis derivatives, then reconstructs scalar responses.
No higher-order-gradient or bitwise equality to ordinary FP32 arithmetic claim.

Production CPU/CUDA weights/counts/outputs/gradients match the prototypes exactly
in checks. Shared-Trainer CPU recovery is exact in all four modes. An initial
verification process exited without diagnostic output; the standalone rerun
passed, and the unknown cause is preserved in preparation-error records.
All 36 core Python files passed formatting/lint before language source freeze.

Evidence: `records/basis-readout-v1-checks.json`, `records/basis-readout-v1-profiles.json`,
`records/basis-readout-v1-integration-checks.json`, `records/basis-readout-v1-development-errors.json`;
frozen resource-screen source and protocol under `dump/basis-readout-v1/`.
The [model result](../ffn_experiments/basis_readout_transformer/result.md) records every
language/control/removal result. [Conditional confirmation](basis_readout_confirmation.md)
was registered before language training and will not launch for this failed recipe.
The separately registered [activation-map diagnostic](ffn_map_diagnostic.md) measures
what the trained full FFN does before the next architecture is chosen.
