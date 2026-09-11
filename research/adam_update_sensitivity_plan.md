# H119 — first Adam update sensitivity (prospective)

## Question and scope

H118 found ordered but inconclusive repeat variation on the selected failed
WikiText seed101 fixture. Test whether Adam's first coordinate normalization
amplifies its tiny saved initial-gradient differences. This is a conditional
mechanism diagnostic, not another seed, a language-model quality experiment or
a novelty claim. The old quality failure stays failed.

The six H117/H118 initial **probe** gradient sets and one common step-zero state
are read unchanged. Probe gradients need not be bitwise identical to actual
training-update-one gradients. Disposable updates below are counterfactuals;
they do not reconstruct the later training trajectories exactly.

## Fixed budget and implementation

- Six gradient sets, each replayed once on CPU and once on CUDA in FP32:
  **12 disposable native AdamW steps**, zero model forwards/backwards, zero
  language-training updates, zero training tokens and zero validation scores.
- One GPU process; CPU cases first, then CUDA cases. No timing repetitions.
- Same maintained Transformer and `parameter_groups`, initial weights, empty
  Adam state, LR 0.0006, betas (0.9, 0.95), epsilon 1e-8, matrix decay 0.1,
  no norm-weight decay, clip norm 1 (PyTorch's 1e-6 clipping guard), four CPU
  threads, TF32 off, existing UV-managed Python/PyTorch environment.
- Keep saved optimizer options, including automatic foreach/fused selection.
  CPU/CUDA may select different kernels; neither is forced to mimic the other.
- Save all 12 clipped gradient maps and 12 updated parameter/optimizer states. Record
  initial/gradient/result hashes, parameter-group names/options, finite checks,
  state-step checks, single-step wall time and CUDA allocator intervals.
  CPU memory is unmeasured/null. No training data or attention is involved:
  these memory/timing measurements cannot qualify model efficiency.
- Hash previous receipt/evidence, maintained files, installed Adam/AdamW and
  clipping source, this plan and scientific scripts before any experiment.
  Snapshot navigation before later publication. Preserve failures; no silent
  retry or adaptive extra allocation.

## Exact-arithmetic calculation

For clipped scalar gradient g, empty moments and epsilon e > 0, bias correction
gives m-hat = g and v-hat = g² at step one. Therefore

    h_e(g) = g / (|g| + e)
    theta_1 = (1 - eta * lambda) theta_0 - eta h_e(g).

The derivative on both sides, including the limit at zero, is

    h'_e(g) = e / (|g| + e)² <= 1/e.

For same-sign a,b the absolute difference is
`e |a-b| / ((|a|+e)(|b|+e))`. The global 1/e Lipschitz bound also follows by
the mean value theorem. These are direct consequences of the existing Adam
equations, not new theorems. A large worst-case bound does not prove that an
observed gradient pair is amplified. Native FP32 moment arithmetic, decay and
parameter rounding also differ from the ideal formula and must be measured.

Sources: [Adam](https://arxiv.org/abs/1412.6980),
[PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html),
[numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
The installed source is separately pinned in the protocol.

## Predeclared analysis and decisions

Distance means symmetric global relative L2:
`||a-b|| / max((||a||+||b||)/2, 1e-12)`, evaluated on CPU in FP64.
Compute all 15 unordered pairs (nine cross-policy, six within-policy) for raw
gradients, CUDA-clipped gradients, ideal adaptive directions, realized native
updates (`theta_0-theta_1`, subtracted in FP64), and final parameters.
Also report CPU-versus-CUDA clipped-gradient/update/parameter distance for all
six inputs.

At actual epsilon 1e-8, define amplification as ideal-direction distance divided
by clipped-gradient distance. Explicitly report null if that denominator is
zero. Attribute squared direction difference to bins of
`max(|g_a|, |g_b|)/e`: [0,1], (1,10], (10,100], (100,infinity).
Report coordinate counts, squared-error shares and opposite-nonzero-sign count.

The fixed **strong near-zero amplification** prediction passes only if ALL
nine cross-policy pairs have amplification >= 10 AND at least 50% of squared
direction difference comes from the first two bins (<=10e). Otherwise reject
that prediction on this fixture; do not replace it with a favorable average.

Pure closed-form ablations use e in {1e-8, 1e-7, 1e-6}, for six inputs each
(18 direction maps and 45 pairs). No native steps at alternative epsilon.
For each larger epsilon, a **small-distortion conditioning** criterion requires
ALL nine cross-policy direction distances <= half their baseline values AND
ALL six directions change by <=1% relative L2 from their baseline direction.
Report failure as well as success; this is not a quality/promotion gate.

## Independent audit and accounting

A separate NumPy FP64 implementation checks all 12 actual steps against the
first-step formula and checks clipping against the raw saved gradients. Fixed
native-parameter tolerances: global relative L2 <=1e-6, maximum absolute error
<=2e-7. Clipped-gradient relative error <=1e-6 against FP64 norm clipping;
each saved first-moment and second-moment map has relative error <=1e-6 against
its exact-arithmetic first-step formula, and every saved step counter is one.
Report realized-update error separately without a relative threshold, since
subtraction from stored FP32 weights exposes rounding of small updates.

Independently recompute all 45 direction-pair metrics and all 12 distortions
(six per larger epsilon), bins and decisions. Floating metrics must agree
within relative 1e-8 / absolute 1e-12; integer counts agree exactly. Verify
finite tensors, names/shapes, 12 one-step states, 24 artifact hashes and seven
zero CUDA allocator boundaries. Audit uses zero optimizer steps and no
forwards/backwards. Formula witnesses include h(0)=0, h(e)=1/2 and the
same-sign difference identity; they are not extra GPU work.

The result must distinguish mechanism support, rejected thresholds, actual
parameter rounding and unresolved long-horizon causality. No larger epsilon
is installed in maintained training from this diagnostic alone. Broader
architectural and lower-VRAM research goals remain unachieved.
