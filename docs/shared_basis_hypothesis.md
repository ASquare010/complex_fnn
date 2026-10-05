# Next hypothesis: a wider shared FFN input basis

Registered on 2026-10-03 before implementation or training. The current language
screen is still finishing WikiText; these decisions use completed TinyStories
and resource evidence only. Do not change source during that running campaign.

## Evidence motivating the change

In `language-v1`, full SwiGLU reaches TinyStories NLL 2.883340. The strongest
compact dense control reaches 2.979796; BlockShuffle reaches 2.956800; Maxout
2.996539; curve-only 3.063614. None passes the active language margins on this
required corpus. The curve-only inference removal changes NLL by about 0.000011,
so it does not establish richer inference computation. Maxout also misses the
resource thresholds in the short width-192 profiles.

The next hypothesis changes how the limited parameter budget is allocated.
Independent narrow FFNs have few hidden features per layer. Sharing feature
detectors across depth could afford a wider feature bank in every layer, while
each layer retains its own output projection and its existing normalization.

## Equation and cost

For each layer l, using the existing FFN-normalized input x_l:

```text
u_l, g_l = split(W_shared x_l)
FFN_l(x_l) = D_l (SiLU(u_l) * g_l)
```

`W_shared` contains one up projection and one gate projection, shared across all
four FFNs. Each `D_l` is independent. Attention, embeddings, normalization,
residual paths and number of passes remain unchanged. There is no extra loop.

For L layers, width d and hidden width h, unique FFN weights are `(L+2) d h`.
For L=4 this is `6dh`, compared with `12dh` for four independent FFNs of the
same hidden width. Use a shared hidden width twice the compact dense control's
width: both have the same parameter count, but the shared candidate exposes
twice as many hidden features in each layer. Relative to full SwiGLU, the shared
hidden width is about 59% as large, saving about 70% of FFN weights and 41% of
projection arithmetic. Sharing itself does not make repeated projection work free.

Count unique tensors for storage and every executed projection for FLOPs.
The current per-block parameter summation must be updated for shared parameters;
otherwise it would double-count storage and misstate the experiment. Check alias
identity, summed gradients, optimizer uniqueness, save/load and exact resume.

## Prior work and interpretation

[ALBERT](https://arxiv.org/abs/1909.11942) and
[Lessons on Parameter Sharing](https://arxiv.org/abs/2104.06022) establish sharing
across Transformer layers. [One Wide Feedforward Is All You Need](https://arxiv.org/abs/2309.01826)
directly studies FFN sharing and widening. These are substantial prior-art overlaps.
This restricted sharing of the input/gate bank with independent output maps is
an experimental design choice, not an established original contribution.

Prediction: reusable detectors plus layer-specific readouts preserve more useful
features than four separately narrow banks at the same weight budget. Risks:
layers need incompatible detectors; joint gradients interfere; attention must
spend capacity aligning representations; the wider bank costs more activation
memory than equally small dense controls; saved weights fail to produce a real
speed or memory gain. A gain could arise from training regularization rather
than wider inference computation. Do not assume the explanation in advance.

## Predeclared hardware selection

Before viewing quality for this mechanism, profile the following fixed options.
Select the smallest width whose short profile meets the resource screen against
both full fused SwiGLU and full GELU. The final resource claim still requires
longer repeated measurements. If none passes, revise before language training.

| Width / heads | Full SwiGLU h | Shared-basis h | Compact SwiGLU h | Compact GELU h |
| --- | ---: | ---: | ---: | ---: |
| 192 / 6 | 512 | 304 | 152 | 228 |
| 384 / 12 | 1024 | 608 | 304 | 456 |
| 512 / 16 | 1376 | 816 | 408 | 612 |

All have four layers, head dimension 32, context 256, batch eight, vocabulary
4096 and BF16 CUDA. Every compared model uses the same chosen backbone. A larger
comparison is a separately registered experiment, not a reinterpretation of the
failed width-192 results. The scale selection is based on hardware viability,
not whichever scale happens to produce a favorable language loss.

Profile actual TinyStories and WikiText batches, 20 warmup plus 100 measured
updates, three alternating-order rounds, with fresh model/optimizer state each
time. Reject above 6 GiB reserved VRAM. Screen for >=1.2x throughput OR >=20%
lower allocated VRAM, with <=5% deterioration in the other resource. Also record
an attention-only execution reference to estimate the memory cost outside FFNs;
it is a diagnostic lower bound, not a proposed qualified replacement.

## Language comparison and removal experiment

At the selected width, train shared-basis, full fused SwiGLU, full GELU, compact
fused SwiGLU and compact GELU with identical data windows and 2,000-update budgets,
seed 101, rate 0.0006, warmup 200, AdamW decay 0.1. Compare both language corpora.
Add an untied model of the same hidden width initialized with copies of the shared
input/gate weights. It has twice the FFN weights and is not a size-matched control;
it isolates the cost of keeping detectors tied during learning. Report that cost.

Copying shared weights into separate tensors after training leaves the function
unchanged and is not a valid mechanism-removal experiment. To investigate whether
the wider bank is used at inference, remove a preregistered half of its hidden
features (even indices), without fitting on validation. Apply the same proportional
removal to controls, and compare with a static mean-output intervention estimated
from training batches only. These interventions can establish dependence on
feature computation, but not semantic explanations by themselves.

The active goal's thresholds remain unchanged: within 1% of the strongest full
dense loss and >=1% better than the strongest compact dense loss on both corpora,
plus the resource criterion. One seed is only a screen. Survivors require equal
rate/seed tuning and frozen independent confirmation; unsuccessful recipes are
retired with their measurements in `result.md`.
