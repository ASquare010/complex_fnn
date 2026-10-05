# Compact refinement: paired-seed confirmation

Registered after the complete 12-run development audit, before confirmation training.
The original refinement protocol supplies the selection and acceptance rules.

Finalists: curve-wide, signed-fine. Parents remain eligible to win.
10 recipes × 3 seeds × 2 corpora = 60 runs.
Seeds 211/307/401, 2000 updates and 4,096,000 targets each; unchanged backbone,
data, lr.0006/warmup200/wd.1, batch8/BF16 and full last-checkpoint validation.
No kernel changes, additional hyperparameter tuning or test-score selection.

| Recipe | Model | FFN | Hidden |
| --- | --- | --- | ---: |
| curve-wide | channel_curve_transformer | self_curve_wide | 512 |
| signed-fine | signed_context_transformer | signed_linear | 512 |
| curve-parent | channel_curve_transformer | self_curve | 512 |
| compact-swiglu-kernel-h346 | base_transformer | swiglu_kernel | 346 |
| compact-gelu-h518 | base_transformer | gelu | 518 |
| signed-parent | signed_context_transformer | signed_linear | 576 |
| compact-swiglu-kernel-h384 | base_transformer | swiglu_kernel | 384 |
| compact-gelu-h576 | base_transformer | gelu | 576 |
| full-swiglu | base_transformer | swiglu_fused | 1376 |
| full-gelu | base_transformer | gelu | 2064 |

Dense hidden sizes are rounded up to each finalist's FFN weight budget.
Report actual counts/mismatch, all paired losses and per-corpus mean changes.
A new variant must improve its parent on both corpus means, with no >5%
VRAM/time regression in isolated resource measurements; otherwise retain the
parent as an eligible choice and describe the tradeoff. No significance claim
from three seeds alone. All recipes keep their recorded failures and measurements.

After language comparisons, profile all confirmation recipes on both corpora:
three alternating rounds, 20 warmup/100 measured updates, original optimizer and
batch/backbone. Freeze the selected compact low-VRAM recipe before independent
seed509/8000-update comparison and one-time test scoring against its parent and
strongest compact/full controls. Final selection is not complete yet.
