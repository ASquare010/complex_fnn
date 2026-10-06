# Coupled channel curves: results

## Current execution: plain PyTorch v1

The active `transformer.py` contains the FFN, block and Transformer model.
CPU and GPU both use normal PyTorch autograd. Custom CUDA compilation and
handwritten backward functions have been removed from active source; the
optimized implementation remains in `ffn_experiments/` for reference.

Initialization, parameter names, 2,121,728 FFN weights and 8,417,792 total weights
are unchanged. Output and input/parameter-gradient comparisons passed in CPU
FP64, CUDA FP32 and CUDA BF16 autocast. Loading the selected TinyStories
checkpoint gives identical CPU logits on the checked input. These checks verify
compatibility, not identical future training trajectories or a fresh language score.

Short resource check on RTX 4070 Laptop: batch 8, length 256, width 512,
four layers, vocabulary 4096, FP32 parameters with BF16 autocast, AdamW and
gradient clipping. Each of three rotating-order rounds used five warmup and
ten timed updates on synthetic token IDs. Times are medians of the three
ten-update measurements; memory is maximum peak allocated memory.

| Execution | Peak allocated MiB | Seconds / 10 updates |
| --- | ---: | ---: |
| Archived optimized Curve-Wide | 401.19 | 0.321 |
| Active plain PyTorch Curve-Wide | 553.19 | 0.419 |
| Dense full SwiGLU | 561.28 | 0.282 |

The readable implementation keeps the parameter saving, but its measured
training memory is only about 1.4% below dense in this check. It uses about 38%
more memory than optimized Curve-Wide and takes about 31% longer here. These
short timings are observations, not a robust throughput study. No new language
experiment was run, so this execution revision is unranked on the leaderboard.
[Local evidence](../../../records/plain-pytorch-v1-checks.json).

## Historical bounded refinement (optimized execution)

Self-Curve's wider-initialization variation (`curve-wide`) advanced to paired-seed
confirmation and is now the frozen overall research choice. Seed101 full validation improves from 2.598287 to 2.591014 on
TinyStories and from 4.288112 to 4.272187 on WikiText, with unchanged 2,121,728 FFN
weights. It ranks first in the six-recipe development comparison. Restoring its
own initial curves gives WikiText 4.272974; this intervention alone does not
identify the cause of the improvement. All removal evidence is retained in
`dump/compact-refinement-v1/`. [Round results](../../../docs/compact_refinement_result.md)
and [confirmation protocol](../../../docs/compact_refinement_confirmation.md).

Confirmation seeds211/307/401 improve over the parent in all six paired runs.
Mean TinyStories NLL is 2.592300 versus 2.605145; WikiText is 4.272862 versus
4.282949. The observed resource medians meet the parent-relative limits, but
variable full-baseline timings do not establish a speedup. The independent
seed509/8000-update comparison and held-out tests are complete and audited.
Curve-Wide beats its parent and compact SwiGLU h384 on validation and test in both
corpora. Held-out NLL is 2.020291 on TinyStories and 3.966762 on WikiText, versus
full SwiGLU 1.974934 and 4.022206. It is the final selected compact recipe; the
full model retains the TinyStories quality advantage. See the
[final summary](../../../docs/compact_refinement_summary.md).
[Frozen choice and limitations](../../../docs/compact_refinement_final_selection.md).
