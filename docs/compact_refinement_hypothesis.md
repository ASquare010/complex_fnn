# Compact FFN refinement, round 1

Registered before implementation or measurement, 2026-10-04. User direction:
improve the best measured recipes with a bounded architecture/initialization
study; no new C++/CUDA kernels or execution-optimization campaign. The practical
compression/memory/near-full-quality milestone is achieved in development;
strict superiority and reliability remain unproven. Preserve the main charter.

## Starting points and evidence

| Parent | TinyStories NLL | WikiText NLL | FFN weights | Training allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| Signed-linear context, h576/g64 | 2.603935 | 4.277294 | 2,490,368 | 415.9 |
| Self-gated curves, width512 | 2.598287 | 4.288112 | 2,121,728 | 401.2 |
| Fixed basis readout, 16 groups of 32 | 2.604161 | 4.283474 | 2,490,368 | 405.5 |

Sources: per-model result.md files and compact records linked in the leaderboard.
These are single-seed development results. Signed-linear is balanced; self curves
trade slightly weaker WikiText for lower size and strong TinyStories; fixed basis
offers a different learned readout mechanism. Tied-basis has slightly better
TinyStories than self curves but worse WikiText; it is not silently erased.
Cached basis is excluded from this shortlist because its measured memory is high.

## Three variations, one per parent

1. **Finer signed gates:** signed-linear with hidden512/groups128 instead of
   hidden576/groups64. Four detectors per context rather than nine. Hypothesis:
   more context selectivity offsets the smaller detector bank. 2,359,296 FFN
   weights (72.09% cut). This changes both allocation and group granularity;
   a gain is not evidence that finer grouping alone caused it. Reuse the existing
   signed activation unchanged. Frozen gate-neutralization and even-feature
   removal use the parent study's definitions.
2. **Wider initial self curves:** keep three trainable branches and all existing
   equations, initialize slopes [.5,1,1.5], offsets [-1,0,1], rather than
   [.75,1,1.25] / [-.5,0,.5]. Retain c=1,e=0 and recalculate initial mean/scale
   with the same quadrature/target variance. Same 2,121,728 weights (74.90% cut).
   Hypothesis: a broader initial range improves learning useful responses.
   This is an initialization/learning-dynamics hypothesis, not a larger function
   class. Compare frozen trained shapes with their own initial shapes, retaining
   original calibration. Existing curve kernels are reused unchanged.
3. **Smaller basis groups:** fixed basis with 32 groups of 16 instead of 16 of
   32. Same six scalar responses and full P/Q projections. 2,293,760 weights
   (72.87% cut). Hypothesis: restricting readout freedom improves generalization
   while retaining useful local feature combinations. Reuse existing readout
   operations unchanged. Test restore-initial-readout and product/linear-only
   frozen removals with original calibration.

## First-stage comparison

Exactly six recipes (three parents and three variations), two corpora, seed101:
12 full language runs. Re-run each parent so the comparison is contemporaneous.
Backbone d512/L4/heads16/context256/V4096, batch8/BF16/threads4, 2000 AdamW updates,
4,096,000 targets, lr.0006/warmup200/wd.1, clipping1. Identical prepared train
windows and full last-checkpoint validation. Tests unopened. No extra tuning
or recipe expansion after seeing partial results. Preserve run/protocol/source
identities and all unsuccessful results; export and refresh leaderboards.

Before training, check parameter counts, unchanged backbone, parent initialization
equivalence, new curve quadrature, CPU gradients and GPU finite training/backward
at actual shape. The new curve path uses unchanged proven kernels; compare it
with ordinary equations at CPU FP64 and CUDA FP32/BF16. Use original tolerances
(FP32 1e-5/3e-4; BF16 output .004/.02, gradients .0001/.05). Save errors honestly.
No resources-first rejection or kernel repair campaign in this phase.

For each variation report relative NLL change versus its parent on each corpus,
not only their average. Also show archived full SwiGLU and compact references as
context; those older records are not contemporaneous confirmation. Preserve actual
parameter mismatches. Training-run memory/time are observations, not isolated
speed claims. Complete all 12 runs and their mechanism interventions before
selecting finalists. Any exception/error receives a result.md rather than a score.

Rank the six recipes by the mean of their two relative NLL costs against full
native SwiGLU, with the worse corpus cost as tie-breaker and fewer FFN weights
next. Report the quality/memory tradeoff alongside this ranking. Keep at most two
finalists; a parent can win. Do not require or claim every variation improves.

## Confirmation and final choice

Before confirmation starts, freeze finalist recipes and parameter-matched ordinary
SwiGLU/GELU controls (round hidden sizes up, report mismatch), plus full native
SwiGLU and GELU. Repeat paired seeds211/307/401 on both corpora with the same
budget and no extra tuning. Re-run the relevant parent if not a finalist.
Report paired NLL differences and all seed results. Isolated resource comparisons
use three alternating rounds, 20 warmup/100 measured updates on the same machine;
no kernel changes. A new variant should improve its parent on both corpus means
without greater than 5% VRAM or time regression; otherwise report its tradeoff
and allow the parent to remain the preferred recipe. Compression stays >=70%.

Select one overall recipe only after these comparisons, using quality first among
the compact low-VRAM options, while reporting any conflict between datasets. If
no new variant reliably improves, select the best existing parent and say so.
The previous within-1%-of-full / 1%-better-than-compact targets remain stretch
comparisons, not a reason to deny the practical milestone already achieved.
Freeze the final selection before an independent seed509/8000-update validation
and one-time test evaluation against its parent and strongest compact/full controls.
No novelty claim from tuning established mechanisms. Save result.md for the
round and per-model variation results; keep leaderboard and Step 1 current.
