# Dataset leaderboards

Generated from recorded results. Lower validation NLL and perplexity are better.
Each table ranks completed runs on the **entire validation split**, using the last
checkpoint. Sampled training validation, test scores, numerical checks and resource
screens do not enter these rankings. Exact same-recipe/seed/loss replays rank once,
with all evidence retained and resource ranges shown. Different losses or seeds
remain separate. Replays are not independent seed confirmation. These are measurements,
not multi-seed confirmation or automatic research-success claims.

Backbone, data fingerprint, token budget, precision and training settings define
separate comparison groups. Source revisions and FFN recipes are preserved in the
linked evidence; grouping does not certify identical implementations. Do not compare
loss across datasets or treat a smaller-backbone table as a matched FFN comparison.

VRAM is peak **allocated training memory** from each run, including the whole model.
Training seconds exclude validation and are observational, not isolated speed benchmarks.
FFN reduction refers only to FFN weights, relative to the largest full SwiGLU
baseline in the same table. Total weights include the unchanged backbone.

Refresh: `uv run python -m main leaderboard`. Complete validation through the shared
trainer and result export both refresh this file automatically. Frozen experimental
launchers must export their compact evidence and then refresh the table.

Separate [dynamic-filter fresh held-out results](step_1_dynamic_composition.md#dynamic-filter-locked-fresh-held-out-evaluation)
are excluded from these validation rankings.

Separate [locked fresh held-out results](step_1_dynamic_composition.md#locked-fresh-held-out-evaluation)
are excluded from these validation rankings.

## TinyStories

### Width 512, 4 layers, 2,000 updates

16 heads; context 256; vocabulary 4096; RoPE 10000; 4,096,000 trained targets; 199,936 validation targets; bf16; batch 8; learning rate 0.0006; warmup 200; weight decay 0.1.
Data SHA-256: `c366ef855146266f94f599942af71b36eb576acc2b1363da4a5d6cd2f939d5ed`.

| Rank | Model / FFN recipe | Seed | NLL ↓ | Perplexity ↓ | Total weights | FFN weights | FFN reduction | Training VRAM MiB ↓ | Training seconds | Evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | full-swiglu (`swiglu_fused`, h=1376) | 307 | 2.534088 | 12.605 | 14,750,208 | 8,454,144 | 0.00% | 560.7 | 89.78 | [record](../records/compact-confirmation-v1-tinystories-full-swiglu-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 2 | full-swiglu-fused (`swiglu_fused`, h=1376) | 101 | 2.538486 | 12.660 | 14,750,208 | 8,454,144 | 0.00% | 557.5–560.7 | 85.46–108.14 | [record](../records/basis-readout-v1-tinystories-full-swiglu-fused.json) ([all 16 sources](#repeated-measurement-evidence)) |
| 3 | full-swiglu-kernel (`swiglu_kernel`, h=1376) | 101 | 2.544240 | 12.734 | 14,750,208 | 8,454,144 | 0.00% | 542.5 | 86.36–100.23 | [record](../records/basis-readout-v1-tinystories-full-swiglu-kernel.json) ([all 14 sources](#repeated-measurement-evidence)) |
| 4 | full-swiglu (`swiglu_fused`, h=1376) | 401 | 2.546830 | 12.767 | 14,750,208 | 8,454,144 | 0.00% | 560.7 | 89.59 | [record](../records/compact-confirmation-v1-tinystories-full-swiglu-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 5 | full-swiglu (`swiglu_fused`, h=1376) | 211 | 2.547593 | 12.776 | 14,750,208 | 8,454,144 | 0.00% | 560.7 | 89.64 | [record](../records/compact-confirmation-v1-tinystories-full-swiglu-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 6 | curve-wide (`self_curve_wide`, h=512) | 307 | 2.577930 | 13.170 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 92.31 | [record](../records/compact-confirmation-v1-tinystories-curve-wide-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 7 | full-gelu (`gelu`, h=2064) | 307 | 2.578456 | 13.177 | 14,750,208 | 8,454,144 | 0.00% | 539.4 | 96.79 | [record](../records/compact-confirmation-v1-tinystories-full-gelu-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 8 | untied-basis (`untied_swiglu`, h=608) | 101 | 2.586359 | 13.281 | 10,031,616 | 3,735,552 | 55.81% | 448.0 | 85.26 | [record](../records/shared-gate-v2-tinystories-untied-basis.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 9 | signed-parent (`signed_linear`, h=576) | 307 | 2.589474 | 13.323 | 8,786,432 | 2,490,368 | 70.54% | 412.4 | 86.09 | [record](../records/compact-confirmation-v1-tinystories-signed-parent-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 10 | curve-wide (`self_curve_wide`, h=512) | 101 | 2.591014 | 13.343 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 81.35 | [record](../records/compact-refinement-v1-tinystories-curve-wide.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 11 | full-gelu (`gelu`, h=2064) | 401 | 2.591216 | 13.346 | 14,750,208 | 8,454,144 | 0.00% | 539.4 | 89.06 | [record](../records/compact-confirmation-v1-tinystories-full-gelu-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 12 | curve-parent (`self_curve`, h=512) | 307 | 2.592524 | 13.363 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 84.91 | [record](../records/compact-confirmation-v1-tinystories-curve-parent-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 13 | full-gelu (`gelu`, h=2064) | 101 | 2.592983 | 13.370 | 14,750,208 | 8,454,144 | 0.00% | 539.4 | 84.03–89.71 | [record](../records/basis-readout-v1-tinystories-full-gelu.json) ([all 16 sources](#repeated-measurement-evidence)) |
| 14 | tied-basis (`tied_basis`, h=512) | 101 | 2.596579 | 13.418 | 8,405,504 | 2,109,440 | 75.05% | 402.6 | 99.99 | [record](../records/basis-readout-v1-tinystories-tied-basis.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 15 | signed-fine (`signed_linear`, h=512) | 307 | 2.596595 | 13.418 | 8,655,360 | 2,359,296 | 72.09% | 405.3 | 83.38 | [record](../records/compact-confirmation-v1-tinystories-signed-fine-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 16 | self-curve (`self_curve`, h=512) | 101 | 2.598287 | 13.441 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 87.13 | [record](../records/channel-curve-v2-tinystories-self-curve.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 17 | curve-parent (`self_curve`, h=512) | 101 | 2.598287 | 13.441 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 78.80 | [record](../records/compact-refinement-v1-tinystories-curve-parent.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 18 | basis-small-groups (`fixed_basis`, h=512) | 101 | 2.598418 | 13.442 | 8,589,824 | 2,293,760 | 72.87% | 403.2 | 90.72 | [record](../records/compact-refinement-v1-tinystories-basis-small-groups.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 19 | full-gelu (`gelu`, h=2064) | 211 | 2.598916 | 13.449 | 14,750,208 | 8,454,144 | 0.00% | 539.4 | 93.48 | [record](../records/compact-confirmation-v1-tinystories-full-gelu-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 20 | curve-wide (`self_curve_wide`, h=512) | 211 | 2.598962 | 13.450 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 80.37 | [record](../records/compact-confirmation-v1-tinystories-curve-wide-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 21 | curve-wide (`self_curve_wide`, h=512) | 401 | 2.600008 | 13.464 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 87.61 | [record](../records/compact-confirmation-v1-tinystories-curve-wide-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 22 | compact-swiglu-kernel-h384 (`swiglu_kernel`, h=384) | 307 | 2.602158 | 13.493 | 8,655,360 | 2,359,296 | 72.09% | 407.5 | 82.43 | [record](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 23 | signed-fine (`signed_linear`, h=512) | 101 | 2.602645 | 13.499 | 8,655,360 | 2,359,296 | 72.09% | 410.0 | 80.31 | [record](../records/compact-refinement-v1-tinystories-signed-fine.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 24 | shared-basis (`shared_swiglu`, h=608) | 101 | 2.603695 | 13.514 | 8,163,840 | 1,867,776 | 77.91% | 425.7 | 80.26 | [record](../records/shared-gate-v2-tinystories-shared-basis.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 25 | signed-parent (`signed_linear`, h=576) | 101 | 2.603935 | 13.517 | 8,786,432 | 2,490,368 | 70.54% | 414.3–415.9 | 71.64–85.52 | [record](../records/compact-refinement-v1-tinystories-signed-parent.json) ([all 4 sources](#repeated-measurement-evidence)) |
| 26 | fixed-basis (`fixed_basis`, h=512) | 101 | 2.604161 | 13.520 | 8,786,432 | 2,490,368 | 70.54% | 405.5 | 90.30–93.11 | [record](../records/basis-readout-v1-tinystories-fixed-basis.json) ([all 4 sources](#repeated-measurement-evidence)) |
| 27 | feature-diagonal (`feature_diagonal`, h=512) | 101 | 2.604415 | 13.523 | 8,395,264 | 2,099,200 | 75.17% | 403.2 | 91.88 | [record](../records/feature-flow-v1-tinystories-feature-diagonal.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 28 | signed-parent (`signed_linear`, h=576) | 401 | 2.604550 | 13.525 | 8,786,432 | 2,490,368 | 70.54% | 412.4 | 83.32 | [record](../records/compact-confirmation-v1-tinystories-signed-parent-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 29 | feature-plain (`feature_plain`, h=512) | 101 | 2.604789 | 13.528 | 8,393,216 | 2,097,152 | 75.19% | 403.2 | 85.50 | [record](../records/feature-flow-v1-tinystories-feature-plain.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 30 | curve-parent (`self_curve`, h=512) | 211 | 2.605385 | 13.536 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 84.14 | [record](../records/compact-confirmation-v1-tinystories-curve-parent-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 31 | feature-cached (`feature_cached`, h=512) | 101 | 2.606072 | 13.546 | 8,458,752 | 2,162,688 | 74.42% | 617.7 | 121.46 | [record](../records/feature-flow-v1-tinystories-feature-cached.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 32 | signed-fine (`signed_linear`, h=512) | 401 | 2.606209 | 13.548 | 8,655,360 | 2,359,296 | 72.09% | 405.3 | 87.35 | [record](../records/compact-confirmation-v1-tinystories-signed-fine-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 33 | narrow-swiglu-kernel (`swiglu_kernel`, h=408) | 101 | 2.606365 | 13.550 | 8,802,816 | 2,506,752 | 70.35% | 407.6 | 84.02–86.13 | [record](../records/basis-readout-v1-tinystories-narrow-swiglu-kernel.json) ([all 6 sources](#repeated-measurement-evidence)) |
| 34 | feature-flow (`feature_flow`, h=512) | 101 | 2.606496 | 13.551 | 8,458,752 | 2,162,688 | 74.42% | 403.7 | 89.16 | [record](../records/feature-flow-v1-tinystories-feature-flow.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 35 | feature-once (`feature_once`, h=512) | 101 | 2.607807 | 13.569 | 8,458,752 | 2,162,688 | 74.42% | 403.7 | 88.35 | [record](../records/feature-flow-v1-tinystories-feature-once.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 36 | signed-parent (`signed_linear`, h=576) | 211 | 2.608383 | 13.577 | 8,786,432 | 2,490,368 | 70.54% | 412.4 | 85.56 | [record](../records/compact-confirmation-v1-tinystories-signed-parent-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 37 | compact-swiglu-kernel-h384 (`swiglu_kernel`, h=384) | 401 | 2.609602 | 13.594 | 8,655,360 | 2,359,296 | 72.09% | 407.5 | 79.67 | [record](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 38 | compact-swiglu-kernel-h384 (`swiglu_kernel`, h=384) | 211 | 2.610283 | 13.603 | 8,655,360 | 2,359,296 | 72.09% | 407.5 | 83.36 | [record](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 39 | compact-gelu-h576 (`gelu`, h=576) | 307 | 2.610545 | 13.606 | 8,655,360 | 2,359,296 | 72.09% | 410.8 | 76.02 | [record](../records/compact-confirmation-v1-tinystories-compact-gelu-h576-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 40 | cached-basis (`cached_basis`, h=512) | 101 | 2.610949 | 13.612 | 8,786,432 | 2,490,368 | 70.54% | 607.5 | 97.46 | [record](../records/basis-readout-v1-tinystories-cached-basis.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 41 | compact-swiglu-kernel-h346 (`swiglu_kernel`, h=346) | 307 | 2.613471 | 13.646 | 8,421,888 | 2,125,824 | 74.85% | 406.1 | 84.24 | [record](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 42 | narrow-swiglu-kernel (`swiglu_kernel`, h=346) | 101 | 2.613739 | 13.650 | 8,421,888 | 2,125,824 | 74.85% | 405.8 | 83.84 | [record](../records/channel-curve-v2-tinystories-narrow-swiglu-kernel.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 43 | compact-swiglu-kernel-h346 (`swiglu_kernel`, h=346) | 401 | 2.613913 | 13.652 | 8,421,888 | 2,125,824 | 74.85% | 406.1 | 83.76 | [record](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 44 | narrow-swiglu-calibrated (`swiglu_kernel`, h=352) | 101 | 2.614896 | 13.666 | 8,458,752 | 2,162,688 | 74.42% | 406.2 | 84.55 | [record](../records/feature-flow-v1-tinystories-narrow-swiglu-calibrated.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 45 | narrow-swiglu-kernel (`swiglu_kernel`, h=368) | 101 | 2.615981 | 13.681 | 8,557,056 | 2,260,992 | 73.26% | 405.9 | 83.73 | [record](../records/context-v1-tinystories-narrow-swiglu-kernel.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 46 | narrow-gelu (`gelu`, h=608) | 101 | 2.616391 | 13.686 | 8,786,432 | 2,490,368 | 70.54% | 408.9 | 80.39–83.60 | [record](../records/basis-readout-v1-tinystories-narrow-gelu.json) ([all 6 sources](#repeated-measurement-evidence)) |
| 47 | narrow-swiglu-kernel (`swiglu_kernel`, h=344) | 101 | 2.617045 | 13.695 | 8,409,600 | 2,113,536 | 75.00% | 405.7 | 83.49 | [record](../records/group-v4-tinystories-narrow-swiglu-kernel.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 48 | signed-tanh (`signed_tanh`, h=576) | 101 | 2.617161 | 13.697 | 8,786,432 | 2,490,368 | 70.54% | 415.9 | 83.56 | [record](../records/signed-v1-tinystories-signed-tanh.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 49 | compact-swiglu-kernel-h346 (`swiglu_kernel`, h=346) | 211 | 2.617221 | 13.698 | 8,421,888 | 2,125,824 | 74.85% | 406.1 | 78.66 | [record](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 50 | curve-parent (`self_curve`, h=512) | 401 | 2.617525 | 13.702 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 82.01 | [record](../records/compact-confirmation-v1-tinystories-curve-parent-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 51 | signed-fine (`signed_linear`, h=512) | 211 | 2.618864 | 13.720 | 8,655,360 | 2,359,296 | 72.09% | 405.3 | 84.46 | [record](../records/compact-confirmation-v1-tinystories-signed-fine-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 52 | compact-gelu-h576 (`gelu`, h=576) | 401 | 2.619195 | 13.725 | 8,655,360 | 2,359,296 | 72.09% | 410.8 | 78.54 | [record](../records/compact-confirmation-v1-tinystories-compact-gelu-h576-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 53 | open-gain (`open_gain`, h=576) | 101 | 2.619272 | 13.726 | 8,788,736 | 2,492,672 | 70.52% | 415.9 | 87.08 | [record](../records/signed-v1-tinystories-open-gain.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 54 | compact-gelu-h518 (`gelu`, h=518) | 307 | 2.620505 | 13.743 | 8,417,792 | 2,121,728 | 74.90% | 409.0 | 75.94 | [record](../records/compact-confirmation-v1-tinystories-compact-gelu-h518-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 55 | compact-gelu-h576 (`gelu`, h=576) | 211 | 2.621967 | 13.763 | 8,655,360 | 2,359,296 | 72.09% | 410.8 | 79.05 | [record](../records/compact-confirmation-v1-tinystories-compact-gelu-h576-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 56 | narrow-gelu (`gelu`, h=518) | 101 | 2.622748 | 13.774 | 8,417,792 | 2,121,728 | 74.90% | 404.1 | 79.98 | [record](../records/channel-curve-v2-tinystories-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 57 | narrow-gelu (`gelu`, h=528) | 101 | 2.622865 | 13.775 | 8,458,752 | 2,162,688 | 74.42% | 404.4 | 81.10 | [record](../records/feature-flow-v1-tinystories-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 58 | narrow-swiglu-kernel (`swiglu_kernel`, h=352) | 101 | 2.622897 | 13.776 | 8,458,752 | 2,162,688 | 74.42% | 406.2 | 85.36 | [record](../records/feature-flow-v1-tinystories-narrow-swiglu-kernel.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 59 | narrow-gelu (`gelu`, h=544) | 101 | 2.623549 | 13.785 | 8,524,288 | 2,228,224 | 73.64% | 404.8 | 79.99 | [record](../records/context-v1-tinystories-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 60 | compact-gelu-h518 (`gelu`, h=518) | 211 | 2.625496 | 13.811 | 8,417,792 | 2,121,728 | 74.90% | 409.0 | 77.79 | [record](../records/compact-confirmation-v1-tinystories-compact-gelu-h518-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 61 | narrow-swiglu-fused (`swiglu_fused`, h=304) | 101 | 2.626184 | 13.821 | 8,163,840 | 1,867,776 | 77.91% | 402.3 | 82.92 | [record](../records/shared-gate-v2-tinystories-narrow-swiglu-fused.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 62 | channel-curve (`channel_curve`, h=512) | 101 | 2.626786 | 13.829 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 83.65 | [record](../records/channel-curve-v2-tinystories-channel-curve.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 63 | fixed-shape (`fixed_shape`, h=512) | 101 | 2.628363 | 13.851 | 8,405,504 | 2,109,440 | 75.05% | 401.1 | 84.12 | [record](../records/channel-curve-v2-tinystories-fixed-shape.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 64 | narrow-gelu (`gelu`, h=512) | 101 | 2.629526 | 13.867 | 8,393,216 | 2,097,152 | 75.19% | 400.9 | 79.82 | [record](../records/group-v4-tinystories-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 65 | narrow-gelu-learned-budget (`gelu`, h=611) | 101 | 2.629668 | 13.869 | 8,798,720 | 2,502,656 | 70.40% | 409.1 | 81.57 | [record](../records/basis-readout-v1-tinystories-narrow-gelu-learned-budget.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 66 | narrow-gelu (`gelu`, h=456) | 101 | 2.630634 | 13.883 | 8,163,840 | 1,867,776 | 77.91% | 397.4 | 79.63 | [record](../records/shared-gate-v2-tinystories-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 67 | compact-gelu-h518 (`gelu`, h=518) | 401 | 2.630823 | 13.885 | 8,417,792 | 2,121,728 | 74.90% | 409.0 | 80.87 | [record](../records/compact-confirmation-v1-tinystories-compact-gelu-h518-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 68 | square (`group_square`, h=512) | 101 | 2.645681 | 14.093 | 8,395,264 | 2,099,200 | 75.17% | 422.9 | 85.25 | [record](../records/group-v4-tinystories-square.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 69 | group-product (`group_product`, h=512) | 101 | 2.651852 | 14.180 | 8,395,264 | 2,099,200 | 75.17% | 401.9 | 82.85 | [record](../records/group-v4-tinystories-group-product.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 70 | plain-silu (`plain`, h=576) | 101 | 2.666190 | 14.385 | 8,655,360 | 2,359,296 | 72.09% | 406.7 | 78.77 | [record](../records/signed-v1-tinystories-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 71 | closed-gain (`closed_gain`, h=576) | 101 | 2.667904 | 14.410 | 8,788,736 | 2,492,672 | 70.52% | 415.9 | 87.36 | [record](../records/signed-v1-tinystories-closed-gain.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 72 | context-product (`context_product`, h=512) | 101 | 2.672406 | 14.475 | 8,526,336 | 2,230,272 | 73.62% | 408.2 | 85.75 | [record](../records/context-v1-tinystories-context-product.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 73 | plain-silu (`plain`, h=512) | 101 | 2.673912 | 14.497 | 8,393,216 | 2,097,152 | 75.19% | 400.9 | 79.47 | [record](../records/channel-curve-v2-tinystories-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 74 | plain-silu (`plain`, h=512) | 101 | 2.673912 | 14.497 | 8,393,216 | 2,097,152 | 75.19% | 400.9 | 79.52 | [record](../records/context-v1-tinystories-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 75 | plain-silu (`plain`, h=512) | 101 | 2.673912 | 14.497 | 8,393,216 | 2,097,152 | 75.19% | 400.9 | 78.53 | [record](../records/group-v4-tinystories-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 76 | context-additive (`context_additive`, h=512) | 101 | 2.674518 | 14.505 | 8,526,336 | 2,230,272 | 73.62% | 408.2 | 87.70 | [record](../records/context-v1-tinystories-context-additive.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 77 | thin-calibrated (`latent_matched_init`, h=512) | 101 | 2.701581 | 14.903 | 8,786,432 | 2,490,368 | 70.54% | 396.0 | 105.96 | [record](../records/latent-v1-tinystories-thin-calibrated.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 78 | thin-recompute (`latent_swiglu`, h=512) | 101 | 2.725223 | 15.260 | 8,786,432 | 2,490,368 | 70.54% | 396.0 | 93.23 | [record](../records/latent-v1-tinystories-thin-recompute.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 79 | wide-cached (`latent_native`, h=1280) | 101 | 2.726152 | 15.274 | 8,786,432 | 2,490,368 | 70.54% | 458.8 | 91.27 | [record](../records/latent-v1-tinystories-wide-cached.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 80 | wide-recompute (`latent_swiglu`, h=1280) | 101 | 2.726152 | 15.274 | 8,786,432 | 2,490,368 | 70.54% | 396.0 | 90.68 | [record](../records/latent-v1-tinystories-wide-recompute.json) ([all 2 sources](#repeated-measurement-evidence)) |

### Width 512, 4 layers, 8,000 updates

16 heads; context 256; vocabulary 4096; RoPE 10000; 16,384,000 trained targets; 199,936 validation targets; bf16; batch 8; learning rate 0.0006; warmup 200; weight decay 0.1.
Data SHA-256: `c366ef855146266f94f599942af71b36eb576acc2b1363da4a5d6cd2f939d5ed`.

| Rank | Model / FFN recipe | Seed | NLL ↓ | Perplexity ↓ | Total weights | FFN weights | FFN reduction | Training VRAM MiB ↓ | Training seconds | Evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | full-swiglu (`swiglu_fused`, h=1376) | 509 | 2.078244 | 7.990 | 14,750,208 | 8,454,144 | 0.00% | 560.7 | 394.55 | [record](../records/compact-final-v1-tinystories-full-swiglu-final.json) ([all 3 sources](#repeated-measurement-evidence)) |
| 2 | curve-wide (`self_curve_wide`, h=512) | 509 | 2.122985 | 8.356 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 375.17 | [record](../records/compact-final-v1-tinystories-curve-wide-final.json) ([all 3 sources](#repeated-measurement-evidence)) |
| 3 | curve-parent (`self_curve`, h=512) | 509 | 2.125866 | 8.380 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 374.00 | [record](../records/compact-final-v1-tinystories-curve-parent-final.json) ([all 3 sources](#repeated-measurement-evidence)) |
| 4 | compact-swiglu-kernel-h384 (`swiglu_kernel`, h=384) | 509 | 2.127756 | 8.396 | 8,655,360 | 2,359,296 | 72.09% | 407.5 | 365.92 | [record](../records/compact-final-v1-tinystories-compact-swiglu-kernel-h384-final.json) ([all 3 sources](#repeated-measurement-evidence)) |

### Width 192, 4 layers, 2,000 updates

6 heads; context 256; vocabulary 4096; RoPE 10000; 4,096,000 trained targets; 199,936 validation targets; bf16; batch 8; learning rate 0.0006; warmup 200; weight decay 0.1.
Data SHA-256: `c366ef855146266f94f599942af71b36eb576acc2b1363da4a5d6cd2f939d5ed`.

| Rank | Model / FFN recipe | Seed | NLL ↓ | Perplexity ↓ | Total weights | FFN weights | FFN reduction | Training VRAM MiB ↓ | Training seconds | Evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | full-swiglu (`swiglu`, h=512) | 101 | 2.883340 | 17.874 | 2,557,632 | 1,179,648 | 0.00% | 252.9 | 68.34 | [record](../records/language-v1-tinystories-full-swiglu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 2 | full-swiglu-fused (`swiglu_fused`, h=512) | 101 | 2.884440 | 17.894 | 2,557,632 | 1,179,648 | 0.00% | 249.9 | 69.85 | [record](../records/language-v1-tinystories-full-swiglu-fused.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 3 | blockshuffle (`swiglu`, h=1024) | 101 | 2.956800 | 19.236 | 1,728,192 | 350,208 | 70.31% | 284.7 | 98.13 | [record](../records/language-v1-tinystories-blockshuffle.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 4 | narrow-swiglu-fused (`swiglu_fused`, h=152) | 101 | 2.979796 | 19.684 | 1,728,192 | 350,208 | 70.31% | 216.6 | 66.18 | [record](../records/language-v1-tinystories-narrow-swiglu-fused.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 5 | full-gelu (`gelu`, h=768) | 101 | 2.979904 | 19.686 | 2,557,632 | 1,179,648 | 0.00% | 241.2 | 69.55 | [record](../records/language-v1-tinystories-full-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 6 | narrow-swiglu (`swiglu`, h=152) | 101 | 2.980331 | 19.694 | 1,728,192 | 350,208 | 70.31% | 218.0 | 60.01 | [record](../records/language-v1-tinystories-narrow-swiglu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 7 | maxout (`maxout`, h=112) | 101 | 2.996539 | 20.016 | 1,723,392 | 345,408 | 70.72% | 215.1 | 68.03 | [record](../records/language-v1-tinystories-maxout.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 8 | narrow-gelu (`gelu`, h=228) | 101 | 3.003410 | 20.154 | 1,728,192 | 350,208 | 70.31% | 212.6 | 66.01 | [record](../records/language-v1-tinystories-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 9 | curve-only (`pairflux_no_exchange`, h=224) | 101 | 3.063614 | 21.405 | 1,722,944 | 344,960 | 70.76% | 222.9 | 73.63 | [record](../records/language-v1-tinystories-curve-only.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 10 | plain-silu (`pairflux_plain`, h=224) | 101 | 3.067467 | 21.487 | 1,722,048 | 344,064 | 70.83% | 212.4 | 67.27 | [record](../records/language-v1-tinystories-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 11 | meanout (`meanout`, h=112) | 101 | 3.121888 | 22.689 | 1,723,392 | 345,408 | 70.72% | 208.1 | 50.51 | [record](../records/language-v1-tinystories-meanout.json) ([all 2 sources](#repeated-measurement-evidence)) |

## WikiText-2

### Width 512, 4 layers, 2,000 updates

16 heads; context 256; vocabulary 4096; RoPE 10000; 4,096,000 trained targets; 322,560 validation targets; bf16; batch 8; learning rate 0.0006; warmup 200; weight decay 0.1.
Data SHA-256: `3d5d350c132eab6e820675e00f041b509aa96e12bcdcb05a90721e19adc69e4b`.

| Rank | Model / FFN recipe | Seed | NLL ↓ | Perplexity ↓ | Total weights | FFN weights | FFN reduction | Training VRAM MiB ↓ | Training seconds | Evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | full-swiglu (`swiglu_fused`, h=1376) | 211 | 4.211107 | 67.431 | 14,750,208 | 8,454,144 | 0.00% | 560.7 | 91.41 | [record](../records/compact-confirmation-v1-wikitext-full-swiglu-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 2 | full-swiglu (`swiglu_fused`, h=1376) | 401 | 4.222958 | 68.235 | 14,750,208 | 8,454,144 | 0.00% | 560.7 | 97.09 | [record](../records/compact-confirmation-v1-wikitext-full-swiglu-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 3 | full-swiglu (`swiglu_fused`, h=1376) | 307 | 4.227179 | 68.524 | 14,750,208 | 8,454,144 | 0.00% | 560.7 | 101.94 | [record](../records/compact-confirmation-v1-wikitext-full-swiglu-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 4 | full-swiglu-fused (`swiglu_fused`, h=1376) | 101 | 4.237625 | 69.243 | 14,750,208 | 8,454,144 | 0.00% | 560.7 | 89.01–92.89 | [record](../records/basis-readout-v1-wikitext-full-swiglu-fused.json) ([all 16 sources](#repeated-measurement-evidence)) |
| 5 | full-swiglu-kernel (`swiglu_kernel`, h=1376) | 101 | 4.239026 | 69.340 | 14,750,208 | 8,454,144 | 0.00% | 540.5 | 86.59–109.15 | [record](../records/basis-readout-v1-wikitext-full-swiglu-kernel.json) ([all 14 sources](#repeated-measurement-evidence)) |
| 6 | full-gelu (`gelu`, h=2064) | 401 | 4.246259 | 69.844 | 14,750,208 | 8,454,144 | 0.00% | 539.4 | 93.67 | [record](../records/compact-confirmation-v1-wikitext-full-gelu-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 7 | full-gelu (`gelu`, h=2064) | 211 | 4.258114 | 70.677 | 14,750,208 | 8,454,144 | 0.00% | 539.4 | 89.91 | [record](../records/compact-confirmation-v1-wikitext-full-gelu-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 8 | full-gelu (`gelu`, h=2064) | 307 | 4.259446 | 70.771 | 14,750,208 | 8,454,144 | 0.00% | 539.4 | 93.94 | [record](../records/compact-confirmation-v1-wikitext-full-gelu-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 9 | signed-parent (`signed_linear`, h=576) | 401 | 4.259721 | 70.790 | 8,786,432 | 2,490,368 | 70.54% | 412.4 | 88.00 | [record](../records/compact-confirmation-v1-wikitext-signed-parent-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 10 | full-gelu (`gelu`, h=2064) | 101 | 4.269168 | 71.462 | 14,750,208 | 8,454,144 | 0.00% | 539.4 | 87.89–93.27 | [record](../records/basis-readout-v1-wikitext-full-gelu.json) ([all 16 sources](#repeated-measurement-evidence)) |
| 11 | signed-fine (`signed_linear`, h=512) | 307 | 4.270331 | 71.545 | 8,655,360 | 2,359,296 | 72.09% | 405.3 | 82.23 | [record](../records/compact-confirmation-v1-wikitext-signed-fine-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 12 | curve-wide (`self_curve_wide`, h=512) | 211 | 4.270796 | 71.579 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 87.24 | [record](../records/compact-confirmation-v1-wikitext-curve-wide-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 13 | signed-parent (`signed_linear`, h=576) | 211 | 4.271308 | 71.615 | 8,786,432 | 2,490,368 | 70.54% | 412.4 | 77.77 | [record](../records/compact-confirmation-v1-wikitext-signed-parent-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 14 | curve-wide (`self_curve_wide`, h=512) | 307 | 4.271972 | 71.663 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 92.23 | [record](../records/compact-confirmation-v1-wikitext-curve-wide-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 15 | signed-fine (`signed_linear`, h=512) | 401 | 4.272032 | 71.667 | 8,655,360 | 2,359,296 | 72.09% | 405.3 | 90.56 | [record](../records/compact-confirmation-v1-wikitext-signed-fine-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 16 | curve-wide (`self_curve_wide`, h=512) | 101 | 4.272187 | 71.678 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 67.95 | [record](../records/compact-refinement-v1-wikitext-curve-wide.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 17 | compact-swiglu-kernel-h346 (`swiglu_kernel`, h=346) | 401 | 4.272403 | 71.694 | 8,421,888 | 2,125,824 | 74.85% | 406.1 | 84.44 | [record](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 18 | untied-basis (`untied_swiglu`, h=608) | 101 | 4.272552 | 71.704 | 10,031,616 | 3,735,552 | 55.81% | 448.0 | 82.87 | [record](../records/shared-gate-v2-wikitext-untied-basis.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 19 | signed-fine (`signed_linear`, h=512) | 211 | 4.273048 | 71.740 | 8,655,360 | 2,359,296 | 72.09% | 405.3 | 85.10 | [record](../records/compact-confirmation-v1-wikitext-signed-fine-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 20 | curve-wide (`self_curve_wide`, h=512) | 401 | 4.275819 | 71.939 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 88.57 | [record](../records/compact-confirmation-v1-wikitext-curve-wide-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 21 | signed-fine (`signed_linear`, h=512) | 101 | 4.276089 | 71.958 | 8,655,360 | 2,359,296 | 72.09% | 410.0 | 68.37 | [record](../records/compact-refinement-v1-wikitext-signed-fine.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 22 | cached-basis (`cached_basis`, h=512) | 101 | 4.276874 | 72.015 | 8,786,432 | 2,490,368 | 70.54% | 607.5 | 97.54 | [record](../records/basis-readout-v1-wikitext-cached-basis.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 23 | signed-parent (`signed_linear`, h=576) | 101 | 4.277294 | 72.045 | 8,786,432 | 2,490,368 | 70.54% | 415.9 | 82.67–84.23 | [record](../records/compact-refinement-v1-wikitext-signed-parent.json) ([all 4 sources](#repeated-measurement-evidence)) |
| 24 | compact-swiglu-kernel-h384 (`swiglu_kernel`, h=384) | 401 | 4.277984 | 72.095 | 8,655,360 | 2,359,296 | 72.09% | 407.5 | 84.33 | [record](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 25 | curve-parent (`self_curve`, h=512) | 211 | 4.278996 | 72.168 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 81.38 | [record](../records/compact-confirmation-v1-wikitext-curve-parent-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 26 | curve-parent (`self_curve`, h=512) | 401 | 4.280770 | 72.296 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 86.81 | [record](../records/compact-confirmation-v1-wikitext-curve-parent-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 27 | compact-swiglu-kernel-h384 (`swiglu_kernel`, h=384) | 211 | 4.283336 | 72.482 | 8,655,360 | 2,359,296 | 72.09% | 407.5 | 71.00 | [record](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 28 | fixed-basis (`fixed_basis`, h=512) | 101 | 4.283474 | 72.492 | 8,786,432 | 2,490,368 | 70.54% | 405.5 | 86.77–92.13 | [record](../records/basis-readout-v1-wikitext-fixed-basis.json) ([all 4 sources](#repeated-measurement-evidence)) |
| 29 | signed-parent (`signed_linear`, h=576) | 307 | 4.283866 | 72.520 | 8,786,432 | 2,490,368 | 70.54% | 412.4 | 72.07 | [record](../records/compact-confirmation-v1-wikitext-signed-parent-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 30 | compact-swiglu-kernel-h384 (`swiglu_kernel`, h=384) | 307 | 4.285271 | 72.622 | 8,655,360 | 2,359,296 | 72.09% | 407.5 | 81.04 | [record](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 31 | compact-swiglu-kernel-h346 (`swiglu_kernel`, h=346) | 211 | 4.286545 | 72.715 | 8,421,888 | 2,125,824 | 74.85% | 406.1 | 83.42 | [record](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 32 | self-curve (`self_curve`, h=512) | 101 | 4.288112 | 72.829 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 89.17 | [record](../records/channel-curve-v2-wikitext-self-curve.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 33 | curve-parent (`self_curve`, h=512) | 101 | 4.288112 | 72.829 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 85.97 | [record](../records/compact-refinement-v1-wikitext-curve-parent.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 34 | narrow-swiglu-kernel (`swiglu_kernel`, h=408) | 101 | 4.288883 | 72.885 | 8,802,816 | 2,506,752 | 70.35% | 407.6 | 83.81–88.63 | [record](../records/basis-readout-v1-wikitext-narrow-swiglu-kernel.json) ([all 6 sources](#repeated-measurement-evidence)) |
| 35 | curve-parent (`self_curve`, h=512) | 307 | 4.289081 | 72.899 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 82.30 | [record](../records/compact-confirmation-v1-wikitext-curve-parent-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 36 | feature-cached (`feature_cached`, h=512) | 101 | 4.291654 | 73.087 | 8,458,752 | 2,162,688 | 74.42% | 617.7 | 122.03 | [record](../records/feature-flow-v1-wikitext-feature-cached.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 37 | feature-diagonal (`feature_diagonal`, h=512) | 101 | 4.293004 | 73.186 | 8,395,264 | 2,099,200 | 75.17% | 403.2 | 90.08 | [record](../records/feature-flow-v1-wikitext-feature-diagonal.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 38 | basis-small-groups (`fixed_basis`, h=512) | 101 | 4.293402 | 73.215 | 8,589,824 | 2,293,760 | 72.87% | 403.2 | 90.94 | [record](../records/compact-refinement-v1-wikitext-basis-small-groups.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 39 | feature-plain (`feature_plain`, h=512) | 101 | 4.296419 | 73.436 | 8,393,216 | 2,097,152 | 75.19% | 403.2 | 88.24 | [record](../records/feature-flow-v1-wikitext-feature-plain.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 40 | feature-once (`feature_once`, h=512) | 101 | 4.296649 | 73.453 | 8,458,752 | 2,162,688 | 74.42% | 403.7 | 89.74 | [record](../records/feature-flow-v1-wikitext-feature-once.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 41 | narrow-swiglu-kernel (`swiglu_kernel`, h=368) | 101 | 4.297109 | 73.487 | 8,557,056 | 2,260,992 | 73.26% | 405.9 | 86.35 | [record](../records/context-v1-wikitext-narrow-swiglu-kernel.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 42 | compact-gelu-h576 (`gelu`, h=576) | 401 | 4.297643 | 73.526 | 8,655,360 | 2,359,296 | 72.09% | 410.8 | 85.11 | [record](../records/compact-confirmation-v1-wikitext-compact-gelu-h576-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 43 | shared-basis (`shared_swiglu`, h=608) | 101 | 4.297722 | 73.532 | 8,163,840 | 1,867,776 | 77.91% | 425.7 | 82.24 | [record](../records/shared-gate-v2-wikitext-shared-basis.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 44 | channel-curve (`channel_curve`, h=512) | 101 | 4.298236 | 73.570 | 8,417,792 | 2,121,728 | 74.90% | 401.2 | 83.49 | [record](../records/channel-curve-v2-wikitext-channel-curve.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 45 | tied-basis (`tied_basis`, h=512) | 101 | 4.298645 | 73.600 | 8,405,504 | 2,109,440 | 75.05% | 402.6 | 99.49 | [record](../records/basis-readout-v1-wikitext-tied-basis.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 46 | feature-flow (`feature_flow`, h=512) | 101 | 4.300017 | 73.701 | 8,458,752 | 2,162,688 | 74.42% | 403.7 | 87.65 | [record](../records/feature-flow-v1-wikitext-feature-flow.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 47 | narrow-swiglu-kernel (`swiglu_kernel`, h=352) | 101 | 4.300213 | 73.715 | 8,458,752 | 2,162,688 | 74.42% | 406.2 | 84.58 | [record](../records/feature-flow-v1-wikitext-narrow-swiglu-kernel.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 48 | compact-gelu-h576 (`gelu`, h=576) | 307 | 4.301718 | 73.827 | 8,655,360 | 2,359,296 | 72.09% | 410.8 | 80.56 | [record](../records/compact-confirmation-v1-wikitext-compact-gelu-h576-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 49 | signed-tanh (`signed_tanh`, h=576) | 101 | 4.302946 | 73.917 | 8,786,432 | 2,490,368 | 70.54% | 415.9 | 85.78 | [record](../records/signed-v1-wikitext-signed-tanh.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 50 | compact-swiglu-kernel-h346 (`swiglu_kernel`, h=346) | 307 | 4.303409 | 73.951 | 8,421,888 | 2,125,824 | 74.85% | 406.1 | 80.36 | [record](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 51 | fixed-shape (`fixed_shape`, h=512) | 101 | 4.304249 | 74.014 | 8,405,504 | 2,109,440 | 75.05% | 401.1 | 86.10 | [record](../records/channel-curve-v2-wikitext-fixed-shape.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 52 | compact-gelu-h576 (`gelu`, h=576) | 211 | 4.304881 | 74.060 | 8,655,360 | 2,359,296 | 72.09% | 410.8 | 78.48 | [record](../records/compact-confirmation-v1-wikitext-compact-gelu-h576-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 53 | compact-gelu-h518 (`gelu`, h=518) | 401 | 4.307029 | 74.220 | 8,417,792 | 2,121,728 | 74.90% | 409.0 | 79.44 | [record](../records/compact-confirmation-v1-wikitext-compact-gelu-h518-s401.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 54 | narrow-swiglu-kernel (`swiglu_kernel`, h=346) | 101 | 4.308013 | 74.293 | 8,421,888 | 2,125,824 | 74.85% | 405.8 | 85.07 | [record](../records/channel-curve-v2-wikitext-narrow-swiglu-kernel.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 55 | narrow-swiglu-kernel (`swiglu_kernel`, h=344) | 101 | 4.308211 | 74.307 | 8,409,600 | 2,113,536 | 75.00% | 405.7 | 84.10 | [record](../records/group-v4-wikitext-narrow-swiglu-kernel.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 56 | open-gain (`open_gain`, h=576) | 101 | 4.309518 | 74.405 | 8,788,736 | 2,492,672 | 70.52% | 415.9 | 85.99 | [record](../records/signed-v1-wikitext-open-gain.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 57 | narrow-swiglu-calibrated (`swiglu_kernel`, h=352) | 101 | 4.309997 | 74.440 | 8,458,752 | 2,162,688 | 74.42% | 406.2 | 86.09 | [record](../records/feature-flow-v1-wikitext-narrow-swiglu-calibrated.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 58 | narrow-swiglu-fused (`swiglu_fused`, h=304) | 101 | 4.311688 | 74.566 | 8,163,840 | 1,867,776 | 77.91% | 402.3 | 84.77 | [record](../records/shared-gate-v2-wikitext-narrow-swiglu-fused.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 59 | narrow-gelu (`gelu`, h=512) | 101 | 4.312711 | 74.643 | 8,393,216 | 2,097,152 | 75.19% | 400.9 | 82.54 | [record](../records/group-v4-wikitext-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 60 | compact-gelu-h518 (`gelu`, h=518) | 307 | 4.315508 | 74.852 | 8,417,792 | 2,121,728 | 74.90% | 409.0 | 79.79 | [record](../records/compact-confirmation-v1-wikitext-compact-gelu-h518-s307.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 61 | narrow-gelu-learned-budget (`gelu`, h=611) | 101 | 4.316168 | 74.901 | 8,798,720 | 2,502,656 | 70.40% | 409.1 | 81.68 | [record](../records/basis-readout-v1-wikitext-narrow-gelu-learned-budget.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 62 | narrow-gelu (`gelu`, h=608) | 101 | 4.317237 | 74.981 | 8,786,432 | 2,490,368 | 70.54% | 408.9 | 82.54–83.97 | [record](../records/basis-readout-v1-wikitext-narrow-gelu.json) ([all 6 sources](#repeated-measurement-evidence)) |
| 63 | compact-gelu-h518 (`gelu`, h=518) | 211 | 4.320845 | 75.252 | 8,417,792 | 2,121,728 | 74.90% | 409.0 | 80.29 | [record](../records/compact-confirmation-v1-wikitext-compact-gelu-h518-s211.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 64 | narrow-gelu (`gelu`, h=544) | 101 | 4.321449 | 75.298 | 8,524,288 | 2,228,224 | 73.64% | 404.8 | 82.23 | [record](../records/context-v1-wikitext-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 65 | narrow-gelu (`gelu`, h=518) | 101 | 4.321492 | 75.301 | 8,417,792 | 2,121,728 | 74.90% | 404.1 | 81.67 | [record](../records/channel-curve-v2-wikitext-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 66 | square (`group_square`, h=512) | 101 | 4.326767 | 75.699 | 8,395,264 | 2,099,200 | 75.17% | 422.9 | 88.36 | [record](../records/group-v4-wikitext-square.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 67 | narrow-gelu (`gelu`, h=528) | 101 | 4.329683 | 75.920 | 8,458,752 | 2,162,688 | 74.42% | 404.4 | 81.19 | [record](../records/feature-flow-v1-wikitext-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 68 | narrow-gelu (`gelu`, h=456) | 101 | 4.330570 | 75.988 | 8,163,840 | 1,867,776 | 77.91% | 397.4 | 80.59 | [record](../records/shared-gate-v2-wikitext-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 69 | group-product (`group_product`, h=512) | 101 | 4.333677 | 76.224 | 8,395,264 | 2,099,200 | 75.17% | 401.9 | 86.12 | [record](../records/group-v4-wikitext-group-product.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 70 | context-additive (`context_additive`, h=512) | 101 | 4.365825 | 78.714 | 8,526,336 | 2,230,272 | 73.62% | 408.2 | 87.73 | [record](../records/context-v1-wikitext-context-additive.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 71 | context-product (`context_product`, h=512) | 101 | 4.367791 | 78.869 | 8,526,336 | 2,230,272 | 73.62% | 408.2 | 85.02 | [record](../records/context-v1-wikitext-context-product.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 72 | plain-silu (`plain`, h=512) | 101 | 4.369832 | 79.030 | 8,393,216 | 2,097,152 | 75.19% | 400.9 | 80.96 | [record](../records/channel-curve-v2-wikitext-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 73 | plain-silu (`plain`, h=512) | 101 | 4.369832 | 79.030 | 8,393,216 | 2,097,152 | 75.19% | 400.9 | 81.25 | [record](../records/context-v1-wikitext-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 74 | plain-silu (`plain`, h=512) | 101 | 4.369832 | 79.030 | 8,393,216 | 2,097,152 | 75.19% | 400.9 | 78.49 | [record](../records/group-v4-wikitext-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 75 | closed-gain (`closed_gain`, h=576) | 101 | 4.373279 | 79.303 | 8,788,736 | 2,492,672 | 70.52% | 415.9 | 85.83 | [record](../records/signed-v1-wikitext-closed-gain.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 76 | plain-silu (`plain`, h=576) | 101 | 4.377303 | 79.623 | 8,655,360 | 2,359,296 | 72.09% | 406.7 | 80.56 | [record](../records/signed-v1-wikitext-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 77 | wide-cached (`latent_native`, h=1280) | 101 | 4.400809 | 81.517 | 8,786,432 | 2,490,368 | 70.54% | 458.8 | 90.96 | [record](../records/latent-v1-wikitext-wide-cached.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 78 | wide-recompute (`latent_swiglu`, h=1280) | 101 | 4.400809 | 81.517 | 8,786,432 | 2,490,368 | 70.54% | 396.0 | 91.72 | [record](../records/latent-v1-wikitext-wide-recompute.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 79 | thin-calibrated (`latent_matched_init`, h=512) | 101 | 4.406500 | 81.982 | 8,786,432 | 2,490,368 | 70.54% | 396.0 | 91.06 | [record](../records/latent-v1-wikitext-thin-calibrated.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 80 | thin-recompute (`latent_swiglu`, h=512) | 101 | 4.438517 | 84.649 | 8,786,432 | 2,490,368 | 70.54% | 396.0 | 90.82 | [record](../records/latent-v1-wikitext-thin-recompute.json) ([all 2 sources](#repeated-measurement-evidence)) |

### Width 512, 4 layers, 8,000 updates

16 heads; context 256; vocabulary 4096; RoPE 10000; 16,384,000 trained targets; 322,560 validation targets; bf16; batch 8; learning rate 0.0006; warmup 200; weight decay 0.1.
Data SHA-256: `3d5d350c132eab6e820675e00f041b509aa96e12bcdcb05a90721e19adc69e4b`.

| Rank | Model / FFN recipe | Seed | NLL ↓ | Perplexity ↓ | Total weights | FFN weights | FFN reduction | Training VRAM MiB ↓ | Training seconds | Evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | curve-wide (`self_curve_wide`, h=512) | 509 | 3.942322 | 51.538 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 36970.15 | [record](../records/compact-final-v1-wikitext-curve-wide-final.json) ([all 3 sources](#repeated-measurement-evidence)) |
| 2 | compact-swiglu-kernel-h384 (`swiglu_kernel`, h=384) | 509 | 3.946195 | 51.738 | 8,655,360 | 2,359,296 | 72.09% | 407.5 | 329.36 | [record](../records/compact-final-v1-wikitext-compact-swiglu-kernel-h384-final.json) ([all 3 sources](#repeated-measurement-evidence)) |
| 3 | curve-parent (`self_curve`, h=512) | 509 | 3.962206 | 52.573 | 8,417,792 | 2,121,728 | 74.90% | 403.2 | 359.32 | [record](../records/compact-final-v1-wikitext-curve-parent-final.json) ([all 3 sources](#repeated-measurement-evidence)) |
| 4 | full-swiglu (`swiglu_fused`, h=1376) | 509 | 3.991952 | 54.160 | 14,750,208 | 8,454,144 | 0.00% | 560.7 | 415.16 | [record](../records/compact-final-v1-wikitext-full-swiglu-final.json) ([all 3 sources](#repeated-measurement-evidence)) |

### Width 192, 4 layers, 2,000 updates

6 heads; context 256; vocabulary 4096; RoPE 10000; 4,096,000 trained targets; 322,560 validation targets; bf16; batch 8; learning rate 0.0006; warmup 200; weight decay 0.1.
Data SHA-256: `3d5d350c132eab6e820675e00f041b509aa96e12bcdcb05a90721e19adc69e4b`.

| Rank | Model / FFN recipe | Seed | NLL ↓ | Perplexity ↓ | Total weights | FFN weights | FFN reduction | Training VRAM MiB ↓ | Training seconds | Evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | full-swiglu (`swiglu`, h=512) | 101 | 4.523139 | 92.124 | 2,557,632 | 1,179,648 | 0.00% | 253.9 | 71.89 | [record](../records/language-v1-wikitext-full-swiglu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 2 | full-swiglu-fused (`swiglu_fused`, h=512) | 101 | 4.523150 | 92.125 | 2,557,632 | 1,179,648 | 0.00% | 250.9 | 75.25 | [record](../records/language-v1-wikitext-full-swiglu-fused.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 3 | narrow-swiglu (`swiglu`, h=152) | 101 | 4.633178 | 102.840 | 1,728,192 | 350,208 | 70.31% | 218.9 | 78.83 | [record](../records/language-v1-wikitext-narrow-swiglu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 4 | narrow-swiglu-fused (`swiglu_fused`, h=152) | 101 | 4.633594 | 102.883 | 1,728,192 | 350,208 | 70.31% | 217.5 | 45.92 | [record](../records/language-v1-wikitext-narrow-swiglu-fused.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 5 | blockshuffle (`swiglu`, h=1024) | 101 | 4.637398 | 103.275 | 1,728,192 | 350,208 | 70.31% | 284.7 | 123.26 | [record](../records/language-v1-wikitext-blockshuffle.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 6 | full-gelu (`gelu`, h=768) | 101 | 4.641271 | 103.676 | 2,557,632 | 1,179,648 | 0.00% | 242.1 | 73.00 | [record](../records/language-v1-wikitext-full-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 7 | narrow-gelu (`gelu`, h=228) | 101 | 4.677101 | 107.458 | 1,728,192 | 350,208 | 70.31% | 213.5 | 69.12 | [record](../records/language-v1-wikitext-narrow-gelu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 8 | maxout (`maxout`, h=112) | 101 | 4.684398 | 108.245 | 1,723,392 | 345,408 | 70.72% | 215.1 | 79.02 | [record](../records/language-v1-wikitext-maxout.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 9 | curve-only (`pairflux_no_exchange`, h=224) | 101 | 4.728093 | 113.080 | 1,722,944 | 344,960 | 70.76% | 223.9 | 84.00 | [record](../records/language-v1-wikitext-curve-only.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 10 | plain-silu (`pairflux_plain`, h=224) | 101 | 4.740294 | 114.468 | 1,722,048 | 344,064 | 70.83% | 213.3 | 89.27 | [record](../records/language-v1-wikitext-plain-silu.json) ([all 2 sources](#repeated-measurement-evidence)) |
| 11 | meanout (`meanout`, h=112) | 101 | 4.819802 | 123.941 | 1,723,392 | 345,408 | 70.72% | 208.1 | 76.73 | [record](../records/language-v1-wikitext-meanout.json) ([all 2 sources](#repeated-measurement-evidence)) |

## Tested recipes without a ranked language result

These are not assigned invented language scores. Their reports retain numerical,
resource or synthetic-task results. Execution revisions are not new mathematical models.

| Recipe | Evidence / status |
| --- | --- |
| Step 2 paragraph compressor | [Reconstruction study](step_2_results.md); target-conditioned autoencoding, deliberately excluded from language-NLL rankings |
| Curve-Wide plain PyTorch v1 | [Numerical and resource checks](../src/models/channel_curve_transformer/result.md); no new language run; ranked Curve-Wide scores above use historical optimized execution |
| PairFlux (combined), no-exchange, no-curve | [Synthetic-task screen](../dump/old-research-docs/retired_models/pairflux_transformer/result.md); curve-only has separate language rows above |
| Looped BlockShuffle / repeated dense | [Synthetic and execution checks](../dump/old-research-docs/retired_models/looped_blockshuffle_transformer/result.md) |
| Shared-basis v1 / gate transport v2 | [Resource screens](../dump/old-research-docs/shared_gate_transport_hypothesis.md); plain sharing and untied controls have language rows above |
| Polynomial sketch / single sketch | [Resource failures](../dump/old-research-docs/polynomial_sketch_result.md) |
| Group-product execution v1–v3 | [Resource revisions](../dump/old-research-docs/group_product_result.md); v4 language rows above |
| Channel-curve execution v1 | [Resource failure](../dump/old-research-docs/channel_curve_result.md); v2 language rows above |
| Learned basis templates | [Resource failure](../dump/old-research-docs/basis_readout_result.md); fixed/tied/cached language rows above |
| Feature-flow execution v1–v3 | [Resource revisions](../dump/old-research-docs/feature_flow_result.md); v4 language rows above |
| Readout reuse v1 | [Resource failure](../dump/old-research-docs/readout_reuse_result.md) |
| Readout reuse v2 / v3 | [Forward failure](../dump/old-research-docs/readout_reuse_v2_result.md) / [gradient failure](../dump/old-research-docs/readout_reuse_v3_result.md) |
| Readout reuse v4 | [Gradient failure](../dump/old-research-docs/readout_reuse_v4_result.md); retired before language |
| Readout reuse v5 | [Resource failure](../dump/old-research-docs/readout_reuse_v5_result.md); numerical checks passed, but slowdown exceeds the gate; no language result |
| Paired readout v1 | [Resource report](../dump/old-research-docs/paired_readout_result.md); numerical checks passed; misses all-control memory/speed gate, no language score |

Frozen-weight removal experiments stay in each study's result report; they are
interventions on trained checkpoints, not independently trained model entries.

## Repeated measurement evidence

<details>
<summary>All grouped sources</summary>

- basis-readout-v1-tinystories-cached-basis-s101-lr0.0006: [basis-readout-v1-tinystories-cached-basis.json](../records/basis-readout-v1-tinystories-cached-basis.json), [summary.json](../dump/runs/20261004T111049Z-083fa21c/summary.json)
- basis-readout-v1-tinystories-fixed-basis-s101-lr0.0006: [basis-readout-v1-tinystories-fixed-basis.json](../records/basis-readout-v1-tinystories-fixed-basis.json), [compact-refinement-v1-tinystories-basis-parent.json](../records/compact-refinement-v1-tinystories-basis-parent.json), [summary.json](../dump/runs/20261004T110851Z-ae0332e4/summary.json), [summary.json](../dump/runs/20261005T020300Z-9520529a/summary.json)
- basis-readout-v1-tinystories-full-gelu-s101-lr0.0006: [basis-readout-v1-tinystories-full-gelu.json](../records/basis-readout-v1-tinystories-full-gelu.json), [channel-curve-v2-tinystories-full-gelu.json](../records/channel-curve-v2-tinystories-full-gelu.json), [context-v1-tinystories-full-gelu.json](../records/context-v1-tinystories-full-gelu.json), [feature-flow-v1-tinystories-full-gelu.json](../records/feature-flow-v1-tinystories-full-gelu.json), [group-v4-tinystories-full-gelu.json](../records/group-v4-tinystories-full-gelu.json), [latent-v1-tinystories-full-gelu.json](../records/latent-v1-tinystories-full-gelu.json), [shared-gate-v2-tinystories-full-gelu.json](../records/shared-gate-v2-tinystories-full-gelu.json), [signed-v1-tinystories-full-gelu.json](../records/signed-v1-tinystories-full-gelu.json), [summary.json](../dump/runs/20261004T052232Z-b95d99e8/summary.json), [summary.json](../dump/runs/20261004T064725Z-9bb7b714/summary.json), [summary.json](../dump/runs/20261004T073311Z-e28c3cc0/summary.json), [summary.json](../dump/runs/20261004T081903Z-c6e0cd7b/summary.json), [summary.json](../dump/runs/20261004T090912Z-22a44a28/summary.json), [summary.json](../dump/runs/20261004T101002Z-ecd163b4/summary.json), [summary.json](../dump/runs/20261004T110159Z-e42652ce/summary.json), [summary.json](../dump/runs/20261004T130306Z-cb536ca0/summary.json)
- basis-readout-v1-tinystories-full-swiglu-fused-s101-lr0.0006: [basis-readout-v1-tinystories-full-swiglu-fused.json](../records/basis-readout-v1-tinystories-full-swiglu-fused.json), [channel-curve-v2-tinystories-full-swiglu-fused.json](../records/channel-curve-v2-tinystories-full-swiglu-fused.json), [context-v1-tinystories-full-swiglu-fused.json](../records/context-v1-tinystories-full-swiglu-fused.json), [feature-flow-v1-tinystories-full-swiglu-fused.json](../records/feature-flow-v1-tinystories-full-swiglu-fused.json), [group-v4-tinystories-full-swiglu-fused.json](../records/group-v4-tinystories-full-swiglu-fused.json), [latent-v1-tinystories-full-swiglu-fused.json](../records/latent-v1-tinystories-full-swiglu-fused.json), [shared-gate-v2-tinystories-full-swiglu-fused.json](../records/shared-gate-v2-tinystories-full-swiglu-fused.json), [signed-v1-tinystories-full-swiglu-fused.json](../records/signed-v1-tinystories-full-swiglu-fused.json), [summary.json](../dump/runs/20261004T052042Z-26530856/summary.json), [summary.json](../dump/runs/20261004T064541Z-b76baf35/summary.json), [summary.json](../dump/runs/20261004T073123Z-bd32952e/summary.json), [summary.json](../dump/runs/20261004T081652Z-b2523ff8/summary.json), [summary.json](../dump/runs/20261004T090724Z-b2ba4307/summary.json), [summary.json](../dump/runs/20261004T100813Z-cebe6414/summary.json), [summary.json](../dump/runs/20261004T110010Z-f8801bbc/summary.json), [summary.json](../dump/runs/20261004T130115Z-22d937af/summary.json)
- basis-readout-v1-tinystories-full-swiglu-kernel-s101-lr0.0006: [basis-readout-v1-tinystories-full-swiglu-kernel.json](../records/basis-readout-v1-tinystories-full-swiglu-kernel.json), [channel-curve-v2-tinystories-full-swiglu-kernel.json](../records/channel-curve-v2-tinystories-full-swiglu-kernel.json), [context-v1-tinystories-full-swiglu-kernel.json](../records/context-v1-tinystories-full-swiglu-kernel.json), [feature-flow-v1-tinystories-full-swiglu-kernel.json](../records/feature-flow-v1-tinystories-full-swiglu-kernel.json), [group-v4-tinystories-full-swiglu-kernel.json](../records/group-v4-tinystories-full-swiglu-kernel.json), [latent-v1-tinystories-full-swiglu-kernel.json](../records/latent-v1-tinystories-full-swiglu-kernel.json), [signed-v1-tinystories-full-swiglu-kernel.json](../records/signed-v1-tinystories-full-swiglu-kernel.json), [summary.json](../dump/runs/20261004T064352Z-069a0b98/summary.json), [summary.json](../dump/runs/20261004T072932Z-b0c52026/summary.json), [summary.json](../dump/runs/20261004T081449Z-1a69fe60/summary.json), [summary.json](../dump/runs/20261004T090532Z-95a8f28b/summary.json), [summary.json](../dump/runs/20261004T100627Z-492200fa/summary.json), [summary.json](../dump/runs/20261004T105821Z-4cf78bc0/summary.json), [summary.json](../dump/runs/20261004T125920Z-30adc55e/summary.json)
- basis-readout-v1-tinystories-narrow-gelu-learned-budget-s101-lr0.0006: [basis-readout-v1-tinystories-narrow-gelu-learned-budget.json](../records/basis-readout-v1-tinystories-narrow-gelu-learned-budget.json), [summary.json](../dump/runs/20261004T110712Z-f4fa7598/summary.json)
- basis-readout-v1-tinystories-narrow-gelu-s101-lr0.0006: [basis-readout-v1-tinystories-narrow-gelu.json](../records/basis-readout-v1-tinystories-narrow-gelu.json), [latent-v1-tinystories-narrow-gelu.json](../records/latent-v1-tinystories-narrow-gelu.json), [signed-v1-tinystories-narrow-gelu.json](../records/signed-v1-tinystories-narrow-gelu.json), [summary.json](../dump/runs/20261004T082234Z-bbae11e0/summary.json), [summary.json](../dump/runs/20261004T091246Z-f7cc5206/summary.json), [summary.json](../dump/runs/20261004T110531Z-652dacb1/summary.json)
- basis-readout-v1-tinystories-narrow-swiglu-kernel-s101-lr0.0006: [basis-readout-v1-tinystories-narrow-swiglu-kernel.json](../records/basis-readout-v1-tinystories-narrow-swiglu-kernel.json), [latent-v1-tinystories-narrow-swiglu-kernel.json](../records/latent-v1-tinystories-narrow-swiglu-kernel.json), [signed-v1-tinystories-narrow-swiglu-kernel.json](../records/signed-v1-tinystories-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T082052Z-f43aa655/summary.json), [summary.json](../dump/runs/20261004T091101Z-0c3fc8a7/summary.json), [summary.json](../dump/runs/20261004T110348Z-96c7bbc0/summary.json)
- basis-readout-v1-tinystories-tied-basis-s101-lr0.0006: [basis-readout-v1-tinystories-tied-basis.json](../records/basis-readout-v1-tinystories-tied-basis.json), [summary.json](../dump/runs/20261004T111254Z-dbc1ed45/summary.json)
- basis-readout-v1-wikitext-cached-basis-s101-lr0.0006: [basis-readout-v1-wikitext-cached-basis.json](../records/basis-readout-v1-wikitext-cached-basis.json), [summary.json](../dump/runs/20261004T112758Z-0831ba98/summary.json)
- basis-readout-v1-wikitext-fixed-basis-s101-lr0.0006: [basis-readout-v1-wikitext-fixed-basis.json](../records/basis-readout-v1-wikitext-fixed-basis.json), [compact-refinement-v1-wikitext-basis-parent.json](../records/compact-refinement-v1-wikitext-basis-parent.json), [summary.json](../dump/runs/20261004T112555Z-8c76d753/summary.json), [summary.json](../dump/runs/20261005T021343Z-0b486434/summary.json)
- basis-readout-v1-wikitext-full-gelu-s101-lr0.0006: [basis-readout-v1-wikitext-full-gelu.json](../records/basis-readout-v1-wikitext-full-gelu.json), [channel-curve-v2-wikitext-full-gelu.json](../records/channel-curve-v2-wikitext-full-gelu.json), [context-v1-wikitext-full-gelu.json](../records/context-v1-wikitext-full-gelu.json), [feature-flow-v1-wikitext-full-gelu.json](../records/feature-flow-v1-wikitext-full-gelu.json), [group-v4-wikitext-full-gelu.json](../records/group-v4-wikitext-full-gelu.json), [latent-v1-wikitext-full-gelu.json](../records/latent-v1-wikitext-full-gelu.json), [shared-gate-v2-wikitext-full-gelu.json](../records/shared-gate-v2-wikitext-full-gelu.json), [signed-v1-wikitext-full-gelu.json](../records/signed-v1-wikitext-full-gelu.json), [summary.json](../dump/runs/20261004T053249Z-534b881d/summary.json), [summary.json](../dump/runs/20261004T070112Z-6a633099/summary.json), [summary.json](../dump/runs/20261004T074710Z-9c233f10/summary.json), [summary.json](../dump/runs/20261004T083642Z-efb74959/summary.json), [summary.json](../dump/runs/20261004T092623Z-b143edfd/summary.json), [summary.json](../dump/runs/20261004T102547Z-3a054295/summary.json), [summary.json](../dump/runs/20261004T111851Z-d74501d3/summary.json), [summary.json](../dump/runs/20261004T132348Z-ece46053/summary.json)
- basis-readout-v1-wikitext-full-swiglu-fused-s101-lr0.0006: [basis-readout-v1-wikitext-full-swiglu-fused.json](../records/basis-readout-v1-wikitext-full-swiglu-fused.json), [channel-curve-v2-wikitext-full-swiglu-fused.json](../records/channel-curve-v2-wikitext-full-swiglu-fused.json), [context-v1-wikitext-full-swiglu-fused.json](../records/context-v1-wikitext-full-swiglu-fused.json), [feature-flow-v1-wikitext-full-swiglu-fused.json](../records/feature-flow-v1-wikitext-full-swiglu-fused.json), [group-v4-wikitext-full-swiglu-fused.json](../records/group-v4-wikitext-full-swiglu-fused.json), [latent-v1-wikitext-full-swiglu-fused.json](../records/latent-v1-wikitext-full-swiglu-fused.json), [shared-gate-v2-wikitext-full-swiglu-fused.json](../records/shared-gate-v2-wikitext-full-swiglu-fused.json), [signed-v1-wikitext-full-swiglu-fused.json](../records/signed-v1-wikitext-full-swiglu-fused.json), [summary.json](../dump/runs/20261004T053058Z-fcdd14d8/summary.json), [summary.json](../dump/runs/20261004T065921Z-277a92bd/summary.json), [summary.json](../dump/runs/20261004T074516Z-b05dbf59/summary.json), [summary.json](../dump/runs/20261004T083447Z-e1a2782c/summary.json), [summary.json](../dump/runs/20261004T092428Z-547324df/summary.json), [summary.json](../dump/runs/20261004T102356Z-eb545053/summary.json), [summary.json](../dump/runs/20261004T111656Z-cb9b24b6/summary.json), [summary.json](../dump/runs/20261004T132151Z-81b29b25/summary.json)
- basis-readout-v1-wikitext-full-swiglu-kernel-s101-lr0.0006: [basis-readout-v1-wikitext-full-swiglu-kernel.json](../records/basis-readout-v1-wikitext-full-swiglu-kernel.json), [channel-curve-v2-wikitext-full-swiglu-kernel.json](../records/channel-curve-v2-wikitext-full-swiglu-kernel.json), [context-v1-wikitext-full-swiglu-kernel.json](../records/context-v1-wikitext-full-swiglu-kernel.json), [feature-flow-v1-wikitext-full-swiglu-kernel.json](../records/feature-flow-v1-wikitext-full-swiglu-kernel.json), [group-v4-wikitext-full-swiglu-kernel.json](../records/group-v4-wikitext-full-swiglu-kernel.json), [latent-v1-wikitext-full-swiglu-kernel.json](../records/latent-v1-wikitext-full-swiglu-kernel.json), [signed-v1-wikitext-full-swiglu-kernel.json](../records/signed-v1-wikitext-full-swiglu-kernel.json), [summary.json](../dump/runs/20261004T065730Z-0069be3d/summary.json), [summary.json](../dump/runs/20261004T074326Z-01915a7d/summary.json), [summary.json](../dump/runs/20261004T083253Z-01dcf7c3/summary.json), [summary.json](../dump/runs/20261004T092213Z-41bc1418/summary.json), [summary.json](../dump/runs/20261004T102204Z-c03d9baa/summary.json), [summary.json](../dump/runs/20261004T111503Z-939f07a6/summary.json), [summary.json](../dump/runs/20261004T131956Z-2bc0e9c6/summary.json)
- basis-readout-v1-wikitext-narrow-gelu-learned-budget-s101-lr0.0006: [basis-readout-v1-wikitext-narrow-gelu-learned-budget.json](../records/basis-readout-v1-wikitext-narrow-gelu-learned-budget.json), [summary.json](../dump/runs/20261004T112413Z-f98c24fa/summary.json)
- basis-readout-v1-wikitext-narrow-gelu-s101-lr0.0006: [basis-readout-v1-wikitext-narrow-gelu.json](../records/basis-readout-v1-wikitext-narrow-gelu.json), [latent-v1-wikitext-narrow-gelu.json](../records/latent-v1-wikitext-narrow-gelu.json), [signed-v1-wikitext-narrow-gelu.json](../records/signed-v1-wikitext-narrow-gelu.json), [summary.json](../dump/runs/20261004T084017Z-b0252e08/summary.json), [summary.json](../dump/runs/20261004T093005Z-76636aff/summary.json), [summary.json](../dump/runs/20261004T112229Z-4ce8492c/summary.json)
- basis-readout-v1-wikitext-narrow-swiglu-kernel-s101-lr0.0006: [basis-readout-v1-wikitext-narrow-swiglu-kernel.json](../records/basis-readout-v1-wikitext-narrow-swiglu-kernel.json), [latent-v1-wikitext-narrow-swiglu-kernel.json](../records/latent-v1-wikitext-narrow-swiglu-kernel.json), [signed-v1-wikitext-narrow-swiglu-kernel.json](../records/signed-v1-wikitext-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T083833Z-7a9ad339/summary.json), [summary.json](../dump/runs/20261004T092815Z-ad913694/summary.json), [summary.json](../dump/runs/20261004T112043Z-5bd63b20/summary.json)
- basis-readout-v1-wikitext-tied-basis-s101-lr0.0006: [basis-readout-v1-wikitext-tied-basis.json](../records/basis-readout-v1-wikitext-tied-basis.json), [summary.json](../dump/runs/20261004T113010Z-5cd3ee27/summary.json)
- channel-curve-v2-tinystories-channel-curve-s101-lr0.0006: [channel-curve-v2-tinystories-channel-curve.json](../records/channel-curve-v2-tinystories-channel-curve.json), [summary.json](../dump/runs/20261004T101510Z-22788d2f/summary.json)
- channel-curve-v2-tinystories-fixed-shape-s101-lr0.0006: [channel-curve-v2-tinystories-fixed-shape.json](../records/channel-curve-v2-tinystories-fixed-shape.json), [summary.json](../dump/runs/20261004T101841Z-c5e2ba47/summary.json)
- channel-curve-v2-tinystories-narrow-gelu-s101-lr0.0006: [channel-curve-v2-tinystories-narrow-gelu.json](../records/channel-curve-v2-tinystories-narrow-gelu.json), [summary.json](../dump/runs/20261004T101333Z-34d81762/summary.json)
- channel-curve-v2-tinystories-narrow-swiglu-kernel-s101-lr0.0006: [channel-curve-v2-tinystories-narrow-swiglu-kernel.json](../records/channel-curve-v2-tinystories-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T101152Z-ccdf5cc4/summary.json)
- channel-curve-v2-tinystories-plain-silu-s101-lr0.0006: [channel-curve-v2-tinystories-plain-silu.json](../records/channel-curve-v2-tinystories-plain-silu.json), [summary.json](../dump/runs/20261004T102026Z-44218b79/summary.json)
- channel-curve-v2-tinystories-self-curve-s101-lr0.0006: [channel-curve-v2-tinystories-self-curve.json](../records/channel-curve-v2-tinystories-self-curve.json), [summary.json](../dump/runs/20261004T101654Z-822406ce/summary.json)
- channel-curve-v2-wikitext-channel-curve-s101-lr0.0006: [channel-curve-v2-wikitext-channel-curve.json](../records/channel-curve-v2-wikitext-channel-curve.json), [summary.json](../dump/runs/20261004T103104Z-d3d2e4b6/summary.json)
- channel-curve-v2-wikitext-fixed-shape-s101-lr0.0006: [channel-curve-v2-wikitext-fixed-shape.json](../records/channel-curve-v2-wikitext-fixed-shape.json), [summary.json](../dump/runs/20261004T103445Z-9a2f5a2b/summary.json)
- channel-curve-v2-wikitext-narrow-gelu-s101-lr0.0006: [channel-curve-v2-wikitext-narrow-gelu.json](../records/channel-curve-v2-wikitext-narrow-gelu.json), [summary.json](../dump/runs/20261004T102923Z-8a37de59/summary.json)
- channel-curve-v2-wikitext-narrow-swiglu-kernel-s101-lr0.0006: [channel-curve-v2-wikitext-narrow-swiglu-kernel.json](../records/channel-curve-v2-wikitext-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T102738Z-934e1b1c/summary.json)
- channel-curve-v2-wikitext-plain-silu-s101-lr0.0006: [channel-curve-v2-wikitext-plain-silu.json](../records/channel-curve-v2-wikitext-plain-silu.json), [summary.json](../dump/runs/20261004T103636Z-97ef077f/summary.json)
- channel-curve-v2-wikitext-self-curve-s101-lr0.0006: [channel-curve-v2-wikitext-self-curve.json](../records/channel-curve-v2-wikitext-self-curve.json), [summary.json](../dump/runs/20261004T103251Z-b92ebda0/summary.json)
- compact-confirmation-v1-tinystories-compact-gelu-h518-s211-lr0.0006: [compact-confirmation-v1-tinystories-compact-gelu-h518-s211.json](../records/compact-confirmation-v1-tinystories-compact-gelu-h518-s211.json), [summary.json](../dump/runs/20261005T022518Z-2c12b705/summary.json)
- compact-confirmation-v1-tinystories-compact-gelu-h518-s307-lr0.0006: [compact-confirmation-v1-tinystories-compact-gelu-h518-s307.json](../records/compact-confirmation-v1-tinystories-compact-gelu-h518-s307.json), [summary.json](../dump/runs/20261005T024220Z-0464cca4/summary.json)
- compact-confirmation-v1-tinystories-compact-gelu-h518-s401-lr0.0006: [compact-confirmation-v1-tinystories-compact-gelu-h518-s401.json](../records/compact-confirmation-v1-tinystories-compact-gelu-h518-s401.json), [summary.json](../dump/runs/20261005T025918Z-84203a29/summary.json)
- compact-confirmation-v1-tinystories-compact-gelu-h576-s211-lr0.0006: [compact-confirmation-v1-tinystories-compact-gelu-h576-s211.json](../records/compact-confirmation-v1-tinystories-compact-gelu-h576-s211.json), [summary.json](../dump/runs/20261005T023012Z-50bd38ea/summary.json)
- compact-confirmation-v1-tinystories-compact-gelu-h576-s307-lr0.0006: [compact-confirmation-v1-tinystories-compact-gelu-h576-s307.json](../records/compact-confirmation-v1-tinystories-compact-gelu-h576-s307.json), [summary.json](../dump/runs/20261005T024712Z-21bed936/summary.json)
- compact-confirmation-v1-tinystories-compact-gelu-h576-s401-lr0.0006: [compact-confirmation-v1-tinystories-compact-gelu-h576-s401.json](../records/compact-confirmation-v1-tinystories-compact-gelu-h576-s401.json), [summary.json](../dump/runs/20261005T030410Z-5d451c09/summary.json)
- compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s211-lr0.0006: [compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s211.json](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s211.json), [summary.json](../dump/runs/20261005T022343Z-7009f10e/summary.json)
- compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s307-lr0.0006: [compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s307.json](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s307.json), [summary.json](../dump/runs/20261005T024039Z-534ac4c4/summary.json)
- compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s401-lr0.0006: [compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s401.json](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h346-s401.json), [summary.json](../dump/runs/20261005T025737Z-5f4e61f9/summary.json)
- compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s211-lr0.0006: [compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s211.json](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s211.json), [summary.json](../dump/runs/20261005T022833Z-16244624/summary.json)
- compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s307-lr0.0006: [compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s307.json](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s307.json), [summary.json](../dump/runs/20261005T024534Z-6ce292b0/summary.json)
- compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s401-lr0.0006: [compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s401.json](../records/compact-confirmation-v1-tinystories-compact-swiglu-kernel-h384-s401.json), [summary.json](../dump/runs/20261005T030234Z-3a9d8ca3/summary.json)
- compact-confirmation-v1-tinystories-curve-parent-s211-lr0.0006: [compact-confirmation-v1-tinystories-curve-parent-s211.json](../records/compact-confirmation-v1-tinystories-curve-parent-s211.json), [summary.json](../dump/runs/20261005T022204Z-e43f9f14/summary.json)
- compact-confirmation-v1-tinystories-curve-parent-s307-lr0.0006: [compact-confirmation-v1-tinystories-curve-parent-s307.json](../records/compact-confirmation-v1-tinystories-curve-parent-s307.json), [summary.json](../dump/runs/20261005T023859Z-34d2d12f/summary.json)
- compact-confirmation-v1-tinystories-curve-parent-s401-lr0.0006: [compact-confirmation-v1-tinystories-curve-parent-s401.json](../records/compact-confirmation-v1-tinystories-curve-parent-s401.json), [summary.json](../dump/runs/20261005T025559Z-47d5befe/summary.json)
- compact-confirmation-v1-tinystories-curve-wide-s211-lr0.0006: [compact-confirmation-v1-tinystories-curve-wide-s211.json](../records/compact-confirmation-v1-tinystories-curve-wide-s211.json), [summary.json](../dump/runs/20261005T021847Z-ca6cce2c/summary.json)
- compact-confirmation-v1-tinystories-curve-wide-s307-lr0.0006: [compact-confirmation-v1-tinystories-curve-wide-s307.json](../records/compact-confirmation-v1-tinystories-curve-wide-s307.json), [summary.json](../dump/runs/20261005T023530Z-fbed9963/summary.json)
- compact-confirmation-v1-tinystories-curve-wide-s401-lr0.0006: [compact-confirmation-v1-tinystories-curve-wide-s401.json](../records/compact-confirmation-v1-tinystories-curve-wide-s401.json), [summary.json](../dump/runs/20261005T025230Z-6d5094eb/summary.json)
- compact-confirmation-v1-tinystories-full-gelu-s211-lr0.0006: [compact-confirmation-v1-tinystories-full-gelu-s211.json](../records/compact-confirmation-v1-tinystories-full-gelu-s211.json), [summary.json](../dump/runs/20261005T023336Z-6fa9d18c/summary.json)
- compact-confirmation-v1-tinystories-full-gelu-s307-lr0.0006: [compact-confirmation-v1-tinystories-full-gelu-s307.json](../records/compact-confirmation-v1-tinystories-full-gelu-s307.json), [summary.json](../dump/runs/20261005T025033Z-02932c90/summary.json)
- compact-confirmation-v1-tinystories-full-gelu-s401-lr0.0006: [compact-confirmation-v1-tinystories-full-gelu-s401.json](../records/compact-confirmation-v1-tinystories-full-gelu-s401.json), [summary.json](../dump/runs/20261005T030734Z-d11dc9eb/summary.json)
- compact-confirmation-v1-tinystories-full-swiglu-s211-lr0.0006: [compact-confirmation-v1-tinystories-full-swiglu-s211.json](../records/compact-confirmation-v1-tinystories-full-swiglu-s211.json), [summary.json](../dump/runs/20261005T023147Z-0910089e/summary.json)
- compact-confirmation-v1-tinystories-full-swiglu-s307-lr0.0006: [compact-confirmation-v1-tinystories-full-swiglu-s307.json](../records/compact-confirmation-v1-tinystories-full-swiglu-s307.json), [summary.json](../dump/runs/20261005T024844Z-9a813115/summary.json)
- compact-confirmation-v1-tinystories-full-swiglu-s401-lr0.0006: [compact-confirmation-v1-tinystories-full-swiglu-s401.json](../records/compact-confirmation-v1-tinystories-full-swiglu-s401.json), [summary.json](../dump/runs/20261005T030545Z-267fc6d4/summary.json)
- compact-confirmation-v1-tinystories-signed-fine-s211-lr0.0006: [compact-confirmation-v1-tinystories-signed-fine-s211.json](../records/compact-confirmation-v1-tinystories-signed-fine-s211.json), [summary.json](../dump/runs/20261005T022023Z-b877b2e4/summary.json)
- compact-confirmation-v1-tinystories-signed-fine-s307-lr0.0006: [compact-confirmation-v1-tinystories-signed-fine-s307.json](../records/compact-confirmation-v1-tinystories-signed-fine-s307.json), [summary.json](../dump/runs/20261005T023719Z-71a6d401/summary.json)
- compact-confirmation-v1-tinystories-signed-fine-s401-lr0.0006: [compact-confirmation-v1-tinystories-signed-fine-s401.json](../records/compact-confirmation-v1-tinystories-signed-fine-s401.json), [summary.json](../dump/runs/20261005T025415Z-b05571bb/summary.json)
- compact-confirmation-v1-tinystories-signed-parent-s211-lr0.0006: [compact-confirmation-v1-tinystories-signed-parent-s211.json](../records/compact-confirmation-v1-tinystories-signed-parent-s211.json), [summary.json](../dump/runs/20261005T022651Z-99c57ab8/summary.json)
- compact-confirmation-v1-tinystories-signed-parent-s307-lr0.0006: [compact-confirmation-v1-tinystories-signed-parent-s307.json](../records/compact-confirmation-v1-tinystories-signed-parent-s307.json), [summary.json](../dump/runs/20261005T024351Z-198a8f48/summary.json)
- compact-confirmation-v1-tinystories-signed-parent-s401-lr0.0006: [compact-confirmation-v1-tinystories-signed-parent-s401.json](../records/compact-confirmation-v1-tinystories-signed-parent-s401.json), [summary.json](../dump/runs/20261005T030054Z-aa865246/summary.json)
- compact-confirmation-v1-wikitext-compact-gelu-h518-s211-lr0.0006: [compact-confirmation-v1-wikitext-compact-gelu-h518-s211.json](../records/compact-confirmation-v1-wikitext-compact-gelu-h518-s211.json), [summary.json](../dump/runs/20261005T031616Z-5b0e241a/summary.json)
- compact-confirmation-v1-wikitext-compact-gelu-h518-s307-lr0.0006: [compact-confirmation-v1-wikitext-compact-gelu-h518-s307.json](../records/compact-confirmation-v1-wikitext-compact-gelu-h518-s307.json), [summary.json](../dump/runs/20261005T033259Z-0361d123/summary.json)
- compact-confirmation-v1-wikitext-compact-gelu-h518-s401-lr0.0006: [compact-confirmation-v1-wikitext-compact-gelu-h518-s401.json](../records/compact-confirmation-v1-wikitext-compact-gelu-h518-s401.json), [summary.json](../dump/runs/20261005T035021Z-0b6d915d/summary.json)
- compact-confirmation-v1-wikitext-compact-gelu-h576-s211-lr0.0006: [compact-confirmation-v1-wikitext-compact-gelu-h576-s211.json](../records/compact-confirmation-v1-wikitext-compact-gelu-h576-s211.json), [summary.json](../dump/runs/20261005T032054Z-47330b89/summary.json)
- compact-confirmation-v1-wikitext-compact-gelu-h576-s307-lr0.0006: [compact-confirmation-v1-wikitext-compact-gelu-h576-s307.json](../records/compact-confirmation-v1-wikitext-compact-gelu-h576-s307.json), [summary.json](../dump/runs/20261005T033742Z-cc6ae2c9/summary.json)
- compact-confirmation-v1-wikitext-compact-gelu-h576-s401-lr0.0006: [compact-confirmation-v1-wikitext-compact-gelu-h576-s401.json](../records/compact-confirmation-v1-wikitext-compact-gelu-h576-s401.json), [summary.json](../dump/runs/20261005T035527Z-daf9d955/summary.json)
- compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s211-lr0.0006: [compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s211.json](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s211.json), [summary.json](../dump/runs/20261005T031435Z-0ecf62bd/summary.json)
- compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s307-lr0.0006: [compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s307.json](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s307.json), [summary.json](../dump/runs/20261005T033122Z-a9743f65/summary.json)
- compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s401-lr0.0006: [compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s401.json](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h346-s401.json), [summary.json](../dump/runs/20261005T034840Z-ef46abe4/summary.json)
- compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s211-lr0.0006: [compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s211.json](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s211.json), [summary.json](../dump/runs/20261005T031927Z-b5b62613/summary.json)
- compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s307-lr0.0006: [compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s307.json](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s307.json), [summary.json](../dump/runs/20261005T033603Z-397cae87/summary.json)
- compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s401-lr0.0006: [compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s401.json](../records/compact-confirmation-v1-wikitext-compact-swiglu-kernel-h384-s401.json), [summary.json](../dump/runs/20261005T035345Z-7ee9948c/summary.json)
- compact-confirmation-v1-wikitext-curve-parent-s211-lr0.0006: [compact-confirmation-v1-wikitext-curve-parent-s211.json](../records/compact-confirmation-v1-wikitext-curve-parent-s211.json), [summary.json](../dump/runs/20261005T031256Z-4b749c5b/summary.json)
- compact-confirmation-v1-wikitext-curve-parent-s307-lr0.0006: [compact-confirmation-v1-wikitext-curve-parent-s307.json](../records/compact-confirmation-v1-wikitext-curve-parent-s307.json), [summary.json](../dump/runs/20261005T032943Z-792639c8/summary.json)
- compact-confirmation-v1-wikitext-curve-parent-s401-lr0.0006: [compact-confirmation-v1-wikitext-curve-parent-s401.json](../records/compact-confirmation-v1-wikitext-curve-parent-s401.json), [summary.json](../dump/runs/20261005T034655Z-7b38e9a4/summary.json)
- compact-confirmation-v1-wikitext-curve-wide-s211-lr0.0006: [compact-confirmation-v1-wikitext-curve-wide-s211.json](../records/compact-confirmation-v1-wikitext-curve-wide-s211.json), [summary.json](../dump/runs/20261005T030929Z-f28493ce/summary.json)
- compact-confirmation-v1-wikitext-curve-wide-s307-lr0.0006: [compact-confirmation-v1-wikitext-curve-wide-s307.json](../records/compact-confirmation-v1-wikitext-curve-wide-s307.json), [summary.json](../dump/runs/20261005T032613Z-41371e26/summary.json)
- compact-confirmation-v1-wikitext-curve-wide-s401-lr0.0006: [compact-confirmation-v1-wikitext-curve-wide-s401.json](../records/compact-confirmation-v1-wikitext-curve-wide-s401.json), [summary.json](../dump/runs/20261005T034319Z-765e0556/summary.json)
- compact-confirmation-v1-wikitext-full-gelu-s211-lr0.0006: [compact-confirmation-v1-wikitext-full-gelu-s211.json](../records/compact-confirmation-v1-wikitext-full-gelu-s211.json), [summary.json](../dump/runs/20261005T032422Z-a9df5bb3/summary.json)
- compact-confirmation-v1-wikitext-full-gelu-s307-lr0.0006: [compact-confirmation-v1-wikitext-full-gelu-s307.json](../records/compact-confirmation-v1-wikitext-full-gelu-s307.json), [summary.json](../dump/runs/20261005T034123Z-2888937d/summary.json)
- compact-confirmation-v1-wikitext-full-gelu-s401-lr0.0006: [compact-confirmation-v1-wikitext-full-gelu-s401.json](../records/compact-confirmation-v1-wikitext-full-gelu-s401.json), [summary.json](../dump/runs/20261005T035909Z-a32e0645/summary.json)
- compact-confirmation-v1-wikitext-full-swiglu-s211-lr0.0006: [compact-confirmation-v1-wikitext-full-swiglu-s211.json](../records/compact-confirmation-v1-wikitext-full-swiglu-s211.json), [summary.json](../dump/runs/20261005T032230Z-7e1ed4ec/summary.json)
- compact-confirmation-v1-wikitext-full-swiglu-s307-lr0.0006: [compact-confirmation-v1-wikitext-full-swiglu-s307.json](../records/compact-confirmation-v1-wikitext-full-swiglu-s307.json), [summary.json](../dump/runs/20261005T033920Z-74ce47fd/summary.json)
- compact-confirmation-v1-wikitext-full-swiglu-s401-lr0.0006: [compact-confirmation-v1-wikitext-full-swiglu-s401.json](../records/compact-confirmation-v1-wikitext-full-swiglu-s401.json), [summary.json](../dump/runs/20261005T035710Z-4670694c/summary.json)
- compact-confirmation-v1-wikitext-signed-fine-s211-lr0.0006: [compact-confirmation-v1-wikitext-signed-fine-s211.json](../records/compact-confirmation-v1-wikitext-signed-fine-s211.json), [summary.json](../dump/runs/20261005T031114Z-8e6ce93e/summary.json)
- compact-confirmation-v1-wikitext-signed-fine-s307-lr0.0006: [compact-confirmation-v1-wikitext-signed-fine-s307.json](../records/compact-confirmation-v1-wikitext-signed-fine-s307.json), [summary.json](../dump/runs/20261005T032803Z-96896491/summary.json)
- compact-confirmation-v1-wikitext-signed-fine-s401-lr0.0006: [compact-confirmation-v1-wikitext-signed-fine-s401.json](../records/compact-confirmation-v1-wikitext-signed-fine-s401.json), [summary.json](../dump/runs/20261005T034507Z-38d5db53/summary.json)
- compact-confirmation-v1-wikitext-signed-parent-s211-lr0.0006: [compact-confirmation-v1-wikitext-signed-parent-s211.json](../records/compact-confirmation-v1-wikitext-signed-parent-s211.json), [summary.json](../dump/runs/20261005T031753Z-af91cc4e/summary.json)
- compact-confirmation-v1-wikitext-signed-parent-s307-lr0.0006: [compact-confirmation-v1-wikitext-signed-parent-s307.json](../records/compact-confirmation-v1-wikitext-signed-parent-s307.json), [summary.json](../dump/runs/20261005T033436Z-b0baa545/summary.json)
- compact-confirmation-v1-wikitext-signed-parent-s401-lr0.0006: [compact-confirmation-v1-wikitext-signed-parent-s401.json](../records/compact-confirmation-v1-wikitext-signed-parent-s401.json), [summary.json](../dump/runs/20261005T035159Z-ff0409c5/summary.json)
- compact-final-v1-tinystories-compact-swiglu-kernel-h384-s509-lr0.0006: [compact-final-v1-tinystories-compact-swiglu-kernel-h384-final.json](../records/compact-final-v1-tinystories-compact-swiglu-kernel-h384-final.json), [compact-final-v1-tinystories-compact-swiglu-kernel-h384-valid.json](../records/compact-final-v1-tinystories-compact-swiglu-kernel-h384-valid.json), [summary.json](../dump/runs/20261005T042402Z-38f5a55b/summary.json)
- compact-final-v1-tinystories-curve-parent-s509-lr0.0006: [compact-final-v1-tinystories-curve-parent-final.json](../records/compact-final-v1-tinystories-curve-parent-final.json), [compact-final-v1-tinystories-curve-parent-valid.json](../records/compact-final-v1-tinystories-curve-parent-valid.json), [summary.json](../dump/runs/20261005T041654Z-23d988b6/summary.json)
- compact-final-v1-tinystories-curve-wide-s509-lr0.0006: [compact-final-v1-tinystories-curve-wide-final.json](../records/compact-final-v1-tinystories-curve-wide-final.json), [compact-final-v1-tinystories-curve-wide-valid.json](../records/compact-final-v1-tinystories-curve-wide-valid.json), [summary.json](../dump/runs/20261005T040945Z-c01e685c/summary.json)
- compact-final-v1-tinystories-full-swiglu-s509-lr0.0006: [compact-final-v1-tinystories-full-swiglu-final.json](../records/compact-final-v1-tinystories-full-swiglu-final.json), [compact-final-v1-tinystories-full-swiglu-valid.json](../records/compact-final-v1-tinystories-full-swiglu-valid.json), [summary.json](../dump/runs/20261005T043058Z-9099c9fe/summary.json)
- compact-final-v1-wikitext-compact-swiglu-kernel-h384-s509-lr0.0006: [compact-final-v1-wikitext-compact-swiglu-kernel-h384-final.json](../records/compact-final-v1-wikitext-compact-swiglu-kernel-h384-final.json), [compact-final-v1-wikitext-compact-swiglu-kernel-h384-valid.json](../records/compact-final-v1-wikitext-compact-swiglu-kernel-h384-valid.json), [summary.json](../dump/runs/20261005T150226Z-565d2437/summary.json)
- compact-final-v1-wikitext-curve-parent-s509-lr0.0006: [compact-final-v1-wikitext-curve-parent-final.json](../records/compact-final-v1-wikitext-curve-parent-final.json), [compact-final-v1-wikitext-curve-parent-valid.json](../records/compact-final-v1-wikitext-curve-parent-valid.json), [summary.json](../dump/runs/20261005T145534Z-f2909c60/summary.json)
- compact-final-v1-wikitext-curve-wide-s509-lr0.0006: [compact-final-v1-wikitext-curve-wide-final.json](../records/compact-final-v1-wikitext-curve-wide-final.json), [compact-final-v1-wikitext-curve-wide-valid.json](../records/compact-final-v1-wikitext-curve-wide-valid.json), [summary.json](../dump/runs/20261005T043839Z-8d5b5e15/summary.json)
- compact-final-v1-wikitext-full-swiglu-s509-lr0.0006: [compact-final-v1-wikitext-full-swiglu-final.json](../records/compact-final-v1-wikitext-full-swiglu-final.json), [compact-final-v1-wikitext-full-swiglu-valid.json](../records/compact-final-v1-wikitext-full-swiglu-valid.json), [summary.json](../dump/runs/20261005T150845Z-65e49e55/summary.json)
- compact-refinement-v1-tinystories-basis-small-groups-s101-lr0.0006: [compact-refinement-v1-tinystories-basis-small-groups.json](../records/compact-refinement-v1-tinystories-basis-small-groups.json), [summary.json](../dump/runs/20261005T020501Z-d2aac7c1/summary.json)
- compact-refinement-v1-tinystories-curve-parent-s101-lr0.0006: [compact-refinement-v1-tinystories-curve-parent.json](../records/compact-refinement-v1-tinystories-curve-parent.json), [summary.json](../dump/runs/20261005T015938Z-4ce8a6d2/summary.json)
- compact-refinement-v1-tinystories-curve-wide-s101-lr0.0006: [compact-refinement-v1-tinystories-curve-wide.json](../records/compact-refinement-v1-tinystories-curve-wide.json), [summary.json](../dump/runs/20261005T020118Z-129953b5/summary.json)
- compact-refinement-v1-tinystories-signed-fine-s101-lr0.0006: [compact-refinement-v1-tinystories-signed-fine.json](../records/compact-refinement-v1-tinystories-signed-fine.json), [summary.json](../dump/runs/20261005T015757Z-e80ec4ca/summary.json)
- compact-refinement-v1-tinystories-signed-parent-s101-lr0.0006: [compact-refinement-v1-tinystories-signed-parent.json](../records/compact-refinement-v1-tinystories-signed-parent.json), [signed-v1-tinystories-signed-linear.json](../records/signed-v1-tinystories-signed-linear.json), [summary.json](../dump/runs/20261004T082921Z-c73bd0ba/summary.json), [summary.json](../dump/runs/20261005T015626Z-d966c739/summary.json)
- compact-refinement-v1-wikitext-basis-small-groups-s101-lr0.0006: [compact-refinement-v1-wikitext-basis-small-groups.json](../records/compact-refinement-v1-wikitext-basis-small-groups.json), [summary.json](../dump/runs/20261005T021543Z-7962ee48/summary.json)
- compact-refinement-v1-wikitext-curve-parent-s101-lr0.0006: [compact-refinement-v1-wikitext-curve-parent.json](../records/compact-refinement-v1-wikitext-curve-parent.json), [summary.json](../dump/runs/20261005T021020Z-d68b1dd7/summary.json)
- compact-refinement-v1-wikitext-curve-wide-s101-lr0.0006: [compact-refinement-v1-wikitext-curve-wide.json](../records/compact-refinement-v1-wikitext-curve-wide.json), [summary.json](../dump/runs/20261005T021211Z-84ef799e/summary.json)
- compact-refinement-v1-wikitext-signed-fine-s101-lr0.0006: [compact-refinement-v1-wikitext-signed-fine.json](../records/compact-refinement-v1-wikitext-signed-fine.json), [summary.json](../dump/runs/20261005T020848Z-4998d9f9/summary.json)
- compact-refinement-v1-wikitext-signed-parent-s101-lr0.0006: [compact-refinement-v1-wikitext-signed-parent.json](../records/compact-refinement-v1-wikitext-signed-parent.json), [signed-v1-wikitext-signed-linear.json](../records/signed-v1-wikitext-signed-linear.json), [summary.json](../dump/runs/20261004T084717Z-4bbf32c6/summary.json), [summary.json](../dump/runs/20261005T020700Z-6037459e/summary.json)
- context-v1-tinystories-context-additive-s101-lr0.0006: [context-v1-tinystories-context-additive.json](../records/context-v1-tinystories-context-additive.json), [summary.json](../dump/runs/20261004T073953Z-77a18162/summary.json)
- context-v1-tinystories-context-product-s101-lr0.0006: [context-v1-tinystories-context-product.json](../records/context-v1-tinystories-context-product.json), [summary.json](../dump/runs/20261004T074140Z-2ad95003/summary.json)
- context-v1-tinystories-narrow-gelu-s101-lr0.0006: [context-v1-tinystories-narrow-gelu.json](../records/context-v1-tinystories-narrow-gelu.json), [summary.json](../dump/runs/20261004T073639Z-14d0f6a3/summary.json)
- context-v1-tinystories-narrow-swiglu-kernel-s101-lr0.0006: [context-v1-tinystories-narrow-swiglu-kernel.json](../records/context-v1-tinystories-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T073458Z-3965bf9a/summary.json)
- context-v1-tinystories-plain-silu-s101-lr0.0006: [context-v1-tinystories-plain-silu.json](../records/context-v1-tinystories-plain-silu.json), [summary.json](../dump/runs/20261004T073816Z-bb788c26/summary.json)
- context-v1-wikitext-context-additive-s101-lr0.0006: [context-v1-wikitext-context-additive.json](../records/context-v1-wikitext-context-additive.json), [summary.json](../dump/runs/20261004T075412Z-9b263ada/summary.json)
- context-v1-wikitext-context-product-s101-lr0.0006: [context-v1-wikitext-context-product.json](../records/context-v1-wikitext-context-product.json), [summary.json](../dump/runs/20261004T075603Z-842f7776/summary.json)
- context-v1-wikitext-narrow-gelu-s101-lr0.0006: [context-v1-wikitext-narrow-gelu.json](../records/context-v1-wikitext-narrow-gelu.json), [summary.json](../dump/runs/20261004T075049Z-df6a75ea/summary.json)
- context-v1-wikitext-narrow-swiglu-kernel-s101-lr0.0006: [context-v1-wikitext-narrow-swiglu-kernel.json](../records/context-v1-wikitext-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T074903Z-2bb2d5e0/summary.json)
- context-v1-wikitext-plain-silu-s101-lr0.0006: [context-v1-wikitext-plain-silu.json](../records/context-v1-wikitext-plain-silu.json), [summary.json](../dump/runs/20261004T075232Z-957f440d/summary.json)
- feature-flow-v1-tinystories-feature-cached-s101-lr0.0006: [feature-flow-v1-tinystories-feature-cached.json](../records/feature-flow-v1-tinystories-feature-cached.json), [summary.json](../dump/runs/20261004T131725Z-fc16d205/summary.json)
- feature-flow-v1-tinystories-feature-diagonal-s101-lr0.0006: [feature-flow-v1-tinystories-feature-diagonal.json](../records/feature-flow-v1-tinystories-feature-diagonal.json), [summary.json](../dump/runs/20261004T131343Z-5ad8b49b/summary.json)
- feature-flow-v1-tinystories-feature-flow-s101-lr0.0006: [feature-flow-v1-tinystories-feature-flow.json](../records/feature-flow-v1-tinystories-feature-flow.json), [summary.json](../dump/runs/20261004T131000Z-4b729091/summary.json)
- feature-flow-v1-tinystories-feature-once-s101-lr0.0006: [feature-flow-v1-tinystories-feature-once.json](../records/feature-flow-v1-tinystories-feature-once.json), [summary.json](../dump/runs/20261004T131151Z-a2ee2042/summary.json)
- feature-flow-v1-tinystories-feature-plain-s101-lr0.0006: [feature-flow-v1-tinystories-feature-plain.json](../records/feature-flow-v1-tinystories-feature-plain.json), [summary.json](../dump/runs/20261004T131537Z-3e653497/summary.json)
- feature-flow-v1-tinystories-narrow-gelu-s101-lr0.0006: [feature-flow-v1-tinystories-narrow-gelu.json](../records/feature-flow-v1-tinystories-narrow-gelu.json), [summary.json](../dump/runs/20261004T130640Z-ecce42a6/summary.json)
- feature-flow-v1-tinystories-narrow-swiglu-calibrated-s101-lr0.0006: [feature-flow-v1-tinystories-narrow-swiglu-calibrated.json](../records/feature-flow-v1-tinystories-narrow-swiglu-calibrated.json), [summary.json](../dump/runs/20261004T130818Z-27b3612f/summary.json)
- feature-flow-v1-tinystories-narrow-swiglu-kernel-s101-lr0.0006: [feature-flow-v1-tinystories-narrow-swiglu-kernel.json](../records/feature-flow-v1-tinystories-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T130457Z-6cfbd922/summary.json)
- feature-flow-v1-wikitext-feature-cached-s101-lr0.0006: [feature-flow-v1-wikitext-feature-cached.json](../records/feature-flow-v1-wikitext-feature-cached.json), [summary.json](../dump/runs/20261004T133838Z-1b58fc4e/summary.json)
- feature-flow-v1-wikitext-feature-diagonal-s101-lr0.0006: [feature-flow-v1-wikitext-feature-diagonal.json](../records/feature-flow-v1-wikitext-feature-diagonal.json), [summary.json](../dump/runs/20261004T133444Z-d62a64a5/summary.json)
- feature-flow-v1-wikitext-feature-flow-s101-lr0.0006: [feature-flow-v1-wikitext-feature-flow.json](../records/feature-flow-v1-wikitext-feature-flow.json), [summary.json](../dump/runs/20261004T133051Z-d683fbec/summary.json)
- feature-flow-v1-wikitext-feature-once-s101-lr0.0006: [feature-flow-v1-wikitext-feature-once.json](../records/feature-flow-v1-wikitext-feature-once.json), [summary.json](../dump/runs/20261004T133247Z-ca1abb8f/summary.json)
- feature-flow-v1-wikitext-feature-plain-s101-lr0.0006: [feature-flow-v1-wikitext-feature-plain.json](../records/feature-flow-v1-wikitext-feature-plain.json), [summary.json](../dump/runs/20261004T133642Z-76296cba/summary.json)
- feature-flow-v1-wikitext-narrow-gelu-s101-lr0.0006: [feature-flow-v1-wikitext-narrow-gelu.json](../records/feature-flow-v1-wikitext-narrow-gelu.json), [summary.json](../dump/runs/20261004T132725Z-ce53be9f/summary.json)
- feature-flow-v1-wikitext-narrow-swiglu-calibrated-s101-lr0.0006: [feature-flow-v1-wikitext-narrow-swiglu-calibrated.json](../records/feature-flow-v1-wikitext-narrow-swiglu-calibrated.json), [summary.json](../dump/runs/20261004T132905Z-d5d1bdae/summary.json)
- feature-flow-v1-wikitext-narrow-swiglu-kernel-s101-lr0.0006: [feature-flow-v1-wikitext-narrow-swiglu-kernel.json](../records/feature-flow-v1-wikitext-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T132541Z-f3aa83e2/summary.json)
- group-v4-tinystories-group-product-s101-lr0.0006: [group-v4-tinystories-group-product.json](../records/group-v4-tinystories-group-product.json), [summary.json](../dump/runs/20261004T065546Z-0000c08a/summary.json)
- group-v4-tinystories-narrow-gelu-s101-lr0.0006: [group-v4-tinystories-narrow-gelu.json](../records/group-v4-tinystories-narrow-gelu.json), [summary.json](../dump/runs/20261004T065050Z-e0316361/summary.json)
- group-v4-tinystories-narrow-swiglu-kernel-s101-lr0.0006: [group-v4-tinystories-narrow-swiglu-kernel.json](../records/group-v4-tinystories-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T064910Z-777ca0f1/summary.json)
- group-v4-tinystories-plain-silu-s101-lr0.0006: [group-v4-tinystories-plain-silu.json](../records/group-v4-tinystories-plain-silu.json), [summary.json](../dump/runs/20261004T065227Z-82261675/summary.json)
- group-v4-tinystories-square-s101-lr0.0006: [group-v4-tinystories-square.json](../records/group-v4-tinystories-square.json), [summary.json](../dump/runs/20261004T065402Z-88d8d06c/summary.json)
- group-v4-wikitext-group-product-s101-lr0.0006: [group-v4-wikitext-group-product.json](../records/group-v4-wikitext-group-product.json), [summary.json](../dump/runs/20261004T070958Z-70e23065/summary.json)
- group-v4-wikitext-narrow-gelu-s101-lr0.0006: [group-v4-wikitext-narrow-gelu.json](../records/group-v4-wikitext-narrow-gelu.json), [summary.json](../dump/runs/20261004T070447Z-e91d01e9/summary.json)
- group-v4-wikitext-narrow-swiglu-kernel-s101-lr0.0006: [group-v4-wikitext-narrow-swiglu-kernel.json](../records/group-v4-wikitext-narrow-swiglu-kernel.json), [summary.json](../dump/runs/20261004T070303Z-609e5664/summary.json)
- group-v4-wikitext-plain-silu-s101-lr0.0006: [group-v4-wikitext-plain-silu.json](../records/group-v4-wikitext-plain-silu.json), [summary.json](../dump/runs/20261004T070629Z-68cc154c/summary.json)
- group-v4-wikitext-square-s101-lr0.0006: [group-v4-wikitext-square.json](../records/group-v4-wikitext-square.json), [summary.json](../dump/runs/20261004T070806Z-0ee3d431/summary.json)
- tinystories-blockshuffle-s101-lr0.0006: [language-v1-tinystories-blockshuffle.json](../records/language-v1-tinystories-blockshuffle.json), [summary.json](../dump/runs/20261004T044146Z-14ab4d8c/summary.json)
- tinystories-curve-only-s101-lr0.0006: [language-v1-tinystories-curve-only.json](../records/language-v1-tinystories-curve-only.json), [summary.json](../dump/runs/20261004T044021Z-f6e08d7d/summary.json)
- tinystories-full-gelu-s101-lr0.0006: [language-v1-tinystories-full-gelu.json](../records/language-v1-tinystories-full-gelu.json), [summary.json](../dump/runs/20261004T043408Z-72b0995b/summary.json)
- tinystories-full-swiglu-fused-s101-lr0.0006: [language-v1-tinystories-full-swiglu-fused.json](../records/language-v1-tinystories-full-swiglu-fused.json), [summary.json](../dump/runs/20261004T043249Z-18699fb6/summary.json)
- tinystories-full-swiglu-s101-lr0.0006: [language-v1-tinystories-full-swiglu.json](../records/language-v1-tinystories-full-swiglu.json), [summary.json](../dump/runs/20261004T043131Z-24be31f1/summary.json)
- tinystories-maxout-s101-lr0.0006: [language-v1-tinystories-maxout.json](../records/language-v1-tinystories-maxout.json), [summary.json](../dump/runs/20261004T044335Z-5bf59fba/summary.json)
- tinystories-meanout-s101-lr0.0006: [language-v1-tinystories-meanout.json](../records/language-v1-tinystories-meanout.json), [summary.json](../dump/runs/20261004T044454Z-11b935c6/summary.json)
- tinystories-narrow-gelu-s101-lr0.0006: [language-v1-tinystories-narrow-gelu.json](../records/language-v1-tinystories-narrow-gelu.json), [summary.json](../dump/runs/20261004T043750Z-eda0f635/summary.json)
- tinystories-narrow-swiglu-fused-s101-lr0.0006: [language-v1-tinystories-narrow-swiglu-fused.json](../records/language-v1-tinystories-narrow-swiglu-fused.json), [summary.json](../dump/runs/20261004T043635Z-c16f77a0/summary.json)
- tinystories-narrow-swiglu-s101-lr0.0006: [language-v1-tinystories-narrow-swiglu.json](../records/language-v1-tinystories-narrow-swiglu.json), [summary.json](../dump/runs/20261004T043528Z-402e3d90/summary.json)
- tinystories-plain-silu-s101-lr0.0006: [language-v1-tinystories-plain-silu.json](../records/language-v1-tinystories-plain-silu.json), [summary.json](../dump/runs/20261004T043904Z-3f2cd0c1/summary.json)
- wikitext-blockshuffle-s101-lr0.0006: [language-v1-wikitext-blockshuffle.json](../records/language-v1-wikitext-blockshuffle.json), [summary.json](../dump/runs/20261004T045707Z-8b2056a4/summary.json)
- wikitext-curve-only-s101-lr0.0006: [language-v1-wikitext-curve-only.json](../records/language-v1-wikitext-curve-only.json), [summary.json](../dump/runs/20261004T045532Z-ccb7817d/summary.json)
- wikitext-full-gelu-s101-lr0.0006: [language-v1-wikitext-full-gelu.json](../records/language-v1-wikitext-full-gelu.json), [summary.json](../dump/runs/20261004T044842Z-0d7fe3cf/summary.json)
- wikitext-full-swiglu-fused-s101-lr0.0006: [language-v1-wikitext-full-swiglu-fused.json](../records/language-v1-wikitext-full-swiglu-fused.json), [summary.json](../dump/runs/20261004T044714Z-93a203c0/summary.json)
- wikitext-full-swiglu-s101-lr0.0006: [language-v1-wikitext-full-swiglu.json](../records/language-v1-wikitext-full-swiglu.json), [summary.json](../dump/runs/20261004T044552Z-2f9c65a2/summary.json)
- wikitext-maxout-s101-lr0.0006: [language-v1-wikitext-maxout.json](../records/language-v1-wikitext-maxout.json), [summary.json](../dump/runs/20261004T045926Z-51fd93ef/summary.json)
- wikitext-meanout-s101-lr0.0006: [language-v1-wikitext-meanout.json](../records/language-v1-wikitext-meanout.json), [summary.json](../dump/runs/20261004T050058Z-3c37cd72/summary.json)
- wikitext-narrow-gelu-s101-lr0.0006: [language-v1-wikitext-narrow-gelu.json](../records/language-v1-wikitext-narrow-gelu.json), [summary.json](../dump/runs/20261004T045229Z-6c828e1e/summary.json)
- wikitext-narrow-swiglu-fused-s101-lr0.0006: [language-v1-wikitext-narrow-swiglu-fused.json](../records/language-v1-wikitext-narrow-swiglu-fused.json), [summary.json](../dump/runs/20261004T045136Z-73833aeb/summary.json)
- wikitext-narrow-swiglu-s101-lr0.0006: [language-v1-wikitext-narrow-swiglu.json](../records/language-v1-wikitext-narrow-swiglu.json), [summary.json](../dump/runs/20261004T045006Z-587daa23/summary.json)
- wikitext-plain-silu-s101-lr0.0006: [language-v1-wikitext-plain-silu.json](../records/language-v1-wikitext-plain-silu.json), [summary.json](../dump/runs/20261004T045350Z-981e2552/summary.json)
- latent-v1-tinystories-thin-calibrated-s101-lr0.0006: [latent-v1-tinystories-thin-calibrated.json](../records/latent-v1-tinystories-thin-calibrated.json), [summary.json](../dump/runs/20261004T092001Z-ffe9e9a9/summary.json)
- latent-v1-tinystories-thin-recompute-s101-lr0.0006: [latent-v1-tinystories-thin-recompute.json](../records/latent-v1-tinystories-thin-recompute.json), [summary.json](../dump/runs/20261004T091808Z-fa72a8ef/summary.json)
- latent-v1-tinystories-wide-cached-s101-lr0.0006: [latent-v1-tinystories-wide-cached.json](../records/latent-v1-tinystories-wide-cached.json), [summary.json](../dump/runs/20261004T091617Z-1c3ebd3b/summary.json)
- latent-v1-tinystories-wide-recompute-s101-lr0.0006: [latent-v1-tinystories-wide-recompute.json](../records/latent-v1-tinystories-wide-recompute.json), [summary.json](../dump/runs/20261004T091427Z-98b49166/summary.json)
- latent-v1-wikitext-thin-calibrated-s101-lr0.0006: [latent-v1-wikitext-thin-calibrated.json](../records/latent-v1-wikitext-thin-calibrated.json), [summary.json](../dump/runs/20261004T093732Z-7e529a5b/summary.json)
- latent-v1-wikitext-thin-recompute-s101-lr0.0006: [latent-v1-wikitext-thin-recompute.json](../records/latent-v1-wikitext-thin-recompute.json), [summary.json](../dump/runs/20261004T093537Z-a366dd6c/summary.json)
- latent-v1-wikitext-wide-cached-s101-lr0.0006: [latent-v1-wikitext-wide-cached.json](../records/latent-v1-wikitext-wide-cached.json), [summary.json](../dump/runs/20261004T093343Z-039fdbc5/summary.json)
- latent-v1-wikitext-wide-recompute-s101-lr0.0006: [latent-v1-wikitext-wide-recompute.json](../records/latent-v1-wikitext-wide-recompute.json), [summary.json](../dump/runs/20261004T093147Z-cd15663c/summary.json)
- shared-gate-v2-tinystories-narrow-gelu-s101-lr0.0006: [shared-gate-v2-tinystories-narrow-gelu.json](../records/shared-gate-v2-tinystories-narrow-gelu.json), [summary.json](../dump/runs/20261004T052559Z-f67b2162/summary.json)
- shared-gate-v2-tinystories-narrow-swiglu-fused-s101-lr0.0006: [shared-gate-v2-tinystories-narrow-swiglu-fused.json](../records/shared-gate-v2-tinystories-narrow-swiglu-fused.json), [summary.json](../dump/runs/20261004T052420Z-5216d321/summary.json)
- shared-gate-v2-tinystories-shared-basis-s101-lr0.0006: [shared-gate-v2-tinystories-shared-basis.json](../records/shared-gate-v2-tinystories-shared-basis.json), [summary.json](../dump/runs/20261004T052736Z-1c407023/summary.json)
- shared-gate-v2-tinystories-untied-basis-s101-lr0.0006: [shared-gate-v2-tinystories-untied-basis.json](../records/shared-gate-v2-tinystories-untied-basis.json), [summary.json](../dump/runs/20261004T052913Z-59dc2020/summary.json)
- shared-gate-v2-wikitext-narrow-gelu-s101-lr0.0006: [shared-gate-v2-wikitext-narrow-gelu.json](../records/shared-gate-v2-wikitext-narrow-gelu.json), [summary.json](../dump/runs/20261004T053630Z-c91d8e53/summary.json)
- shared-gate-v2-wikitext-narrow-swiglu-fused-s101-lr0.0006: [shared-gate-v2-wikitext-narrow-swiglu-fused.json](../records/shared-gate-v2-wikitext-narrow-swiglu-fused.json), [summary.json](../dump/runs/20261004T053445Z-4f6b4db6/summary.json)
- shared-gate-v2-wikitext-shared-basis-s101-lr0.0006: [shared-gate-v2-wikitext-shared-basis.json](../records/shared-gate-v2-wikitext-shared-basis.json), [summary.json](../dump/runs/20261004T053810Z-2f35048a/summary.json)
- shared-gate-v2-wikitext-untied-basis-s101-lr0.0006: [shared-gate-v2-wikitext-untied-basis.json](../records/shared-gate-v2-wikitext-untied-basis.json), [summary.json](../dump/runs/20261004T053951Z-64008190/summary.json)
- signed-v1-tinystories-closed-gain-s101-lr0.0006: [signed-v1-tinystories-closed-gain.json](../records/signed-v1-tinystories-closed-gain.json), [summary.json](../dump/runs/20261004T082548Z-61e472a1/summary.json)
- signed-v1-tinystories-open-gain-s101-lr0.0006: [signed-v1-tinystories-open-gain.json](../records/signed-v1-tinystories-open-gain.json), [summary.json](../dump/runs/20261004T082735Z-98c638ea/summary.json)
- signed-v1-tinystories-plain-silu-s101-lr0.0006: [signed-v1-tinystories-plain-silu.json](../records/signed-v1-tinystories-plain-silu.json), [summary.json](../dump/runs/20261004T082412Z-e5f9c09e/summary.json)
- signed-v1-tinystories-signed-tanh-s101-lr0.0006: [signed-v1-tinystories-signed-tanh.json](../records/signed-v1-tinystories-signed-tanh.json), [summary.json](../dump/runs/20261004T083107Z-084e62fb/summary.json)
- signed-v1-wikitext-closed-gain-s101-lr0.0006: [signed-v1-wikitext-closed-gain.json](../records/signed-v1-wikitext-closed-gain.json), [summary.json](../dump/runs/20261004T084339Z-be00224f/summary.json)
- signed-v1-wikitext-open-gain-s101-lr0.0006: [signed-v1-wikitext-open-gain.json](../records/signed-v1-wikitext-open-gain.json), [summary.json](../dump/runs/20261004T084528Z-88a29d61/summary.json)
- signed-v1-wikitext-plain-silu-s101-lr0.0006: [signed-v1-wikitext-plain-silu.json](../records/signed-v1-wikitext-plain-silu.json), [summary.json](../dump/runs/20261004T084200Z-f7841449/summary.json)
- signed-v1-wikitext-signed-tanh-s101-lr0.0006: [signed-v1-wikitext-signed-tanh.json](../records/signed-v1-wikitext-signed-tanh.json), [summary.json](../dump/runs/20261004T084906Z-1e367fb8/summary.json)

</details>


## Step 3 rolling language models

Last-checkpoint full prepared-validation NLL; lower is better. Each table fixes
the dataset, parameter configuration and training budget. Loop rows reuse the
same jointly trained checkpoint: one loop is an inference reference, not an
independently trained baseline. Reconstruction scores are excluded.

### HuggingFaceTB/smol-smoltalk — cohort `ff04ce44de5a`

| Rank | Run / loops | NLL | Core / encoder params | FFN params | Targets | Peak allocated MiB | Seed / updates | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | rolling-memory-lm-v1 / 2 | 4.830327 | 39,052,032 / 4,798,720 | 7,133,184 | 115,093 | 4154.8 | 17 / 1200 | [local](../records/rolling-memory-lm-v1-1200.json) |
| 2 | rolling-memory-lm-v1 / 4 | 4.836622 | 39,052,032 / 4,798,720 | 7,133,184 | 115,093 | 4154.8 | 17 / 1200 | [local](../records/rolling-memory-lm-v1-1200.json) |
| 3 | rolling-memory-lm-v1 / 1 | 4.845762 | 39,052,032 / 4,798,720 | 7,133,184 | 115,093 | 4154.8 | 17 / 1200 | [local](../records/rolling-memory-lm-v1-1200.json) |

### HuggingFaceFW/fineweb-edu — cohort `ff04ce44de5a`

| Rank | Run / loops | NLL | Core / encoder params | FFN params | Targets | Peak allocated MiB | Seed / updates | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | rolling-memory-lm-v1 / 2 | 5.871381 | 39,052,032 / 4,798,720 | 7,133,184 | 138,520 | 4154.8 | 17 / 1200 | [local](../records/rolling-memory-lm-v1-1200.json) |
| 2 | rolling-memory-lm-v1 / 4 | 5.873713 | 39,052,032 / 4,798,720 | 7,133,184 | 138,520 | 4154.8 | 17 / 1200 | [local](../records/rolling-memory-lm-v1-1200.json) |
| 3 | rolling-memory-lm-v1 / 1 | 5.879878 | 39,052,032 / 4,798,720 | 7,133,184 | 138,520 | 4154.8 | 17 / 1200 | [local](../records/rolling-memory-lm-v1-1200.json) |

Zero/shuffled-memory interventions are reported separately in [Step 3 results](step_3_results.md).


## Step 3 dense versus rolling: matched training targets

Both language cores start from scratch. The dense baseline has 256 raw tokens;
the rolling core additionally sees up to 64 compressed history vectors and a
separately pretrained encoder. This is approximately core-size-matched, not
equal-information, equal-system-size or equal-compute. All rows use the entire
prepared validation split at that checkpoint. Loop rows reuse rolling weights.

[Protocol and interpretation](step_3_comparison.md).

### HuggingFaceTB/smol-smoltalk / 1,200 updates / `3f7f0c7ce8bd`

Identical 4,416,813 scored training targets; schedule `1ef0141e7f73`; seed 17.

| Rank | Model / loops | NLL | Core / encoder params | FFN params | Peak MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense / 1 | 4.466478 | 38,545,152 / 0 | 21,233,664 | 1253.9 | 134.3 | [local](../records/rolling-vs-dense-3600-v1-dense-1200.json) |
| 2 | rolling / 2 | 5.575582 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 478.9 | [local](../records/rolling-vs-dense-3600-v1-rolling-1200.json) |
| 3 | rolling / 1 | 5.583298 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 478.9 | [local](../records/rolling-vs-dense-3600-v1-rolling-1200.json) |
| 4 | rolling / 4 | 5.588969 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 478.9 | [local](../records/rolling-vs-dense-3600-v1-rolling-1200.json) |

### HuggingFaceTB/smol-smoltalk / 2,400 updates / `c36db97266b8`

Identical 8,834,515 scored training targets; schedule `09e8afdb8226`; seed 17.

| Rank | Model / loops | NLL | Core / encoder params | FFN params | Peak MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense / 1 | 4.171615 | 38,545,152 / 0 | 21,233,664 | 1253.9 | 279.1 | [local](../records/rolling-vs-dense-3600-v1-dense-2400.json) |
| 2 | rolling / 2 | 5.133216 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 940.6 | [local](../records/rolling-vs-dense-3600-v1-rolling-2400.json) |
| 3 | rolling / 4 | 5.144300 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 940.6 | [local](../records/rolling-vs-dense-3600-v1-rolling-2400.json) |
| 4 | rolling / 1 | 5.152397 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 940.6 | [local](../records/rolling-vs-dense-3600-v1-rolling-2400.json) |

### HuggingFaceTB/smol-smoltalk / 3,600 updates / `08925f1f7fbb`

Identical 12,574,061 scored training targets; schedule `945bb231bf9d`; seed 17.

| Rank | Model / loops | NLL | Core / encoder params | FFN params | Peak MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense / 1 | 2.693145 | 38,545,152 / 0 | 21,233,664 | 1253.9 | 423.9 | [local](../records/rolling-vs-dense-3600-v1-dense-3600.json) |
| 2 | rolling / 2 | 3.599201 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 1468.5 | [local](../records/rolling-vs-dense-3600-v1-rolling-3600.json) |
| 3 | rolling / 4 | 3.616680 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 1468.5 | [local](../records/rolling-vs-dense-3600-v1-rolling-3600.json) |
| 4 | rolling / 1 | 3.626758 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 1468.5 | [local](../records/rolling-vs-dense-3600-v1-rolling-3600.json) |

### HuggingFaceFW/fineweb-edu / 1,200 updates / `3f7f0c7ce8bd`

Identical 4,416,813 scored training targets; schedule `1ef0141e7f73`; seed 17.

| Rank | Model / loops | NLL | Core / encoder params | FFN params | Peak MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense / 1 | 4.414330 | 38,545,152 / 0 | 21,233,664 | 1253.9 | 134.3 | [local](../records/rolling-vs-dense-3600-v1-dense-1200.json) |
| 2 | rolling / 2 | 5.473161 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 478.9 | [local](../records/rolling-vs-dense-3600-v1-rolling-1200.json) |
| 3 | rolling / 1 | 5.480787 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 478.9 | [local](../records/rolling-vs-dense-3600-v1-rolling-1200.json) |
| 4 | rolling / 4 | 5.484020 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 478.9 | [local](../records/rolling-vs-dense-3600-v1-rolling-1200.json) |

### HuggingFaceFW/fineweb-edu / 2,400 updates / `c36db97266b8`

Identical 8,834,515 scored training targets; schedule `09e8afdb8226`; seed 17.

| Rank | Model / loops | NLL | Core / encoder params | FFN params | Peak MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense / 1 | 4.120650 | 38,545,152 / 0 | 21,233,664 | 1253.9 | 279.1 | [local](../records/rolling-vs-dense-3600-v1-dense-2400.json) |
| 2 | rolling / 2 | 5.097618 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 940.6 | [local](../records/rolling-vs-dense-3600-v1-rolling-2400.json) |
| 3 | rolling / 1 | 5.110422 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 940.6 | [local](../records/rolling-vs-dense-3600-v1-rolling-2400.json) |
| 4 | rolling / 4 | 5.113655 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 940.6 | [local](../records/rolling-vs-dense-3600-v1-rolling-2400.json) |

### HuggingFaceFW/fineweb-edu / 3,600 updates / `08925f1f7fbb`

Identical 12,574,061 scored training targets; schedule `945bb231bf9d`; seed 17.

| Rank | Model / loops | NLL | Core / encoder params | FFN params | Peak MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense / 1 | 4.211167 | 38,545,152 / 0 | 21,233,664 | 1253.9 | 423.9 | [local](../records/rolling-vs-dense-3600-v1-dense-3600.json) |
| 2 | rolling / 2 | 5.003624 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 1468.5 | [local](../records/rolling-vs-dense-3600-v1-rolling-3600.json) |
| 3 | rolling / 4 | 5.018080 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 1468.5 | [local](../records/rolling-vs-dense-3600-v1-rolling-3600.json) |
| 4 | rolling / 1 | 5.019767 | 39,052,032 / 4,798,720 | 7,133,184 | 4154.8 | 1468.5 | [local](../records/rolling-vs-dense-3600-v1-rolling-3600.json) |


## Step 3 single-pass memory reader: paired continuation

Both arms start from the same 3,600-update dense checkpoint. The reader adds parameters
and a separately pretrained frozen encoder. These are continuation results, not fresh runs.
[Protocol and diagnostics](step_3_memory_reader.md).

### HuggingFaceTB/smol-smoltalk / continuation / `682df8f7f261`

Seed 17; 600 additional updates; 2,009,135 identical scored targets; schedule `aaeac0c8171f`.

| Rank | Model | Full validation NLL | Core / encoder params | FFN params | Peak MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | reader | 2.665881 | 38,824,705 / 4,798,720 | 21,233,664 | 1279.7 | 85.6 | [local](../records/memory-reader-600-v1-reader.json) |
| 2 | dense | 2.665899 | 38,545,152 / 0 | 21,233,664 | 1253.9 | 76.8 | [local](../records/memory-reader-600-v1-dense.json) |

### HuggingFaceFW/fineweb-edu / continuation / `682df8f7f261`

Seed 17; 600 additional updates; 2,009,135 identical scored targets; schedule `aaeac0c8171f`.

| Rank | Model | Full validation NLL | Core / encoder params | FFN params | Peak MiB | Training seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | reader | 4.128996 | 38,824,705 / 4,798,720 | 21,233,664 | 1279.7 | 85.6 | [local](../records/memory-reader-600-v1-reader.json) |
| 2 | dense | 4.129006 | 38,545,152 / 0 | 21,233,664 | 1253.9 | 76.8 | [local](../records/memory-reader-600-v1-dense.json) |


## Step 3 Curve-Wide: raw versus encoded context

Same source text and prediction targets: 256 older + 256 recent tokens.
The encoded arm replaces the older chunk with eight frozen encoder vectors.
Fresh language cores, approximately equal core parameters; dense has six blocks and Curve-Wide ten.
Encoder pretraining/parameters are additional. [Protocol](step_3_context_comparison.md).

### HuggingFaceTB/smol-smoltalk / 600 updates / `869229a0b3ce`

Seed 17; 2,005,324 identical scored targets; schedule `2e14b0261112`.

| Rank | Model | Full validation NLL | Token accuracy | History-window NLL | Core / encoder params | FFN params | Peak allocated MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense_raw | 3.755294 | 33.17% | 3.746817 | 38,545,152 / 0 | 21,233,664 | 1201.4 | 118.6 | [local](../records/curve-context-600-v1-dense_raw.json) |
| 2 | curve_raw | 3.858403 | 32.26% | 3.846981 | 38,643,456 / 0 | 11,888,640 | 1924.3 | 295.6 | [local](../records/curve-context-600-v1-curve_raw.json) |
| 3 | curve_encoded | 3.895410 | 31.72% | 3.889731 | 38,840,320 / 4,798,720 | 11,888,640 | 1362.1 | 259.3 | [local](../records/curve-context-600-v1-curve_encoded.json) |

### HuggingFaceFW/fineweb-edu / 600 updates / `869229a0b3ce`

Seed 17; 2,005,324 identical scored targets; schedule `2e14b0261112`.

| Rank | Model | Full validation NLL | Token accuracy | History-window NLL | Core / encoder params | FFN params | Peak allocated MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense_raw | 4.982010 | 17.28% | 4.966479 | 38,545,152 / 0 | 21,233,664 | 1201.4 | 118.6 | [local](../records/curve-context-600-v1-dense_raw.json) |
| 2 | curve_raw | 5.062317 | 16.97% | 5.046933 | 38,643,456 / 0 | 11,888,640 | 1924.3 | 295.6 | [local](../records/curve-context-600-v1-curve_raw.json) |
| 3 | curve_encoded | 5.082637 | 16.68% | 5.079356 | 38,840,320 / 4,798,720 | 11,888,640 | 1362.1 | 259.3 | [local](../records/curve-context-600-v1-curve_encoded.json) |


## Reopened Step 1: equal-parameter FFNs

Four identical Transformer layers, width 384, 256 raw tokens; no encoder, loops or extra depth.
Approximately 8.65M parameters each. [Protocol and interpretation](step_1_equal_parameters.md).

### HuggingFaceTB/smol-smoltalk / 600 updates / `8f9afdaa0947`

Seed 17; 2,005,324 identical scored targets; schedule `2e14b0261112`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense | 4.227783 | 29.03% | 8,654,208 / 4,718,592 | 435.6 | 41.2 | [local](../records/ffn-equal-600-v1-dense.json) |
| 2 | expanded_curve | 4.230823 | 28.94% | 8,653,056 / 4,717,440 | 876.7 | 56.1 | [local](../records/ffn-equal-600-v1-expanded_curve.json) |
| 3 | learned_prelu | 4.455010 | 26.77% | 8,654,200 / 4,718,584 | 425.1 | 40.8 | [local](../records/ffn-equal-600-v1-learned_prelu.json) |
| 4 | fixed_prelu | 4.455579 | 26.80% | 8,654,208 / 4,718,592 | 423.9 | 39.2 | [local](../records/ffn-equal-600-v1-fixed_prelu.json) |
| 5 | learned_silu_tanh | 4.481004 | 26.43% | 8,654,200 / 4,718,584 | 451.8 | 37.5 | [local](../records/ffn-equal-600-v1-learned_silu_tanh.json) |
| 6 | wide_silu | 4.481322 | 26.41% | 8,654,208 / 4,718,592 | 423.9 | 41.4 | [local](../records/ffn-equal-600-v1-wide_silu.json) |

### HuggingFaceFW/fineweb-edu / 600 updates / `8f9afdaa0947`

Seed 17; 2,005,324 identical scored targets; schedule `2e14b0261112`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | expanded_curve | 5.296054 | 15.51% | 8,653,056 / 4,717,440 | 876.7 | 56.1 | [local](../records/ffn-equal-600-v1-expanded_curve.json) |
| 2 | dense | 5.298029 | 15.52% | 8,654,208 / 4,718,592 | 435.6 | 41.2 | [local](../records/ffn-equal-600-v1-dense.json) |
| 3 | fixed_prelu | 5.463947 | 14.44% | 8,654,208 / 4,718,592 | 423.9 | 39.2 | [local](../records/ffn-equal-600-v1-fixed_prelu.json) |
| 4 | learned_prelu | 5.464313 | 14.41% | 8,654,200 / 4,718,584 | 425.1 | 40.8 | [local](../records/ffn-equal-600-v1-learned_prelu.json) |
| 5 | learned_silu_tanh | 5.480683 | 14.18% | 8,654,200 / 4,718,584 | 451.8 | 37.5 | [local](../records/ffn-equal-600-v1-learned_silu_tanh.json) |
| 6 | wide_silu | 5.480942 | 14.21% | 8,654,208 / 4,718,592 | 423.9 | 41.4 | [local](../records/ffn-equal-600-v1-wide_silu.json) |


## Step 1: FFN architecture comparisons, matched parameters

Backbone, parameter budget, data and training schedule are matched within each cohort; no encoder or loops.
[Single-branch study](step_1_single_curve.md) and [feature-mixing research](step_1_feature_mixing.md). Sizes/budgets remain separate.

### HuggingFaceTB/smol-smoltalk / 1800 updates / `05df563da10f`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | filter_tanh | 3.354039 | 38.92% | 8,654,208 / 4,718,592 | 452.4 | 474.0 | 156.4 | [local](../records/ffn-nonlinear-filter-1800-s461-filter_tanh.json) |
| 2 | filter_signed_square | 3.370427 | 38.76% | 8,654,208 / 4,718,592 | 456.9 | 474.0 | 159.4 | [local](../records/ffn-nonlinear-filter-1800-s461-filter_signed_square.json) |

### HuggingFaceTB/smol-smoltalk / 5400 updates / `089e67c87f18`

4 layers; width 384; 256 raw tokens.
Seed 461; 18,067,829 identical scored targets; schedule `1b98493063b4`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 2.712704 | 47.46% | 8,654,208 / 4,718,592 | 446.9 | 468.0 | 427.3 | [local](../records/ffn-filter-continuation-5400-s461-dynamic_filter.json) |
| 2 | dense | 2.801623 | 45.74% | 8,654,208 / 4,718,592 | 436.4 | 466.0 | 329.2 | [local](../records/ffn-filter-continuation-5400-s461-dense.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `0a71939a74d8`

4 layers; width 384; 256 raw tokens.
Seed 239; 6,034,737 identical scored targets; schedule `e11a74cf1f9b`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | relu_squared | 3.361140 | 38.43% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 139.7 | [local](../records/ffn-associative-1800-s239-relu_squared.json) |
| 2 | shifted_relu_squared | 3.361789 | 38.49% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 113.1 | [local](../records/ffn-associative-1800-s239-shifted_relu_squared.json) |
| 3 | dense | 3.393559 | 37.99% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 141.4 | [local](../records/ffn-associative-1800-s239-dense.json) |
| 4 | associative_normalized | 3.414679 | 37.95% | 8,654,208 / 4,718,592 | 476.8 | 498.0 | 172.8 | [local](../records/ffn-associative-1800-s239-associative_normalized.json) |
| 5 | associative_raw | 3.415693 | 37.88% | 8,654,208 / 4,718,592 | 444.6 | 472.0 | 148.3 | [local](../records/ffn-associative-1800-s239-associative_raw.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `0cd79764e28f`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | axis_parallel | 3.348746 | 38.83% | 8,654,208 / 4,718,592 | 460.0 | 492.0 | 165.8 | [local](../records/ffn-two-axis-confirm-1800-s293-axis_parallel.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `0f6fb5414f17`

4 layers; width 384; 256 raw tokens.
Seed 101; 6,027,819 identical scored targets; schedule `c8535b205682`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | relu_squared | 3.356948 | 38.46% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 121.7 | [local](../records/ffn-feature-1800-s101-relu_squared.json) |
| 2 | group_refine | 3.370661 | 38.29% | 8,646,016 / 4,710,400 | 445.4 | 470.0 | 100.7 | [local](../records/ffn-feature-1800-s101-group_refine.json) |
| 3 | dense | 3.371073 | 38.34% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 116.3 | [local](../records/ffn-feature-1800-s101-dense.json) |
| 4 | group_product | 3.371853 | 38.30% | 8,646,016 / 4,710,400 | 467.1 | 490.0 | 115.3 | [local](../records/ffn-feature-1800-s101-group_product.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `14a7ea39034f`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.352524 | 39.03% | 8,654,208 / 4,718,592 | 459.4 | 482.0 | 117.8 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | ridge_01 | 3.362974 | 39.02% | 8,654,208 / 4,718,592 | 459.4 | 482.0 | 137.9 | [local](../records/ffn-ridge-1800-s461-ridge_01.json) |

### HuggingFaceTB/smol-smoltalk / 10000 updates / `1631256244b4`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | router_mlp | 2.410576 | 51.64% | 8,654,208 / 4,718,592 | 454.4 | 472.0 | 1042.8 | [local](../records/ffn-branch-refinement-10000-s461-router_mlp.json) |
| 2 | branch2 | 2.411672 | 51.59% | 8,654,208 / 4,718,592 | 447.0 | 468.0 | 964.4 | [local](../records/ffn-branch-refinement-10000-s461-branch2.json) |
| 3 | ungated | 2.415453 | 51.50% | 8,654,208 / 4,718,592 | 425.6 | 448.0 | 865.3 | [local](../records/ffn-branch-refinement-10000-s461-ungated.json) |
| 4 | matrix4x4 | 2.436932 | 51.20% | 8,654,208 / 4,718,592 | 447.0 | 472.0 | 1378.2 | [local](../records/ffn-branch-refinement-10000-s461-matrix4x4.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `16a449d0779f`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | reflective | 3.358345 | 38.84% | 8,654,208 / 4,718,592 | 442.0 | 460.0 | 203.5 | [local](../records/ffn-reflective-1800-s461-reflective.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `1dcde507f8a5`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | convolution_raw | 3.354088 | 38.77% | 8,654,208 / 4,718,592 | 448.7 | 468.0 | 167.2 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 2 | convolution_squared | 3.370874 | 38.74% | 8,654,208 / 4,718,592 | 452.7 | 470.0 | 170.4 | [local](../records/ffn-convolution-1800-s461-convolution_squared.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `22d10226d422`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.324885 | 39.12% | 8,654,208 / 4,718,592 | 447.4 | 468.0 | 152.0 | [local](../records/ffn-dynamic-filter-confirm-1800-s311-dynamic_filter.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `266a5ebc4d04`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | wide_conv | 3.343073 | 39.10% | 8,654,208 / 4,718,592 | 449.4 | 472.0 | 177.6 | [local](../records/ffn-wide-convolution-1800-s461-wide_conv.json) |

### HuggingFaceTB/smol-smoltalk / 10000 updates / `26d13f07961c`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | attention_energy | 2.411464 | 51.72% | 8,654,208 / 4,718,592 | 460.5 | 488.0 | 1020.9 | [local](../records/ffn-evidence-10000-s461-attention_energy.json) |
| 2 | evidence_balance | 2.411573 | 51.69% | 8,654,208 / 4,718,592 | 448.1 | 468.0 | 1928.5 | [local](../records/ffn-evidence-10000-s461-evidence_balance.json) |
| 3 | reaction_catalyst | 2.412067 | 51.65% | 8,654,208 / 4,718,592 | 470.1 | 490.0 | 1385.4 | [local](../records/ffn-evidence-10000-s461-reaction_catalyst.json) |
| 4 | evidence_support | 2.412164 | 51.71% | 8,654,208 / 4,718,592 | 448.1 | 468.0 | 1123.4 | [local](../records/ffn-evidence-10000-s461-evidence_support.json) |
| 5 | attention_alignment | 2.412334 | 51.74% | 8,654,208 / 4,718,592 | 463.1 | 488.0 | 1015.4 | [local](../records/ffn-evidence-10000-s461-attention_alignment.json) |
| 6 | reaction_post | 2.413226 | 51.66% | 8,654,208 / 4,718,592 | 446.9 | 468.0 | 1016.8 | [local](../records/ffn-evidence-10000-s461-reaction_post.json) |
| 7 | cross_evidence | 2.413500 | 51.49% | 8,654,208 / 4,718,592 | 454.3 | 472.0 | 1924.5 | [local](../records/ffn-evidence-10000-s461-cross_evidence.json) |
| 8 | learned_evidence | 2.433106 | 51.32% | 8,654,208 / 4,718,592 | 450.9 | 468.0 | 1184.8 | [local](../records/ffn-evidence-10000-s461-learned_evidence.json) |

### HuggingFaceTB/smol-smoltalk / 10000 updates / `2898c8c1d063`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | branch_rich_router | 2.411784 | 51.60% | 8,654,208 / 4,718,592 | 450.4 | 474.0 | 1118.5 | [local](../records/ffn-branch-next-10000-s461-branch_rich_router.json) |
| 2 | branch_hierarchical | 2.412127 | 51.69% | 8,654,208 / 4,718,592 | 444.6 | 482.0 | 1510.7 | [local](../records/ffn-branch-next-10000-s461-branch_hierarchical.json) |
| 3 | branch_threshold | 2.425312 | 51.48% | 8,654,208 / 4,718,592 | 451.1 | 472.0 | 1204.0 | [local](../records/ffn-branch-next-10000-s461-branch_threshold.json) |
| 4 | branch_temperature | 2.431398 | 51.37% | 8,654,208 / 4,718,592 | 450.0 | 472.0 | 1016.0 | [local](../records/ffn-branch-next-10000-s461-branch_temperature.json) |
| 5 | branch_bipolar | 2.497739 | 50.24% | 8,654,208 / 4,718,592 | 471.5 | 508.0 | 1034.5 | [local](../records/ffn-branch-next-10000-s461-branch_bipolar.json) |

### HuggingFaceTB/smol-smoltalk / 10000 updates / `325904e3a840`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 2.411506 | 51.69% | 8,654,208 / 4,718,592 | 454.5 | 480.0 | 848.8 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | bottleneck_squared | 2.411984 | 51.51% | 8,648,064 / 4,712,448 | 444.1 | 474.0 | 806.9 | [local](../records/ffn-final-10000-s461-bottleneck_squared.json) |
| 3 | competitive_256 | 2.412565 | 51.51% | 8,654,208 / 4,718,592 | 454.8 | 482.0 | 916.7 | [local](../records/ffn-final-10000-s461-competitive_256.json) |
| 4 | bottleneck_silu | 2.413933 | 51.57% | 8,648,064 / 4,712,448 | 444.1 | 474.0 | 821.6 | [local](../records/ffn-final-10000-s461-bottleneck_silu.json) |
| 5 | correlation_diagonal | 2.417012 | 51.45% | 8,654,208 / 4,718,592 | 451.5 | 488.0 | 845.8 | [local](../records/ffn-final-10000-s461-correlation_diagonal.json) |
| 6 | competitive_diagonal | 2.418648 | 51.41% | 8,654,208 / 4,718,592 | 454.8 | 482.0 | 911.8 | [local](../records/ffn-final-10000-s461-competitive_diagonal.json) |
| 7 | bottleneck_linear | 2.420762 | 51.39% | 8,648,064 / 4,712,448 | 443.6 | 474.0 | 748.8 | [local](../records/ffn-final-10000-s461-bottleneck_linear.json) |
| 8 | branch_softmax | 2.426687 | 51.43% | 8,654,208 / 4,718,592 | 454.5 | 480.0 | 785.1 | [local](../records/ffn-final-10000-s461-branch_softmax.json) |
| 9 | convolution_raw | 2.427686 | 51.40% | 8,654,208 / 4,718,592 | 452.4 | 480.0 | 881.1 | [local](../records/ffn-final-10000-s461-convolution_raw.json) |
| 10 | convolution_squared | 2.428213 | 51.45% | 8,654,208 / 4,718,592 | 456.4 | 482.0 | 892.5 | [local](../records/ffn-final-10000-s461-convolution_squared.json) |
| 11 | hybrid_normalized | 2.430317 | 51.32% | 8,654,208 / 4,718,592 | 459.2 | 486.0 | 1318.3 | [local](../records/ffn-final-10000-s461-hybrid_normalized.json) |
| 12 | dynamic_filter | 2.432455 | 51.34% | 8,654,208 / 4,718,592 | 448.4 | 480.0 | 825.1 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 13 | divisive_16 | 2.438159 | 51.27% | 8,654,208 / 4,718,592 | 451.4 | 482.0 | 686.9 | [local](../records/ffn-final-10000-s461-divisive_16.json) |
| 14 | cyclic_relu | 2.440047 | 51.17% | 8,654,208 / 4,718,592 | 447.1 | 480.0 | 559.0 | [local](../records/ffn-final-10000-s461-cyclic_relu.json) |
| 15 | axis_parallel | 2.445161 | 51.21% | 8,654,208 / 4,718,592 | 460.8 | 492.0 | 926.2 | [local](../records/ffn-final-10000-s461-axis_parallel.json) |
| 16 | contrast32 | 2.483469 | 50.67% | 8,654,208 / 4,718,592 | 420.8 | 460.0 | 780.1 | [local](../records/ffn-final-10000-s461-contrast32.json) |
| 17 | contrast128 | 2.484188 | 50.63% | 8,654,208 / 4,718,592 | 421.2 | 460.0 | 745.7 | [local](../records/ffn-final-10000-s461-contrast128.json) |
| 18 | distributed_fixed | 2.492875 | 50.33% | 8,654,208 / 4,718,592 | 476.0 | 500.0 | 671.5 | [local](../records/ffn-final-10000-s461-distributed_fixed.json) |
| 19 | expanded_curve | 2.526605 | 49.53% | 8,653,056 / 4,717,440 | 875.0 | 920.0 | 1628.0 | [local](../records/ffn-final-10000-s461-expanded_curve.json) |
| 20 | associative_normalized | 2.532203 | 49.76% | 8,654,208 / 4,718,592 | 472.8 | 498.0 | 927.4 | [local](../records/ffn-final-10000-s461-associative_normalized.json) |
| 21 | cyclic_signed | 2.537507 | 49.55% | 8,654,208 / 4,718,592 | 448.4 | 480.0 | 672.4 | [local](../records/ffn-final-10000-s461-cyclic_signed.json) |
| 22 | dense | 2.549469 | 49.40% | 8,654,208 / 4,718,592 | 436.9 | 462.0 | 660.1 | [local](../records/ffn-final-10000-s461-dense.json) |
| 23 | channel_curve_original | 2.585674 | 49.10% | 5,133,696 / 1,198,080 | 437.4 | 464.0 | 995.9 | [local](../records/ffn-final-10000-s461-channel_curve_original.json) |
| 24 | associative_raw | 2.590846 | 48.80% | 8,654,208 / 4,718,592 | 447.2 | 472.0 | 1059.8 | [local](../records/ffn-final-10000-s461-associative_raw.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `32675da46b70`

4 layers; width 384; 256 raw tokens.
Seed 23; 6,004,377 identical scored targets; schedule `68d7786248b5`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | shifted_relu_squared | 3.354515 | 38.66% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 143.8 | [local](../records/ffn-single-confirm-1800-s23-shifted_relu_squared.json) |
| 2 | relu_squared | 3.361007 | 38.65% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 138.1 | [local](../records/ffn-single-confirm-1800-s23-relu_squared.json) |
| 3 | self_gate_fixed | 3.361668 | 38.56% | 8,654,208 / 4,718,592 | 447.8 | 468.0 | 139.0 | [local](../records/ffn-single-confirm-1800-s23-self_gate_fixed.json) |
| 4 | dense | 3.373691 | 38.34% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 142.3 | [local](../records/ffn-single-confirm-1800-s23-dense.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `35d4b2769d86`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | lattice | 3.399442 | 38.16% | 8,653,184 / 4,717,568 | 465.1 | 492.0 | 187.7 | [local](../records/ffn-lattice-1800-s461-lattice.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `36bf279d9fbb`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | fold4 | 3.658606 | 34.76% | 8,654,208 / 4,718,592 | 451.4 | 488.0 | 185.1 | [local](../records/ffn-fold-1800-s461-fold4.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `3a47f9d979d5`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | competitive_diagonal | 3.369100 | 38.55% | 8,654,208 / 4,718,592 | 455.3 | 470.0 | 188.7 | [local](../records/ffn-competitive-1800-s461-competitive_diagonal.json) |
| 2 | competitive_256 | 3.370112 | 38.57% | 8,654,208 / 4,718,592 | 455.3 | 470.0 | 182.4 | [local](../records/ffn-competitive-1800-s461-competitive_256.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `3efab775b2b9`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.340158 | 39.13% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 147.6 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 2 | relu_squared | 3.357382 | 38.65% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 120.8 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 3 | shifted_relu_squared | 3.361548 | 38.51% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 133.5 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 4 | dense | 3.368282 | 38.53% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 124.5 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `40a72d28e862`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | contrast32 | 3.441675 | 37.63% | 8,654,208 / 4,718,592 | 422.4 | 448.0 | 128.4 | [local](../records/ffn-energy-contrast-1800-s461-contrast32.json) |
| 2 | contrast128 | 3.447880 | 37.53% | 8,654,208 / 4,718,592 | 422.2 | 448.0 | 125.2 | [local](../records/ffn-energy-contrast-1800-s461-contrast128.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `442fa71bc080`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.344238 | 39.09% | 8,654,208 / 4,718,592 | 459.4 | 482.0 | 173.7 | [local](../records/ffn-ridge-confirm-1800-s293-ridge_1.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `468213268c67`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | symmetric_pairs | 3.368154 | 38.48% | 8,654,208 / 4,718,592 | 430.1 | 454.0 | 172.7 | [local](../records/ffn-exterior-1800-s461-symmetric_pairs.json) |
| 2 | exterior | 3.369273 | 38.56% | 8,654,208 / 4,718,592 | 429.6 | 454.0 | 176.2 | [local](../records/ffn-exterior-1800-s461-exterior.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `46850c14c66b`

4 layers; width 384; 256 raw tokens.
Seed 131; 6,036,217 identical scored targets; schedule `f70bb98e4209`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | relu_squared | 3.357899 | 38.65% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 56.0 | [local](../records/ffn-global-1800-s131-relu_squared.json) |
| 2 | dense | 3.390508 | 38.09% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 55.4 | [local](../records/ffn-global-1800-s131-dense.json) |
| 3 | global_squared | 3.398010 | 38.12% | 8,654,208 / 4,718,592 | 426.5 | 450.0 | 51.9 | [local](../records/ffn-global-1800-s131-global_squared.json) |
| 4 | global_silu | 3.620380 | 34.77% | 8,654,208 / 4,718,592 | 431.4 | 450.0 | 57.0 | [local](../records/ffn-global-1800-s131-global_silu.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `473707211823`

4 layers; width 384; 256 raw tokens.
Seed 47; 6,036,303 identical scored targets; schedule `a9c9d0456ded`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | shifted_relu_squared | 3.345774 | 38.87% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 136.8 | [local](../records/ffn-single-confirm-1800-s47-shifted_relu_squared.json) |
| 2 | relu_squared | 3.349586 | 38.84% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 156.3 | [local](../records/ffn-single-confirm-1800-s47-relu_squared.json) |
| 3 | self_gate_fixed | 3.360859 | 38.75% | 8,654,208 / 4,718,592 | 447.8 | 468.0 | 147.0 | [local](../records/ffn-single-confirm-1800-s47-self_gate_fixed.json) |
| 4 | dense | 3.389255 | 38.15% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 139.4 | [local](../records/ffn-single-confirm-1800-s47-dense.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `4cd587cccc81`

6 layers; width 512; 256 raw tokens.
Seed 71; 6,009,808 identical scored targets; schedule `9cecb6d433f5`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter_transfer | 3.144704 | 41.21% | 21,076,480 / 12,681,216 | 833.7 | 852.0 | 215.7 | [local](../records/ffn-dynamic-filter-transfer-1800-s71-dynamic_filter_transfer.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `4fe1b6824485`

4 layers; width 384; 256 raw tokens.
Seed 173; 6,003,971 identical scored targets; schedule `520733d86928`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | relu_squared | 3.352462 | 38.71% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 112.1 | [local](../records/ffn-cyclic-1800-s173-relu_squared.json) |
| 2 | cyclic_relu | 3.362832 | 38.51% | 8,654,208 / 4,718,592 | 447.3 | 468.0 | 116.8 | [local](../records/ffn-cyclic-1800-s173-cyclic_relu.json) |
| 3 | cyclic_signed | 3.368954 | 38.41% | 8,654,208 / 4,718,592 | 446.5 | 468.0 | 118.5 | [local](../records/ffn-cyclic-1800-s173-cyclic_signed.json) |
| 4 | shifted_relu_squared | 3.372779 | 38.27% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 97.1 | [local](../records/ffn-cyclic-1800-s173-shifted_relu_squared.json) |
| 5 | dense | 3.380961 | 38.34% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 106.7 | [local](../records/ffn-cyclic-1800-s173-dense.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `50e28a6bfaa7`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | post_filter | 3.374329 | 38.35% | 8,654,208 / 4,718,592 | 468.9 | 488.0 | 151.6 | [local](../records/ffn-post-filter-1800-s461-post_filter.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `54f450449db8`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | two_hop | 3.356606 | 39.07% | 8,654,208 / 4,718,592 | 456.9 | 474.0 | 126.1 | [local](../records/ffn-two-hop-filter-1800-s461-two_hop.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `57bbe07938e8`

4 layers; width 384; 256 raw tokens.
Seed 197; 6,025,451 identical scored targets; schedule `6042d2050765`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | relu_squared | 3.343508 | 38.94% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 65.0 | [local](../records/ffn-bottleneck-1800-s197-relu_squared.json) |
| 2 | bottleneck_squared | 3.344227 | 38.82% | 8,648,064 / 4,712,448 | 445.1 | 474.0 | 177.2 | [local](../records/ffn-bottleneck-1800-s197-bottleneck_squared.json) |
| 3 | bottleneck_silu | 3.351270 | 38.76% | 8,648,064 / 4,712,448 | 445.1 | 474.0 | 95.9 | [local](../records/ffn-bottleneck-1800-s197-bottleneck_silu.json) |
| 4 | bottleneck_linear | 3.351841 | 38.75% | 8,648,064 / 4,712,448 | 444.6 | 474.0 | 136.4 | [local](../records/ffn-bottleneck-1800-s197-bottleneck_linear.json) |
| 5 | shifted_relu_squared | 3.354247 | 38.62% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 146.7 | [local](../records/ffn-bottleneck-1800-s197-shifted_relu_squared.json) |
| 6 | dense | 3.389887 | 38.17% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 61.0 | [local](../records/ffn-bottleneck-1800-s197-dense.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `646b457a5f94`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | gatebank_m4 | 3.386655 | 38.49% | 8,651,136 / 4,715,520 | 425.3 | 452.0 | 128.9 | [local](../records/ffn-gatebank-1800-s461-gatebank_m4.json) |
| 2 | gatebank_m2 | 3.393742 | 38.43% | 8,654,208 / 4,718,592 | 429.3 | 450.0 | 125.8 | [local](../records/ffn-gatebank-1800-s461-gatebank_m2.json) |
| 3 | gatebank_silu_m4 | 3.397856 | 38.10% | 8,649,600 / 4,713,984 | 427.6 | 452.0 | 123.8 | [local](../records/ffn-gatebank-1800-s461-gatebank_silu_m4.json) |
| 4 | gatebank_m1 | 3.411986 | 38.13% | 8,654,208 / 4,718,592 | 432.4 | 454.0 | 111.8 | [local](../records/ffn-gatebank-1800-s461-gatebank_m1.json) |

### HuggingFaceTB/smol-smoltalk / 10000 updates / `689cf5f6df86`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dual_plain_quarter | 2.415987 | 51.54% | 8,654,208 / 4,718,592 | 441.8 | 468.0 | 985.7 | [local](../records/ffn-dual-path-10000-s461-dual_plain_quarter.json) |
| 2 | dual_router_small | 2.416228 | 51.47% | 8,654,208 / 4,718,592 | 452.7 | 470.0 | 1142.3 | [local](../records/ffn-dual-path-10000-s461-dual_router_small.json) |
| 3 | dual_filter_small | 2.423266 | 51.35% | 8,654,208 / 4,718,592 | 451.5 | 468.0 | 1088.8 | [local](../records/ffn-dual-path-10000-s461-dual_filter_small.json) |
| 4 | dual_filter_quarter | 2.423454 | 51.49% | 8,654,208 / 4,718,592 | 448.9 | 478.0 | 1090.5 | [local](../records/ffn-dual-path-10000-s461-dual_filter_quarter.json) |
| 5 | dual_filter_half | 2.424349 | 51.44% | 8,654,208 / 4,718,592 | 450.8 | 474.0 | 1090.3 | [local](../records/ffn-dual-path-10000-s461-dual_filter_half.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `6e867a7fa7bd`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | matrix_chain | 3.407659 | 38.38% | 8,654,208 / 4,718,592 | 424.4 | 452.0 | 177.7 | [local](../records/ffn-matrix-chain-1800-s461-matrix_chain.json) |
| 2 | matrix_parallel | 3.417661 | 37.94% | 8,654,208 / 4,718,592 | 423.3 | 452.0 | 177.2 | [local](../records/ffn-matrix-chain-1800-s461-matrix_parallel.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `77a6452f8504`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | pair_normalized | 3.353935 | 38.89% | 8,654,208 / 4,718,592 | 433.0 | 482.0 | 185.6 | [local](../records/ffn-pair-1800-s461-pair_normalized.json) |
| 2 | pair_raw | 3.355598 | 38.81% | 8,654,208 / 4,718,592 | 444.4 | 468.0 | 161.7 | [local](../records/ffn-pair-1800-s461-pair_raw.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `7ad9fed670d6`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | axis_parallel | 3.338122 | 39.13% | 8,654,208 / 4,718,592 | 460.0 | 492.0 | 166.0 | [local](../records/ffn-two-axis-1800-s461-axis_parallel.json) |

### HuggingFaceTB/smol-smoltalk / 600 updates / `7d54fb13c977`

4 layers; width 384; 256 raw tokens.
Seed 17; 2,005,324 identical scored targets; schedule `2e14b0261112`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | self_gate_learned | 4.212373 | 29.28% | 8,654,176 / 4,718,560 | 469.9 | 508.0 | 50.1 | [local](../records/ffn-single-600-v1-self_gate_learned.json) |
| 2 | self_gate_fixed | 4.212848 | 29.10% | 8,654,208 / 4,718,592 | 447.8 | 468.0 | 38.4 | [local](../records/ffn-single-600-v1-self_gate_fixed.json) |
| 3 | dense | 4.227783 | 29.03% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 37.6 | [local](../records/ffn-single-600-v1-dense.json) |
| 4 | shifted_relu_squared | 4.229148 | 29.48% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 48.5 | [local](../records/ffn-single-600-v1-shifted_relu_squared.json) |
| 5 | relu_squared | 4.231309 | 29.39% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 47.2 | [local](../records/ffn-single-600-v1-relu_squared.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `a69710c97156`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | divisive_global | 3.359828 | 38.71% | 8,654,208 / 4,718,592 | 445.4 | 470.0 | 150.4 | [local](../records/ffn-divisive-1800-s461-divisive_global.json) |
| 2 | divisive_16 | 3.396719 | 38.31% | 8,654,208 / 4,718,592 | 452.3 | 470.0 | 154.0 | [local](../records/ffn-divisive-1800-s461-divisive_16.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `a9605ff6ed7f`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | distributed_fixed | 3.428987 | 37.52% | 8,654,208 / 4,718,592 | 471.4 | 492.0 | 136.1 | [local](../records/ffn-distributed-curve-1800-s461-distributed_fixed.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `ad4dbc63dd18`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | phase_rotation | 3.345356 | 39.06% | 8,654,208 / 4,718,592 | 460.9 | 474.0 | 169.6 | [local](../records/ffn-phase-1800-s461-phase_rotation.json) |
| 2 | phase_diagonal | 3.348351 | 38.83% | 8,654,208 / 4,718,592 | 454.1 | 474.0 | 155.2 | [local](../records/ffn-phase-1800-s461-phase_diagonal.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `add6c6661f55`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | axis_parallel | 3.328431 | 39.12% | 8,654,208 / 4,718,592 | 464.4 | 492.0 | 163.0 | [local](../records/ffn-two-axis-confirm-1800-s311-axis_parallel.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `b0c31c89b710`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.362123 | 38.86% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 192.9 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | relu_squared | 3.368997 | 38.57% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 83.5 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 3 | shifted_relu_squared | 3.374200 | 38.50% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 102.5 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 4 | dense | 3.375814 | 38.51% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 87.6 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 5 | product_dictionary | 3.463462 | 36.77% | 8,654,208 / 4,718,592 | 421.8 | 452.0 | 157.0 | [local](../records/ffn-dictionary-1800-s461-product_dictionary.json) |
| 6 | flat_dictionary | 3.532873 | 35.52% | 8,654,208 / 4,718,592 | 448.4 | 468.0 | 149.9 | [local](../records/ffn-dictionary-1800-s461-flat_dictionary.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `b4a47704b9dd`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.339692 | 39.07% | 8,654,208 / 4,718,592 | 459.4 | 482.0 | 216.5 | [local](../records/ffn-ridge-confirm-1800-s311-ridge_1.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `b7767ee6f51b`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.351981 | 39.00% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 173.5 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 3.371116 | 38.40% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 132.4 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 3 | relu_squared | 3.375245 | 38.37% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 111.8 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 4 | dense | 3.387184 | 38.27% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 110.1 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `b7bdbd67e33f`

4 layers; width 384; 256 raw tokens.
Seed 277; 6,030,702 identical scored targets; schedule `45bdbb23ef3f`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.337261 | 39.11% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 161.5 | [local](../records/ffn-hybrid-1800-s277-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 3.348131 | 38.84% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 124.6 | [local](../records/ffn-hybrid-1800-s277-shifted_relu_squared.json) |
| 3 | hybrid_local | 3.350322 | 38.80% | 8,654,208 / 4,718,592 | 448.7 | 472.0 | 148.6 | [local](../records/ffn-hybrid-1800-s277-hybrid_local.json) |
| 4 | relu_squared | 3.355939 | 38.71% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 122.2 | [local](../records/ffn-hybrid-1800-s277-relu_squared.json) |
| 5 | dense | 3.386486 | 38.13% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 130.9 | [local](../records/ffn-hybrid-1800-s277-dense.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `bf45cc05490d`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 3.359723 | 38.51% | 8,654,208 / 4,718,592 | 448.0 | 468.0 | 170.9 | [local](../records/ffn-branch-1800-s461-branch_sigmoid.json) |
| 2 | branch_softmax | 3.366048 | 38.57% | 8,654,208 / 4,718,592 | 445.8 | 468.0 | 168.5 | [local](../records/ffn-branch-1800-s461-branch_softmax.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `dae0d7e49b16`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.347085 | 38.79% | 8,654,208 / 4,718,592 | 446.9 | 468.0 | 147.3 | [local](../records/ffn-dynamic-filter-confirm-1800-s293-dynamic_filter.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `dd25e30aa846`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | pair_normalized | 3.353779 | 38.71% | 8,654,208 / 4,718,592 | 430.8 | 462.0 | 185.4 | [local](../records/ffn-pair-confirm-1800-s293-pair_normalized.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `e06a4b34c685`

4 layers; width 384; 256 raw tokens.
Seed 359; 6,036,078 identical scored targets; schedule `8b1b92893968`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.344550 | 38.97% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 171.0 | [local](../records/ffn-linear-feature-1800-s359-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 3.365062 | 38.50% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 90.8 | [local](../records/ffn-linear-feature-1800-s359-shifted_relu_squared.json) |
| 3 | relu_squared | 3.368650 | 38.47% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 119.7 | [local](../records/ffn-linear-feature-1800-s359-relu_squared.json) |
| 4 | dense | 3.375793 | 38.38% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 123.8 | [local](../records/ffn-linear-feature-1800-s359-dense.json) |
| 5 | feature_linear_squared | 3.386192 | 38.17% | 8,654,208 / 4,718,592 | 460.6 | 478.0 | 180.2 | [local](../records/ffn-linear-feature-1800-s359-feature_linear_squared.json) |
| 6 | feature_linear_elu | 3.432628 | 37.37% | 8,654,208 / 4,718,592 | 460.6 | 478.0 | 191.8 | [local](../records/ffn-linear-feature-1800-s359-feature_linear_elu.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `e3d09b38e3ef`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | static_wide | 3.376262 | 38.44% | 8,655,232 / 4,719,616 | 447.6 | 468.0 | 146.5 | [local](../records/ffn-static-filter-1800-s461-static_wide.json) |
| 2 | static_narrow | 3.391998 | 38.15% | 8,262,016 / 4,326,400 | 438.9 | 468.0 | 151.2 | [local](../records/ffn-static-filter-1800-s461-static_narrow.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `e44975432e18`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | maxout_2 | 3.503124 | 36.56% | 8,654,208 / 4,718,592 | 453.6 | 478.0 | 141.8 | [local](../records/ffn-competitive-pool-1800-s461-maxout_2.json) |
| 2 | maxout_3 | 3.514838 | 36.44% | 8,654,208 / 4,718,592 | 433.7 | 478.0 | 136.3 | [local](../records/ffn-competitive-pool-1800-s461-maxout_3.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `e6da8cfc27eb`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | pair_normalized | 3.349054 | 38.85% | 8,654,208 / 4,718,592 | 435.6 | 462.0 | 183.1 | [local](../records/ffn-pair-confirm-1800-s311-pair_normalized.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `e99cbd32b0f3`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | self_correlation | 3.364279 | 38.67% | 8,654,208 / 4,718,592 | 454.1 | 476.0 | 167.2 | [local](../records/ffn-correlation-1800-s461-self_correlation.json) |
| 2 | correlation_diagonal | 3.367135 | 38.59% | 8,654,208 / 4,718,592 | 453.4 | 476.0 | 167.5 | [local](../records/ffn-correlation-1800-s461-correlation_diagonal.json) |

### HuggingFaceTB/smol-smoltalk / 10000 updates / `ecbf85ac7385`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | gate16 | 2.412582 | 51.76% | 8,654,208 / 4,718,592 | 446.9 | 472.0 | 1365.8 | [local](../records/ffn-evolution-10000-s461-gate16.json) |
| 2 | partial_positive_sine | 2.413345 | 51.55% | 8,654,208 / 4,718,592 | 457.5 | 484.0 | 1329.8 | [local](../records/ffn-evolution-10000-s461-partial_positive_sine.json) |
| 3 | partial_rational | 2.413616 | 51.61% | 8,654,208 / 4,718,592 | 457.5 | 484.0 | 1466.2 | [local](../records/ffn-evolution-10000-s461-partial_rational.json) |
| 4 | gate_sine | 2.418575 | 51.52% | 8,654,208 / 4,718,592 | 448.0 | 468.0 | 1034.4 | [local](../records/ffn-evolution-10000-s461-gate_sine.json) |
| 5 | partial_sine | 2.422035 | 51.49% | 8,654,208 / 4,718,592 | 457.5 | 482.0 | 1435.7 | [local](../records/ffn-evolution-10000-s461-partial_sine.json) |
| 6 | gate8 | 2.426850 | 51.31% | 8,654,208 / 4,718,592 | 446.6 | 472.0 | 1104.4 | [local](../records/ffn-evolution-10000-s461-gate8.json) |
| 7 | filter_updategate | 2.440036 | 51.28% | 8,654,208 / 4,718,592 | 472.3 | 508.0 | 1141.6 | [local](../records/ffn-evolution-10000-s461-filter_updategate.json) |
| 8 | filter_postgate | 2.451138 | 50.93% | 8,654,208 / 4,718,592 | 469.7 | 508.0 | 1132.1 | [local](../records/ffn-evolution-10000-s461-filter_postgate.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `f08c102ce325`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 39.01% | 8,654,208 / 4,718,592 | 446.9 | 468.0 | 155.2 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `f2b9a6856be2`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | filter_bank16 | 3.347661 | 38.91% | 8,654,208 / 4,718,592 | 449.6 | 472.0 | 157.8 | [local](../records/ffn-filter-bank-1800-s461-filter_bank16.json) |
| 2 | filter_bank4 | 3.348846 | 39.00% | 8,654,208 / 4,718,592 | 449.6 | 472.0 | 153.2 | [local](../records/ffn-filter-bank-1800-s461-filter_bank4.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `f4c86eb83f20`

6 layers; width 512; 256 raw tokens.
Seed 71; 6,009,808 identical scored targets; schedule `9cecb6d433f5`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dense | 3.169490 | 40.80% | 21,076,480 / 12,681,216 | 824.1 | 848.0 | 178.3 | [local](../records/ffn-single-scale-1800-s71-dense.json) |
| 2 | shifted_relu_squared | 3.182611 | 40.66% | 21,076,564 / 12,681,300 | 786.8 | 862.0 | 176.6 | [local](../records/ffn-single-scale-1800-s71-shifted_relu_squared.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `f4d461ea7d29`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | jet_factored | 3.376662 | 38.43% | 8,655,744 / 4,720,128 | 460.1 | 480.0 | 213.8 | [local](../records/ffn-jet-factored-1800-s461-jet_factored.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `f9e40f874579`

6 layers; width 512; 256 raw tokens.
Seed 71; 6,009,808 identical scored targets; schedule `9cecb6d433f5`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_transfer | 3.140375 | 41.40% | 21,076,480 / 12,681,216 | 850.4 | 878.0 | 278.9 | [local](../records/ffn-hybrid-transfer-1800-s71-hybrid_transfer.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `fa6c0482e1a4`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | tensor_banks | 3.389340 | 38.12% | 8,654,208 / 4,718,592 | 442.5 | 474.0 | 106.4 | [local](../records/ffn-tensor-response-1800-s461-tensor_banks.json) |
| 2 | tensor_full | 3.413476 | 37.89% | 8,654,208 / 4,718,592 | 421.6 | 474.0 | 140.4 | [local](../records/ffn-tensor-response-1800-s461-tensor_full.json) |

### HuggingFaceTB/smol-smoltalk / 1800 updates / `feb84f4598a1`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | jet_split | 3.397434 | 38.02% | 8,654,208 / 4,718,592 | 430.3 | 450.0 | 160.3 | [local](../records/ffn-jet-1800-s461-jet_split.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `05df563da10f`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | filter_tanh | 4.588307 | 21.92% | 8,654,208 / 4,718,592 | 452.4 | 474.0 | 156.4 | [local](../records/ffn-nonlinear-filter-1800-s461-filter_tanh.json) |
| 2 | filter_signed_square | 4.602801 | 21.73% | 8,654,208 / 4,718,592 | 456.9 | 474.0 | 159.4 | [local](../records/ffn-nonlinear-filter-1800-s461-filter_signed_square.json) |

### HuggingFaceFW/fineweb-edu / 5400 updates / `089e67c87f18`

4 layers; width 384; 256 raw tokens.
Seed 461; 18,067,829 identical scored targets; schedule `1b98493063b4`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.060270 | 28.06% | 8,654,208 / 4,718,592 | 446.9 | 468.0 | 427.3 | [local](../records/ffn-filter-continuation-5400-s461-dynamic_filter.json) |
| 2 | dense | 4.166560 | 25.97% | 8,654,208 / 4,718,592 | 436.4 | 466.0 | 329.2 | [local](../records/ffn-filter-continuation-5400-s461-dense.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `0a71939a74d8`

4 layers; width 384; 256 raw tokens.
Seed 239; 6,034,737 identical scored targets; schedule `e11a74cf1f9b`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | relu_squared | 4.604727 | 21.52% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 139.7 | [local](../records/ffn-associative-1800-s239-relu_squared.json) |
| 2 | shifted_relu_squared | 4.605155 | 21.41% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 113.1 | [local](../records/ffn-associative-1800-s239-shifted_relu_squared.json) |
| 3 | dense | 4.648254 | 20.82% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 141.4 | [local](../records/ffn-associative-1800-s239-dense.json) |
| 4 | associative_normalized | 4.672715 | 20.80% | 8,654,208 / 4,718,592 | 476.8 | 498.0 | 172.8 | [local](../records/ffn-associative-1800-s239-associative_normalized.json) |
| 5 | associative_raw | 4.679615 | 20.58% | 8,654,208 / 4,718,592 | 444.6 | 472.0 | 148.3 | [local](../records/ffn-associative-1800-s239-associative_raw.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `0cd79764e28f`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | axis_parallel | 4.600753 | 21.63% | 8,654,208 / 4,718,592 | 460.0 | 492.0 | 165.8 | [local](../records/ffn-two-axis-confirm-1800-s293-axis_parallel.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `0f6fb5414f17`

4 layers; width 384; 256 raw tokens.
Seed 101; 6,027,819 identical scored targets; schedule `c8535b205682`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | relu_squared | 4.610242 | 21.49% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 121.7 | [local](../records/ffn-feature-1800-s101-relu_squared.json) |
| 2 | group_refine | 4.616297 | 21.39% | 8,646,016 / 4,710,400 | 445.4 | 470.0 | 100.7 | [local](../records/ffn-feature-1800-s101-group_refine.json) |
| 3 | group_product | 4.616769 | 21.39% | 8,646,016 / 4,710,400 | 467.1 | 490.0 | 115.3 | [local](../records/ffn-feature-1800-s101-group_product.json) |
| 4 | dense | 4.628968 | 21.22% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 116.3 | [local](../records/ffn-feature-1800-s101-dense.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `14a7ea39034f`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.590279 | 21.99% | 8,654,208 / 4,718,592 | 459.4 | 482.0 | 117.8 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | ridge_01 | 4.603095 | 21.80% | 8,654,208 / 4,718,592 | 459.4 | 482.0 | 137.9 | [local](../records/ffn-ridge-1800-s461-ridge_01.json) |

### HuggingFaceFW/fineweb-edu / 10000 updates / `1631256244b4`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | router_mlp | 3.828213 | 31.15% | 8,654,208 / 4,718,592 | 454.4 | 472.0 | 1042.8 | [local](../records/ffn-branch-refinement-10000-s461-router_mlp.json) |
| 2 | branch2 | 3.831924 | 31.05% | 8,654,208 / 4,718,592 | 447.0 | 468.0 | 964.4 | [local](../records/ffn-branch-refinement-10000-s461-branch2.json) |
| 3 | ungated | 3.831977 | 31.06% | 8,654,208 / 4,718,592 | 425.6 | 448.0 | 865.3 | [local](../records/ffn-branch-refinement-10000-s461-ungated.json) |
| 4 | matrix4x4 | 3.847200 | 30.80% | 8,654,208 / 4,718,592 | 447.0 | 472.0 | 1378.2 | [local](../records/ffn-branch-refinement-10000-s461-matrix4x4.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `16a449d0779f`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | reflective | 4.598324 | 21.64% | 8,654,208 / 4,718,592 | 442.0 | 460.0 | 203.5 | [local](../records/ffn-reflective-1800-s461-reflective.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `1dcde507f8a5`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | convolution_raw | 4.591732 | 21.80% | 8,654,208 / 4,718,592 | 448.7 | 468.0 | 167.2 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 2 | convolution_squared | 4.606602 | 21.73% | 8,654,208 / 4,718,592 | 452.7 | 470.0 | 170.4 | [local](../records/ffn-convolution-1800-s461-convolution_squared.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `22d10226d422`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.574550 | 22.09% | 8,654,208 / 4,718,592 | 447.4 | 468.0 | 152.0 | [local](../records/ffn-dynamic-filter-confirm-1800-s311-dynamic_filter.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `266a5ebc4d04`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | wide_conv | 4.588174 | 21.94% | 8,654,208 / 4,718,592 | 449.4 | 472.0 | 177.6 | [local](../records/ffn-wide-convolution-1800-s461-wide_conv.json) |

### HuggingFaceFW/fineweb-edu / 10000 updates / `26d13f07961c`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | evidence_balance | 3.820250 | 31.20% | 8,654,208 / 4,718,592 | 448.1 | 468.0 | 1928.5 | [local](../records/ffn-evidence-10000-s461-evidence_balance.json) |
| 2 | reaction_catalyst | 3.820372 | 31.22% | 8,654,208 / 4,718,592 | 470.1 | 490.0 | 1385.4 | [local](../records/ffn-evidence-10000-s461-reaction_catalyst.json) |
| 3 | attention_energy | 3.820679 | 31.25% | 8,654,208 / 4,718,592 | 460.5 | 488.0 | 1020.9 | [local](../records/ffn-evidence-10000-s461-attention_energy.json) |
| 4 | evidence_support | 3.820992 | 31.20% | 8,654,208 / 4,718,592 | 448.1 | 468.0 | 1123.4 | [local](../records/ffn-evidence-10000-s461-evidence_support.json) |
| 5 | reaction_post | 3.823193 | 31.29% | 8,654,208 / 4,718,592 | 446.9 | 468.0 | 1016.8 | [local](../records/ffn-evidence-10000-s461-reaction_post.json) |
| 6 | attention_alignment | 3.825272 | 31.25% | 8,654,208 / 4,718,592 | 463.1 | 488.0 | 1015.4 | [local](../records/ffn-evidence-10000-s461-attention_alignment.json) |
| 7 | cross_evidence | 3.827581 | 31.25% | 8,654,208 / 4,718,592 | 454.3 | 472.0 | 1924.5 | [local](../records/ffn-evidence-10000-s461-cross_evidence.json) |
| 8 | learned_evidence | 3.841977 | 30.94% | 8,654,208 / 4,718,592 | 450.9 | 468.0 | 1184.8 | [local](../records/ffn-evidence-10000-s461-learned_evidence.json) |

### HuggingFaceFW/fineweb-edu / 10000 updates / `2898c8c1d063`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | branch_hierarchical | 3.823591 | 31.19% | 8,654,208 / 4,718,592 | 444.6 | 482.0 | 1510.7 | [local](../records/ffn-branch-next-10000-s461-branch_hierarchical.json) |
| 2 | branch_rich_router | 3.829860 | 31.16% | 8,654,208 / 4,718,592 | 450.4 | 474.0 | 1118.5 | [local](../records/ffn-branch-next-10000-s461-branch_rich_router.json) |
| 3 | branch_threshold | 3.832266 | 31.05% | 8,654,208 / 4,718,592 | 451.1 | 472.0 | 1204.0 | [local](../records/ffn-branch-next-10000-s461-branch_threshold.json) |
| 4 | branch_temperature | 3.844234 | 30.89% | 8,654,208 / 4,718,592 | 450.0 | 472.0 | 1016.0 | [local](../records/ffn-branch-next-10000-s461-branch_temperature.json) |
| 5 | branch_bipolar | 3.897990 | 30.14% | 8,654,208 / 4,718,592 | 471.5 | 508.0 | 1034.5 | [local](../records/ffn-branch-next-10000-s461-branch_bipolar.json) |

### HuggingFaceFW/fineweb-edu / 10000 updates / `325904e3a840`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 3.820069 | 31.29% | 8,654,208 / 4,718,592 | 454.5 | 480.0 | 848.8 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | competitive_256 | 3.828861 | 30.98% | 8,654,208 / 4,718,592 | 454.8 | 482.0 | 916.7 | [local](../records/ffn-final-10000-s461-competitive_256.json) |
| 3 | correlation_diagonal | 3.831073 | 31.07% | 8,654,208 / 4,718,592 | 451.5 | 488.0 | 845.8 | [local](../records/ffn-final-10000-s461-correlation_diagonal.json) |
| 4 | bottleneck_squared | 3.831778 | 30.98% | 8,648,064 / 4,712,448 | 444.1 | 474.0 | 806.9 | [local](../records/ffn-final-10000-s461-bottleneck_squared.json) |
| 5 | competitive_diagonal | 3.832150 | 31.00% | 8,654,208 / 4,718,592 | 454.8 | 482.0 | 911.8 | [local](../records/ffn-final-10000-s461-competitive_diagonal.json) |
| 6 | bottleneck_silu | 3.832321 | 31.07% | 8,648,064 / 4,712,448 | 444.1 | 474.0 | 821.6 | [local](../records/ffn-final-10000-s461-bottleneck_silu.json) |
| 7 | bottleneck_linear | 3.834979 | 31.04% | 8,648,064 / 4,712,448 | 443.6 | 474.0 | 748.8 | [local](../records/ffn-final-10000-s461-bottleneck_linear.json) |
| 8 | convolution_raw | 3.837629 | 31.08% | 8,654,208 / 4,718,592 | 452.4 | 480.0 | 881.1 | [local](../records/ffn-final-10000-s461-convolution_raw.json) |
| 9 | convolution_squared | 3.838434 | 31.04% | 8,654,208 / 4,718,592 | 456.4 | 482.0 | 892.5 | [local](../records/ffn-final-10000-s461-convolution_squared.json) |
| 10 | branch_softmax | 3.838659 | 31.00% | 8,654,208 / 4,718,592 | 454.5 | 480.0 | 785.1 | [local](../records/ffn-final-10000-s461-branch_softmax.json) |
| 11 | dynamic_filter | 3.842274 | 31.09% | 8,654,208 / 4,718,592 | 448.4 | 480.0 | 825.1 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 12 | hybrid_normalized | 3.844517 | 30.93% | 8,654,208 / 4,718,592 | 459.2 | 486.0 | 1318.3 | [local](../records/ffn-final-10000-s461-hybrid_normalized.json) |
| 13 | divisive_16 | 3.847830 | 30.98% | 8,654,208 / 4,718,592 | 451.4 | 482.0 | 686.9 | [local](../records/ffn-final-10000-s461-divisive_16.json) |
| 14 | cyclic_relu | 3.859481 | 30.74% | 8,654,208 / 4,718,592 | 447.1 | 480.0 | 559.0 | [local](../records/ffn-final-10000-s461-cyclic_relu.json) |
| 15 | axis_parallel | 3.862047 | 30.86% | 8,654,208 / 4,718,592 | 460.8 | 492.0 | 926.2 | [local](../records/ffn-final-10000-s461-axis_parallel.json) |
| 16 | contrast32 | 3.883158 | 30.50% | 8,654,208 / 4,718,592 | 420.8 | 460.0 | 780.1 | [local](../records/ffn-final-10000-s461-contrast32.json) |
| 17 | contrast128 | 3.889906 | 30.35% | 8,654,208 / 4,718,592 | 421.2 | 460.0 | 745.7 | [local](../records/ffn-final-10000-s461-contrast128.json) |
| 18 | distributed_fixed | 3.898126 | 30.09% | 8,654,208 / 4,718,592 | 476.0 | 500.0 | 671.5 | [local](../records/ffn-final-10000-s461-distributed_fixed.json) |
| 19 | associative_normalized | 3.940869 | 29.48% | 8,654,208 / 4,718,592 | 472.8 | 498.0 | 927.4 | [local](../records/ffn-final-10000-s461-associative_normalized.json) |
| 20 | expanded_curve | 3.947265 | 29.19% | 8,653,056 / 4,717,440 | 875.0 | 920.0 | 1628.0 | [local](../records/ffn-final-10000-s461-expanded_curve.json) |
| 21 | cyclic_signed | 3.961265 | 28.84% | 8,654,208 / 4,718,592 | 448.4 | 480.0 | 672.4 | [local](../records/ffn-final-10000-s461-cyclic_signed.json) |
| 22 | channel_curve_original | 3.975031 | 29.25% | 5,133,696 / 1,198,080 | 437.4 | 464.0 | 995.9 | [local](../records/ffn-final-10000-s461-channel_curve_original.json) |
| 23 | dense | 3.980314 | 28.59% | 8,654,208 / 4,718,592 | 436.9 | 462.0 | 660.1 | [local](../records/ffn-final-10000-s461-dense.json) |
| 24 | associative_raw | 4.017801 | 28.22% | 8,654,208 / 4,718,592 | 447.2 | 472.0 | 1059.8 | [local](../records/ffn-final-10000-s461-associative_raw.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `32675da46b70`

4 layers; width 384; 256 raw tokens.
Seed 23; 6,004,377 identical scored targets; schedule `68d7786248b5`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | shifted_relu_squared | 4.603224 | 21.54% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 143.8 | [local](../records/ffn-single-confirm-1800-s23-shifted_relu_squared.json) |
| 2 | relu_squared | 4.609273 | 21.53% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 138.1 | [local](../records/ffn-single-confirm-1800-s23-relu_squared.json) |
| 3 | dense | 4.630876 | 21.08% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 142.3 | [local](../records/ffn-single-confirm-1800-s23-dense.json) |
| 4 | self_gate_fixed | 4.630951 | 21.07% | 8,654,208 / 4,718,592 | 447.8 | 468.0 | 139.0 | [local](../records/ffn-single-confirm-1800-s23-self_gate_fixed.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `35d4b2769d86`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | lattice | 4.634058 | 21.22% | 8,653,184 / 4,717,568 | 465.1 | 492.0 | 187.7 | [local](../records/ffn-lattice-1800-s461-lattice.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `36bf279d9fbb`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | fold4 | 4.855533 | 18.95% | 8,654,208 / 4,718,592 | 451.4 | 488.0 | 185.1 | [local](../records/ffn-fold-1800-s461-fold4.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `3a47f9d979d5`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | competitive_diagonal | 4.608517 | 21.55% | 8,654,208 / 4,718,592 | 455.3 | 470.0 | 188.7 | [local](../records/ffn-competitive-1800-s461-competitive_diagonal.json) |
| 2 | competitive_256 | 4.609286 | 21.54% | 8,654,208 / 4,718,592 | 455.3 | 470.0 | 182.4 | [local](../records/ffn-competitive-1800-s461-competitive_256.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `3efab775b2b9`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.594062 | 21.83% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 147.6 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 2 | relu_squared | 4.606856 | 21.44% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 120.8 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 3 | shifted_relu_squared | 4.610709 | 21.43% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 133.5 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 4 | dense | 4.628676 | 21.19% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 124.5 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `40a72d28e862`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | contrast32 | 4.662402 | 21.14% | 8,654,208 / 4,718,592 | 422.4 | 448.0 | 128.4 | [local](../records/ffn-energy-contrast-1800-s461-contrast32.json) |
| 2 | contrast128 | 4.667947 | 21.06% | 8,654,208 / 4,718,592 | 422.2 | 448.0 | 125.2 | [local](../records/ffn-energy-contrast-1800-s461-contrast128.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `442fa71bc080`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.591870 | 21.93% | 8,654,208 / 4,718,592 | 459.4 | 482.0 | 173.7 | [local](../records/ffn-ridge-confirm-1800-s293-ridge_1.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `468213268c67`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | symmetric_pairs | 4.601977 | 21.61% | 8,654,208 / 4,718,592 | 430.1 | 454.0 | 172.7 | [local](../records/ffn-exterior-1800-s461-symmetric_pairs.json) |
| 2 | exterior | 4.605759 | 21.58% | 8,654,208 / 4,718,592 | 429.6 | 454.0 | 176.2 | [local](../records/ffn-exterior-1800-s461-exterior.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `46850c14c66b`

4 layers; width 384; 256 raw tokens.
Seed 131; 6,036,217 identical scored targets; schedule `f70bb98e4209`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | relu_squared | 4.611759 | 21.38% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 56.0 | [local](../records/ffn-global-1800-s131-relu_squared.json) |
| 2 | global_squared | 4.642904 | 21.08% | 8,654,208 / 4,718,592 | 426.5 | 450.0 | 51.9 | [local](../records/ffn-global-1800-s131-global_squared.json) |
| 3 | dense | 4.649682 | 20.89% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 55.4 | [local](../records/ffn-global-1800-s131-dense.json) |
| 4 | global_silu | 4.808214 | 18.78% | 8,654,208 / 4,718,592 | 431.4 | 450.0 | 57.0 | [local](../records/ffn-global-1800-s131-global_silu.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `473707211823`

4 layers; width 384; 256 raw tokens.
Seed 47; 6,036,303 identical scored targets; schedule `a9c9d0456ded`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | shifted_relu_squared | 4.594372 | 21.69% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 136.8 | [local](../records/ffn-single-confirm-1800-s47-shifted_relu_squared.json) |
| 2 | relu_squared | 4.594802 | 21.72% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 156.3 | [local](../records/ffn-single-confirm-1800-s47-relu_squared.json) |
| 3 | self_gate_fixed | 4.622619 | 21.26% | 8,654,208 / 4,718,592 | 447.8 | 468.0 | 147.0 | [local](../records/ffn-single-confirm-1800-s47-self_gate_fixed.json) |
| 4 | dense | 4.650878 | 20.88% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 139.4 | [local](../records/ffn-single-confirm-1800-s47-dense.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `4cd587cccc81`

6 layers; width 512; 256 raw tokens.
Seed 71; 6,009,808 identical scored targets; schedule `9cecb6d433f5`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter_transfer | 4.450644 | 22.94% | 21,076,480 / 12,681,216 | 833.7 | 852.0 | 215.7 | [local](../records/ffn-dynamic-filter-transfer-1800-s71-dynamic_filter_transfer.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `4fe1b6824485`

4 layers; width 384; 256 raw tokens.
Seed 173; 6,003,971 identical scored targets; schedule `520733d86928`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | relu_squared | 4.601715 | 21.56% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 112.1 | [local](../records/ffn-cyclic-1800-s173-relu_squared.json) |
| 2 | cyclic_relu | 4.614036 | 21.41% | 8,654,208 / 4,718,592 | 447.3 | 468.0 | 116.8 | [local](../records/ffn-cyclic-1800-s173-cyclic_relu.json) |
| 3 | shifted_relu_squared | 4.621429 | 21.31% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 97.1 | [local](../records/ffn-cyclic-1800-s173-shifted_relu_squared.json) |
| 4 | cyclic_signed | 4.630289 | 21.19% | 8,654,208 / 4,718,592 | 446.5 | 468.0 | 118.5 | [local](../records/ffn-cyclic-1800-s173-cyclic_signed.json) |
| 5 | dense | 4.640691 | 20.96% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 106.7 | [local](../records/ffn-cyclic-1800-s173-dense.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `50e28a6bfaa7`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | post_filter | 4.614462 | 21.40% | 8,654,208 / 4,718,592 | 468.9 | 488.0 | 151.6 | [local](../records/ffn-post-filter-1800-s461-post_filter.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `54f450449db8`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | two_hop | 4.590315 | 21.87% | 8,654,208 / 4,718,592 | 456.9 | 474.0 | 126.1 | [local](../records/ffn-two-hop-filter-1800-s461-two_hop.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `57bbe07938e8`

4 layers; width 384; 256 raw tokens.
Seed 197; 6,025,451 identical scored targets; schedule `6042d2050765`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | bottleneck_squared | 4.604657 | 21.56% | 8,648,064 / 4,712,448 | 445.1 | 474.0 | 177.2 | [local](../records/ffn-bottleneck-1800-s197-bottleneck_squared.json) |
| 2 | relu_squared | 4.605027 | 21.63% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 65.0 | [local](../records/ffn-bottleneck-1800-s197-relu_squared.json) |
| 3 | bottleneck_silu | 4.608432 | 21.56% | 8,648,064 / 4,712,448 | 445.1 | 474.0 | 95.9 | [local](../records/ffn-bottleneck-1800-s197-bottleneck_silu.json) |
| 4 | bottleneck_linear | 4.608804 | 21.54% | 8,648,064 / 4,712,448 | 444.6 | 474.0 | 136.4 | [local](../records/ffn-bottleneck-1800-s197-bottleneck_linear.json) |
| 5 | shifted_relu_squared | 4.609789 | 21.51% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 146.7 | [local](../records/ffn-bottleneck-1800-s197-shifted_relu_squared.json) |
| 6 | dense | 4.655772 | 20.73% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 61.0 | [local](../records/ffn-bottleneck-1800-s197-dense.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `646b457a5f94`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | gatebank_m4 | 4.626929 | 21.54% | 8,651,136 / 4,715,520 | 425.3 | 452.0 | 128.9 | [local](../records/ffn-gatebank-1800-s461-gatebank_m4.json) |
| 2 | gatebank_m2 | 4.638090 | 21.40% | 8,654,208 / 4,718,592 | 429.3 | 450.0 | 125.8 | [local](../records/ffn-gatebank-1800-s461-gatebank_m2.json) |
| 3 | gatebank_silu_m4 | 4.644421 | 20.94% | 8,649,600 / 4,713,984 | 427.6 | 452.0 | 123.8 | [local](../records/ffn-gatebank-1800-s461-gatebank_silu_m4.json) |
| 4 | gatebank_m1 | 4.654855 | 21.25% | 8,654,208 / 4,718,592 | 432.4 | 454.0 | 111.8 | [local](../records/ffn-gatebank-1800-s461-gatebank_m1.json) |

### HuggingFaceFW/fineweb-edu / 10000 updates / `689cf5f6df86`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dual_plain_quarter | 3.821241 | 31.12% | 8,654,208 / 4,718,592 | 441.8 | 468.0 | 985.7 | [local](../records/ffn-dual-path-10000-s461-dual_plain_quarter.json) |
| 2 | dual_router_small | 3.833562 | 31.04% | 8,654,208 / 4,718,592 | 452.7 | 470.0 | 1142.3 | [local](../records/ffn-dual-path-10000-s461-dual_router_small.json) |
| 3 | dual_filter_quarter | 3.833604 | 31.08% | 8,654,208 / 4,718,592 | 448.9 | 478.0 | 1090.5 | [local](../records/ffn-dual-path-10000-s461-dual_filter_quarter.json) |
| 4 | dual_filter_half | 3.840192 | 30.97% | 8,654,208 / 4,718,592 | 450.8 | 474.0 | 1090.3 | [local](../records/ffn-dual-path-10000-s461-dual_filter_half.json) |
| 5 | dual_filter_small | 3.843426 | 30.87% | 8,654,208 / 4,718,592 | 451.5 | 468.0 | 1088.8 | [local](../records/ffn-dual-path-10000-s461-dual_filter_small.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `6e867a7fa7bd`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | matrix_chain | 4.632481 | 21.60% | 8,654,208 / 4,718,592 | 424.4 | 452.0 | 177.7 | [local](../records/ffn-matrix-chain-1800-s461-matrix_chain.json) |
| 2 | matrix_parallel | 4.645310 | 21.18% | 8,654,208 / 4,718,592 | 423.3 | 452.0 | 177.2 | [local](../records/ffn-matrix-chain-1800-s461-matrix_parallel.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `77a6452f8504`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | pair_normalized | 4.594075 | 21.75% | 8,654,208 / 4,718,592 | 433.0 | 482.0 | 185.6 | [local](../records/ffn-pair-1800-s461-pair_normalized.json) |
| 2 | pair_raw | 4.594245 | 21.76% | 8,654,208 / 4,718,592 | 444.4 | 468.0 | 161.7 | [local](../records/ffn-pair-1800-s461-pair_raw.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `7ad9fed670d6`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | axis_parallel | 4.572839 | 22.19% | 8,654,208 / 4,718,592 | 460.0 | 492.0 | 166.0 | [local](../records/ffn-two-axis-1800-s461-axis_parallel.json) |

### HuggingFaceFW/fineweb-edu / 600 updates / `7d54fb13c977`

4 layers; width 384; 256 raw tokens.
Seed 17; 2,005,324 identical scored targets; schedule `2e14b0261112`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | self_gate_fixed | 5.281015 | 15.58% | 8,654,208 / 4,718,592 | 447.8 | 468.0 | 38.4 | [local](../records/ffn-single-600-v1-self_gate_fixed.json) |
| 2 | self_gate_learned | 5.283186 | 15.55% | 8,654,176 / 4,718,560 | 469.9 | 508.0 | 50.1 | [local](../records/ffn-single-600-v1-self_gate_learned.json) |
| 3 | dense | 5.298029 | 15.52% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 37.6 | [local](../records/ffn-single-600-v1-dense.json) |
| 4 | relu_squared | 5.302391 | 15.87% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 47.2 | [local](../records/ffn-single-600-v1-relu_squared.json) |
| 5 | shifted_relu_squared | 5.303146 | 15.88% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 48.5 | [local](../records/ffn-single-600-v1-shifted_relu_squared.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `a69710c97156`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | divisive_global | 4.599601 | 21.74% | 8,654,208 / 4,718,592 | 445.4 | 470.0 | 150.4 | [local](../records/ffn-divisive-1800-s461-divisive_global.json) |
| 2 | divisive_16 | 4.633002 | 21.37% | 8,654,208 / 4,718,592 | 452.3 | 470.0 | 154.0 | [local](../records/ffn-divisive-1800-s461-divisive_16.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `a9605ff6ed7f`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | distributed_fixed | 4.658676 | 20.79% | 8,654,208 / 4,718,592 | 471.4 | 492.0 | 136.1 | [local](../records/ffn-distributed-curve-1800-s461-distributed_fixed.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `ad4dbc63dd18`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | phase_rotation | 4.584918 | 22.05% | 8,654,208 / 4,718,592 | 460.9 | 474.0 | 169.6 | [local](../records/ffn-phase-1800-s461-phase_rotation.json) |
| 2 | phase_diagonal | 4.586413 | 21.94% | 8,654,208 / 4,718,592 | 454.1 | 474.0 | 155.2 | [local](../records/ffn-phase-1800-s461-phase_diagonal.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `add6c6661f55`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | axis_parallel | 4.589981 | 21.82% | 8,654,208 / 4,718,592 | 464.4 | 492.0 | 163.0 | [local](../records/ffn-two-axis-confirm-1800-s311-axis_parallel.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `b0c31c89b710`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.595735 | 21.88% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 192.9 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 4.603346 | 21.60% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 102.5 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 3 | relu_squared | 4.607443 | 21.61% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 83.5 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 4 | dense | 4.616679 | 21.54% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 87.6 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 5 | product_dictionary | 4.692148 | 20.16% | 8,654,208 / 4,718,592 | 421.8 | 452.0 | 157.0 | [local](../records/ffn-dictionary-1800-s461-product_dictionary.json) |
| 6 | flat_dictionary | 4.743684 | 19.33% | 8,654,208 / 4,718,592 | 448.4 | 468.0 | 149.9 | [local](../records/ffn-dictionary-1800-s461-flat_dictionary.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `b4a47704b9dd`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.594346 | 21.96% | 8,654,208 / 4,718,592 | 459.4 | 482.0 | 216.5 | [local](../records/ffn-ridge-confirm-1800-s311-ridge_1.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `b7767ee6f51b`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.596793 | 21.90% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 173.5 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 4.611920 | 21.40% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 132.4 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 3 | relu_squared | 4.617303 | 21.28% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 111.8 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 4 | dense | 4.645015 | 21.00% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 110.1 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `b7bdbd67e33f`

4 layers; width 384; 256 raw tokens.
Seed 277; 6,030,702 identical scored targets; schedule `45bdbb23ef3f`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.587031 | 21.94% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 161.5 | [local](../records/ffn-hybrid-1800-s277-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 4.592627 | 21.76% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 124.6 | [local](../records/ffn-hybrid-1800-s277-shifted_relu_squared.json) |
| 3 | relu_squared | 4.597271 | 21.58% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 122.2 | [local](../records/ffn-hybrid-1800-s277-relu_squared.json) |
| 4 | hybrid_local | 4.597547 | 21.64% | 8,654,208 / 4,718,592 | 448.7 | 472.0 | 148.6 | [local](../records/ffn-hybrid-1800-s277-hybrid_local.json) |
| 5 | dense | 4.646887 | 20.99% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 130.9 | [local](../records/ffn-hybrid-1800-s277-dense.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `bf45cc05490d`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 4.594728 | 21.61% | 8,654,208 / 4,718,592 | 448.0 | 468.0 | 170.9 | [local](../records/ffn-branch-1800-s461-branch_sigmoid.json) |
| 2 | branch_softmax | 4.607250 | 21.54% | 8,654,208 / 4,718,592 | 445.8 | 468.0 | 168.5 | [local](../records/ffn-branch-1800-s461-branch_softmax.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `dae0d7e49b16`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.596409 | 21.75% | 8,654,208 / 4,718,592 | 446.9 | 468.0 | 147.3 | [local](../records/ffn-dynamic-filter-confirm-1800-s293-dynamic_filter.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `dd25e30aa846`

4 layers; width 384; 256 raw tokens.
Seed 293; 6,031,843 identical scored targets; schedule `86a09f952724`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | pair_normalized | 4.601568 | 21.62% | 8,654,208 / 4,718,592 | 430.8 | 462.0 | 185.4 | [local](../records/ffn-pair-confirm-1800-s293-pair_normalized.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `e06a4b34c685`

4 layers; width 384; 256 raw tokens.
Seed 359; 6,036,078 identical scored targets; schedule `8b1b92893968`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.601724 | 21.69% | 8,654,208 / 4,718,592 | 453.6 | 474.0 | 171.0 | [local](../records/ffn-linear-feature-1800-s359-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 4.617942 | 21.24% | 8,654,200 / 4,718,584 | 421.4 | 448.0 | 90.8 | [local](../records/ffn-linear-feature-1800-s359-shifted_relu_squared.json) |
| 3 | relu_squared | 4.624365 | 21.25% | 8,654,208 / 4,718,592 | 418.6 | 448.0 | 119.7 | [local](../records/ffn-linear-feature-1800-s359-relu_squared.json) |
| 4 | feature_linear_squared | 4.635795 | 21.16% | 8,654,208 / 4,718,592 | 460.6 | 478.0 | 180.2 | [local](../records/ffn-linear-feature-1800-s359-feature_linear_squared.json) |
| 5 | dense | 4.642909 | 20.99% | 8,654,208 / 4,718,592 | 435.6 | 454.0 | 123.8 | [local](../records/ffn-linear-feature-1800-s359-dense.json) |
| 6 | feature_linear_elu | 4.676605 | 20.32% | 8,654,208 / 4,718,592 | 460.6 | 478.0 | 191.8 | [local](../records/ffn-linear-feature-1800-s359-feature_linear_elu.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `e3d09b38e3ef`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | static_wide | 4.618059 | 21.42% | 8,655,232 / 4,719,616 | 447.6 | 468.0 | 146.5 | [local](../records/ffn-static-filter-1800-s461-static_wide.json) |
| 2 | static_narrow | 4.626710 | 21.30% | 8,262,016 / 4,326,400 | 438.9 | 468.0 | 151.2 | [local](../records/ffn-static-filter-1800-s461-static_narrow.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `e44975432e18`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | maxout_2 | 4.713439 | 20.22% | 8,654,208 / 4,718,592 | 453.6 | 478.0 | 141.8 | [local](../records/ffn-competitive-pool-1800-s461-maxout_2.json) |
| 2 | maxout_3 | 4.730428 | 20.08% | 8,654,208 / 4,718,592 | 433.7 | 478.0 | 136.3 | [local](../records/ffn-competitive-pool-1800-s461-maxout_3.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `e6da8cfc27eb`

4 layers; width 384; 256 raw tokens.
Seed 311; 6,017,388 identical scored targets; schedule `ecf60b3e4ff8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | pair_normalized | 4.603144 | 21.58% | 8,654,208 / 4,718,592 | 435.6 | 462.0 | 183.1 | [local](../records/ffn-pair-confirm-1800-s311-pair_normalized.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `e99cbd32b0f3`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | self_correlation | 4.605401 | 21.59% | 8,654,208 / 4,718,592 | 454.1 | 476.0 | 167.2 | [local](../records/ffn-correlation-1800-s461-self_correlation.json) |
| 2 | correlation_diagonal | 4.606093 | 21.62% | 8,654,208 / 4,718,592 | 453.4 | 476.0 | 167.5 | [local](../records/ffn-correlation-1800-s461-correlation_diagonal.json) |

### HuggingFaceFW/fineweb-edu / 10000 updates / `ecbf85ac7385`

4 layers; width 384; 256 raw tokens.
Seed 461; 33,440,772 identical scored targets; schedule `82811a483a8c`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | gate16 | 3.826844 | 31.14% | 8,654,208 / 4,718,592 | 446.9 | 472.0 | 1365.8 | [local](../records/ffn-evolution-10000-s461-gate16.json) |
| 2 | gate_sine | 3.827993 | 31.08% | 8,654,208 / 4,718,592 | 448.0 | 468.0 | 1034.4 | [local](../records/ffn-evolution-10000-s461-gate_sine.json) |
| 3 | partial_rational | 3.829353 | 31.07% | 8,654,208 / 4,718,592 | 457.5 | 484.0 | 1466.2 | [local](../records/ffn-evolution-10000-s461-partial_rational.json) |
| 4 | partial_positive_sine | 3.830455 | 31.16% | 8,654,208 / 4,718,592 | 457.5 | 484.0 | 1329.8 | [local](../records/ffn-evolution-10000-s461-partial_positive_sine.json) |
| 5 | partial_sine | 3.833127 | 31.07% | 8,654,208 / 4,718,592 | 457.5 | 482.0 | 1435.7 | [local](../records/ffn-evolution-10000-s461-partial_sine.json) |
| 6 | gate8 | 3.837566 | 30.96% | 8,654,208 / 4,718,592 | 446.6 | 472.0 | 1104.4 | [local](../records/ffn-evolution-10000-s461-gate8.json) |
| 7 | filter_updategate | 3.855924 | 30.80% | 8,654,208 / 4,718,592 | 472.3 | 508.0 | 1141.6 | [local](../records/ffn-evolution-10000-s461-filter_updategate.json) |
| 8 | filter_postgate | 3.865842 | 30.54% | 8,654,208 / 4,718,592 | 469.7 | 508.0 | 1132.1 | [local](../records/ffn-evolution-10000-s461-filter_postgate.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `f08c102ce325`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 22.02% | 8,654,208 / 4,718,592 | 446.9 | 468.0 | 155.2 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `f2b9a6856be2`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | filter_bank4 | 4.586387 | 21.96% | 8,654,208 / 4,718,592 | 449.6 | 472.0 | 153.2 | [local](../records/ffn-filter-bank-1800-s461-filter_bank4.json) |
| 2 | filter_bank16 | 4.587389 | 21.91% | 8,654,208 / 4,718,592 | 449.6 | 472.0 | 157.8 | [local](../records/ffn-filter-bank-1800-s461-filter_bank16.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `f4c86eb83f20`

6 layers; width 512; 256 raw tokens.
Seed 71; 6,009,808 identical scored targets; schedule `9cecb6d433f5`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | shifted_relu_squared | 4.476021 | 22.56% | 21,076,564 / 12,681,300 | 786.8 | 862.0 | 176.6 | [local](../records/ffn-single-scale-1800-s71-shifted_relu_squared.json) |
| 2 | dense | 4.481850 | 22.52% | 21,076,480 / 12,681,216 | 824.1 | 848.0 | 178.3 | [local](../records/ffn-single-scale-1800-s71-dense.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `f4d461ea7d29`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | jet_factored | 4.613338 | 21.51% | 8,655,744 / 4,720,128 | 460.1 | 480.0 | 213.8 | [local](../records/ffn-jet-factored-1800-s461-jet_factored.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `f9e40f874579`

6 layers; width 512; 256 raw tokens.
Seed 71; 6,009,808 identical scored targets; schedule `9cecb6d433f5`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_transfer | 4.442687 | 23.29% | 21,076,480 / 12,681,216 | 850.4 | 878.0 | 278.9 | [local](../records/ffn-hybrid-transfer-1800-s71-hybrid_transfer.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `fa6c0482e1a4`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | tensor_banks | 4.631435 | 21.15% | 8,654,208 / 4,718,592 | 442.5 | 474.0 | 106.4 | [local](../records/ffn-tensor-response-1800-s461-tensor_banks.json) |
| 2 | tensor_full | 4.664352 | 20.83% | 8,654,208 / 4,718,592 | 421.6 | 474.0 | 140.4 | [local](../records/ffn-tensor-response-1800-s461-tensor_full.json) |

### HuggingFaceFW/fineweb-edu / 1800 updates / `feb84f4598a1`

4 layers; width 384; 256 raw tokens.
Seed 461; 6,028,850 identical scored targets; schedule `9bb297e2f1d8`.

| Rank | FFN | Full validation NLL | Token accuracy | Total / FFN params | Peak allocated MiB | Peak reserved MiB | Update seconds | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | jet_split | 4.632504 | 21.29% | 8,654,208 / 4,718,592 | 430.3 | 450.0 | 160.3 | [local](../records/ffn-jet-1800-s461-jet_split.json) |


## Step 1: audited reuse of saved controls

These tables reference existing runs; reused rows are not additional training or independent replications.

### Two-hop filter / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-two-hop-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | wide_conv | 4.588174 | 8,654,208 / 4,718,592 | 449.37 | 472.00 | [local](../records/ffn-wide-convolution-1800-s461-wide_conv.json) |
| 3 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 4 | two_hop | 4.590315 | 8,654,208 / 4,718,592 | 456.88 | 474.00 | [local](../records/ffn-two-hop-filter-1800-s461-two_hop.json) |
| 5 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 6 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 7 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 9 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Two-hop filter / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-two-hop-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | wide_conv | 3.343073 | 8,654,208 / 4,718,592 | 449.37 | 472.00 | [local](../records/ffn-wide-convolution-1800-s461-wide_conv.json) |
| 2 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 3 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 4 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 5 | two_hop | 3.356606 | 8,654,208 / 4,718,592 | 456.88 | 474.00 | [local](../records/ffn-two-hop-filter-1800-s461-two_hop.json) |
| 6 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 7 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 9 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Tensor responses / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-tensor-response-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | wide_conv | 4.588174 | 8,654,208 / 4,718,592 | 449.37 | 472.00 | [local](../records/ffn-wide-convolution-1800-s461-wide_conv.json) |
| 3 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 4 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 5 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 9 | tensor_banks | 4.631435 | 8,654,208 / 4,718,592 | 442.53 | 474.00 | [local](../records/ffn-tensor-response-1800-s461-tensor_banks.json) |
| 10 | tensor_full | 4.664352 | 8,654,208 / 4,718,592 | 421.59 | 474.00 | [local](../records/ffn-tensor-response-1800-s461-tensor_full.json) |

### Tensor responses / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-tensor-response-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | wide_conv | 3.343073 | 8,654,208 / 4,718,592 | 449.37 | 472.00 | [local](../records/ffn-wide-convolution-1800-s461-wide_conv.json) |
| 2 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 3 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 4 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 5 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 9 | tensor_banks | 3.389340 | 8,654,208 / 4,718,592 | 442.53 | 474.00 | [local](../records/ffn-tensor-response-1800-s461-tensor_banks.json) |
| 10 | tensor_full | 3.413476 | 8,654,208 / 4,718,592 | 421.59 | 474.00 | [local](../records/ffn-tensor-response-1800-s461-tensor_full.json) |

### Wide generated convolution / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-wide-convolution-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | wide_conv | 4.588174 | 8,654,208 / 4,718,592 | 449.37 | 472.00 | [local](../records/ffn-wide-convolution-1800-s461-wide_conv.json) |
| 3 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 4 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 5 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Wide generated convolution / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-wide-convolution-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | wide_conv | 3.343073 | 8,654,208 / 4,718,592 | 449.37 | 472.00 | [local](../records/ffn-wide-convolution-1800-s461-wide_conv.json) |
| 2 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 3 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 4 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 5 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Energy contrast / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-energy-contrast-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 8 | contrast32 | 4.662402 | 8,654,208 / 4,718,592 | 422.40 | 448.00 | [local](../records/ffn-energy-contrast-1800-s461-contrast32.json) |
| 9 | contrast128 | 4.667947 | 8,654,208 / 4,718,592 | 422.19 | 448.00 | [local](../records/ffn-energy-contrast-1800-s461-contrast128.json) |

### Energy contrast / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-energy-contrast-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 8 | contrast32 | 3.441675 | 8,654,208 / 4,718,592 | 422.40 | 448.00 | [local](../records/ffn-energy-contrast-1800-s461-contrast32.json) |
| 9 | contrast128 | 3.447880 | 8,654,208 / 4,718,592 | 422.19 | 448.00 | [local](../records/ffn-energy-contrast-1800-s461-contrast128.json) |

### Parallel two-axis generated filter / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-two-axis-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | axis_parallel | 4.572839 | 8,654,208 / 4,718,592 | 460.00 | 492.00 | [local](../records/ffn-two-axis-1800-s461-axis_parallel.json) |
| 2 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 3 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 4 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 5 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Parallel two-axis generated filter / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-two-axis-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | axis_parallel | 3.338122 | 8,654,208 / 4,718,592 | 460.00 | 492.00 | [local](../records/ffn-two-axis-1800-s461-axis_parallel.json) |
| 2 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 3 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 4 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 5 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Nonlinear generated coefficients / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-nonlinear-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | filter_tanh | 4.588307 | 8,654,208 / 4,718,592 | 452.41 | 474.00 | [local](../records/ffn-nonlinear-filter-1800-s461-filter_tanh.json) |
| 3 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 4 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 5 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | filter_signed_square | 4.602801 | 8,654,208 / 4,718,592 | 456.88 | 474.00 | [local](../records/ffn-nonlinear-filter-1800-s461-filter_signed_square.json) |
| 7 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 9 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Nonlinear generated coefficients / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-nonlinear-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | filter_tanh | 3.354039 | 8,654,208 / 4,718,592 | 452.41 | 474.00 | [local](../records/ffn-nonlinear-filter-1800-s461-filter_tanh.json) |
| 4 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 5 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | filter_signed_square | 3.370427 | 8,654,208 / 4,718,592 | 456.88 | 474.00 | [local](../records/ffn-nonlinear-filter-1800-s461-filter_signed_square.json) |
| 8 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 9 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Post-activation generated filter / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-post-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | post_filter | 4.614462 | 8,654,208 / 4,718,592 | 468.88 | 488.00 | [local](../records/ffn-post-filter-1800-s461-post_filter.json) |
| 8 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Post-activation generated filter / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-post-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | post_filter | 3.374329 | 8,654,208 / 4,718,592 | 468.88 | 488.00 | [local](../records/ffn-post-filter-1800-s461-post_filter.json) |
| 8 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Multiple generated filters / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-filter-bank-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | filter_bank4 | 4.586387 | 8,654,208 / 4,718,592 | 449.63 | 472.00 | [local](../records/ffn-filter-bank-1800-s461-filter_bank4.json) |
| 3 | filter_bank16 | 4.587389 | 8,654,208 / 4,718,592 | 449.60 | 472.00 | [local](../records/ffn-filter-bank-1800-s461-filter_bank16.json) |
| 4 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 5 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 6 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 7 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 9 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Multiple generated filters / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-filter-bank-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | filter_bank16 | 3.347661 | 8,654,208 / 4,718,592 | 449.60 | 472.00 | [local](../records/ffn-filter-bank-1800-s461-filter_bank16.json) |
| 3 | filter_bank4 | 3.348846 | 8,654,208 / 4,718,592 | 449.63 | 472.00 | [local](../records/ffn-filter-bank-1800-s461-filter_bank4.json) |
| 4 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 5 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 6 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 7 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 9 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Competitive pooling / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-competitive-pool-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 8 | maxout_2 | 4.713439 | 8,654,208 / 4,718,592 | 453.63 | 478.00 | [local](../records/ffn-competitive-pool-1800-s461-maxout_2.json) |
| 9 | maxout_3 | 4.730428 | 8,654,208 / 4,718,592 | 433.69 | 478.00 | [local](../records/ffn-competitive-pool-1800-s461-maxout_3.json) |

### Competitive pooling / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-competitive-pool-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 8 | maxout_2 | 3.503124 | 8,654,208 / 4,718,592 | 453.63 | 478.00 | [local](../records/ffn-competitive-pool-1800-s461-maxout_2.json) |
| 9 | maxout_3 | 3.514838 | 8,654,208 / 4,718,592 | 433.69 | 478.00 | [local](../records/ffn-competitive-pool-1800-s461-maxout_3.json) |

### Full-width self-correlation / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-correlation-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | self_correlation | 4.605401 | 8,654,208 / 4,718,592 | 454.08 | 476.00 | [local](../records/ffn-correlation-1800-s461-self_correlation.json) |
| 7 | correlation_diagonal | 4.606093 | 8,654,208 / 4,718,592 | 453.39 | 476.00 | [local](../records/ffn-correlation-1800-s461-correlation_diagonal.json) |
| 8 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 9 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Full-width self-correlation / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-correlation-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | self_correlation | 3.364279 | 8,654,208 / 4,718,592 | 454.08 | 476.00 | [local](../records/ffn-correlation-1800-s461-self_correlation.json) |
| 6 | correlation_diagonal | 3.367135 | 8,654,208 / 4,718,592 | 453.39 | 476.00 | [local](../records/ffn-correlation-1800-s461-correlation_diagonal.json) |
| 7 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 9 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Dense nonlinear branch routing / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-branch-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | branch_sigmoid | 4.594728 | 8,654,208 / 4,718,592 | 447.97 | 468.00 | [local](../records/ffn-branch-1800-s461-branch_sigmoid.json) |
| 5 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | branch_softmax | 4.607250 | 8,654,208 / 4,718,592 | 445.84 | 468.00 | [local](../records/ffn-branch-1800-s461-branch_softmax.json) |
| 8 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 9 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Dense nonlinear branch routing / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-branch-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | branch_sigmoid | 3.359723 | 8,654,208 / 4,718,592 | 447.97 | 468.00 | [local](../records/ffn-branch-1800-s461-branch_sigmoid.json) |
| 5 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | branch_softmax | 3.366048 | 8,654,208 / 4,718,592 | 445.84 | 468.00 | [local](../records/ffn-branch-1800-s461-branch_softmax.json) |
| 7 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 9 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Phase rotations / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-phase-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | phase_rotation | 4.584918 | 8,654,208 / 4,718,592 | 460.88 | 474.00 | [local](../records/ffn-phase-1800-s461-phase_rotation.json) |
| 3 | phase_diagonal | 4.586413 | 8,654,208 / 4,718,592 | 454.06 | 474.00 | [local](../records/ffn-phase-1800-s461-phase_diagonal.json) |
| 4 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 5 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 6 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 7 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 9 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Phase rotations / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-phase-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | phase_rotation | 3.345356 | 8,654,208 / 4,718,592 | 460.88 | 474.00 | [local](../records/ffn-phase-1800-s461-phase_rotation.json) |
| 3 | phase_diagonal | 3.348351 | 8,654,208 / 4,718,592 | 454.06 | 474.00 | [local](../records/ffn-phase-1800-s461-phase_diagonal.json) |
| 4 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 5 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 6 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 7 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 9 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Divisive feature competition / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-divisive-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | divisive_global | 4.599601 | 8,654,208 / 4,718,592 | 445.39 | 470.00 | [local](../records/ffn-divisive-1800-s461-divisive_global.json) |
| 6 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 9 | divisive_16 | 4.633002 | 8,654,208 / 4,718,592 | 452.25 | 470.00 | [local](../records/ffn-divisive-1800-s461-divisive_16.json) |

### Divisive feature competition / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-divisive-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | divisive_global | 3.359828 | 8,654,208 / 4,718,592 | 445.39 | 470.00 | [local](../records/ffn-divisive-1800-s461-divisive_global.json) |
| 5 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 9 | divisive_16 | 3.396719 | 8,654,208 / 4,718,592 | 452.25 | 470.00 | [local](../records/ffn-divisive-1800-s461-divisive_16.json) |

### Exterior and symmetric pairs / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-exterior-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | symmetric_pairs | 4.601977 | 8,654,208 / 4,718,592 | 430.12 | 454.00 | [local](../records/ffn-exterior-1800-s461-symmetric_pairs.json) |
| 6 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | exterior | 4.605759 | 8,654,208 / 4,718,592 | 429.62 | 454.00 | [local](../records/ffn-exterior-1800-s461-exterior.json) |
| 8 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 9 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Exterior and symmetric pairs / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-exterior-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | symmetric_pairs | 3.368154 | 8,654,208 / 4,718,592 | 430.12 | 454.00 | [local](../records/ffn-exterior-1800-s461-symmetric_pairs.json) |
| 6 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | exterior | 3.369273 | 8,654,208 / 4,718,592 | 429.62 | 454.00 | [local](../records/ffn-exterior-1800-s461-exterior.json) |
| 8 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 9 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Jet split / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-jet-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 3 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 4 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 5 | jet_split | 4.632504 | 8,654,208 / 4,718,592 | 430.28 | 450.00 | [local](../records/ffn-jet-1800-s461-jet_split.json) |

### Jet split / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-jet-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 3 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 4 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 5 | jet_split | 3.397434 | 8,654,208 / 4,718,592 | 430.28 | 450.00 | [local](../records/ffn-jet-1800-s461-jet_split.json) |

### Static filters (narrow: smaller matched-width control) / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-static-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 8 | static_wide | 4.618059 | 8,655,232 / 4,719,616 | 447.58 | 468.00 | [local](../records/ffn-static-filter-1800-s461-static_wide.json) |
| 9 | static_narrow | 4.626710 | 8,262,016 / 4,326,400 | 438.93 | 468.00 | [local](../records/ffn-static-filter-1800-s461-static_narrow.json) |

### Static filters (narrow: smaller matched-width control) / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-static-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 8 | static_wide | 3.376262 | 8,655,232 / 4,719,616 | 447.58 | 468.00 | [local](../records/ffn-static-filter-1800-s461-static_wide.json) |
| 9 | static_narrow | 3.391998 | 8,262,016 / 4,326,400 | 438.93 | 468.00 | [local](../records/ffn-static-filter-1800-s461-static_narrow.json) |

### Wide dynamic filter / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-dynamic-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.581171 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Wide dynamic filter / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-dynamic-filter-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.345057 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-1800-s461-dynamic_filter.json) |
| 2 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Four-stage folding / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-fold-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 5 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 7 | fold4 | 4.855533 | 8,654,208 / 4,718,592 | 451.38 | 488.00 | [local](../records/ffn-fold-1800-s461-fold4.json) |

### Four-stage folding / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-fold-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 5 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 7 | fold4 | 3.658606 | 8,654,208 / 4,718,592 | 451.38 | 488.00 | [local](../records/ffn-fold-1800-s461-fold4.json) |

### Conditional matrix chain / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-matrix-chain-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 5 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 7 | matrix_chain | 4.632481 | 8,654,208 / 4,718,592 | 424.38 | 452.00 | [local](../records/ffn-matrix-chain-1800-s461-matrix_chain.json) |
| 8 | matrix_parallel | 4.645310 | 8,654,208 / 4,718,592 | 423.25 | 452.00 | [local](../records/ffn-matrix-chain-1800-s461-matrix_parallel.json) |

### Conditional matrix chain / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-matrix-chain-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 5 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 7 | matrix_chain | 3.407659 | 8,654,208 / 4,718,592 | 424.38 | 452.00 | [local](../records/ffn-matrix-chain-1800-s461-matrix_chain.json) |
| 8 | matrix_parallel | 3.417661 | 8,654,208 / 4,718,592 | 423.25 | 452.00 | [local](../records/ffn-matrix-chain-1800-s461-matrix_parallel.json) |

### Explicit pair scores / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-pair-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | pair_normalized | 4.594075 | 8,654,208 / 4,718,592 | 433.00 | 482.00 | [local](../records/ffn-pair-1800-s461-pair_normalized.json) |
| 4 | pair_raw | 4.594245 | 8,654,208 / 4,718,592 | 444.38 | 468.00 | [local](../records/ffn-pair-1800-s461-pair_raw.json) |
| 5 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 8 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Explicit pair scores / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-pair-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | pair_normalized | 3.353935 | 8,654,208 / 4,718,592 | 433.00 | 482.00 | [local](../records/ffn-pair-1800-s461-pair_normalized.json) |
| 3 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 4 | pair_raw | 3.355598 | 8,654,208 / 4,718,592 | 444.38 | 468.00 | [local](../records/ffn-pair-1800-s461-pair_raw.json) |
| 5 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 6 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Jet geometry / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-jet-factored-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 3 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 4 | jet_factored | 4.613338 | 8,655,744 / 4,720,128 | 460.14 | 480.00 | [local](../records/ffn-jet-factored-1800-s461-jet_factored.json) |
| 5 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 6 | jet_split | 4.632504 | 8,654,208 / 4,718,592 | 430.28 | 450.00 | [local](../records/ffn-jet-1800-s461-jet_split.json) |

### Jet geometry / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-jet-factored-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 3 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 4 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 5 | jet_factored | 3.376662 | 8,655,744 / 4,720,128 | 460.14 | 480.00 | [local](../records/ffn-jet-factored-1800-s461-jet_factored.json) |
| 6 | jet_split | 3.397434 | 8,654,208 / 4,718,592 | 430.28 | 450.00 | [local](../records/ffn-jet-1800-s461-jet_split.json) |

### GateBank / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-gatebank-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 3 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 4 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 5 | gatebank_m4 | 4.626929 | 8,651,136 / 4,715,520 | 425.27 | 452.00 | [local](../records/ffn-gatebank-1800-s461-gatebank_m4.json) |
| 6 | gatebank_m2 | 4.638090 | 8,654,208 / 4,718,592 | 429.31 | 450.00 | [local](../records/ffn-gatebank-1800-s461-gatebank_m2.json) |
| 7 | gatebank_silu_m4 | 4.644421 | 8,649,600 / 4,713,984 | 427.56 | 452.00 | [local](../records/ffn-gatebank-1800-s461-gatebank_silu_m4.json) |
| 8 | gatebank_m1 | 4.654855 | 8,654,208 / 4,718,592 | 432.44 | 454.00 | [local](../records/ffn-gatebank-1800-s461-gatebank_m1.json) |

### GateBank / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-gatebank-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 3 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 4 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 5 | gatebank_m4 | 3.386655 | 8,651,136 / 4,715,520 | 425.27 | 452.00 | [local](../records/ffn-gatebank-1800-s461-gatebank_m4.json) |
| 6 | gatebank_m2 | 3.393742 | 8,654,208 / 4,718,592 | 429.31 | 450.00 | [local](../records/ffn-gatebank-1800-s461-gatebank_m2.json) |
| 7 | gatebank_silu_m4 | 3.397856 | 8,649,600 / 4,713,984 | 427.56 | 452.00 | [local](../records/ffn-gatebank-1800-s461-gatebank_silu_m4.json) |
| 8 | gatebank_m1 | 3.411986 | 8,654,208 / 4,718,592 | 432.44 | 454.00 | [local](../records/ffn-gatebank-1800-s461-gatebank_m1.json) |

### Distributed curves / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-distributed-curve-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 3 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 4 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 5 | distributed_fixed | 4.658676 | 8,654,208 / 4,718,592 | 471.40 | 492.00 | [local](../records/ffn-distributed-curve-1800-s461-distributed_fixed.json) |

### Distributed curves / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-distributed-curve-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 2 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 3 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 4 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 5 | distributed_fixed | 3.428987 | 8,654,208 / 4,718,592 | 471.40 | 492.00 | [local](../records/ffn-distributed-curve-1800-s461-distributed_fixed.json) |

### Ridge association / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-ridge-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 3 | ridge_01 | 4.603095 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_01.json) |
| 4 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 5 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Ridge association / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-ridge-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 3 | ridge_01 | 3.362974 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_01.json) |
| 4 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 5 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Circular convolution / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-convolution-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 5 | convolution_squared | 4.606602 | 8,654,208 / 4,718,592 | 452.66 | 470.00 | [local](../records/ffn-convolution-1800-s461-convolution_squared.json) |
| 6 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Circular convolution / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-convolution-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 5 | convolution_squared | 3.370874 | 8,654,208 / 4,718,592 | 452.66 | 470.00 | [local](../records/ffn-convolution-1800-s461-convolution_squared.json) |
| 6 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Competitive coding / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-competitive-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 5 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | competitive_diagonal | 4.608517 | 8,654,208 / 4,718,592 | 455.32 | 470.00 | [local](../records/ffn-competitive-1800-s461-competitive_diagonal.json) |
| 7 | competitive_256 | 4.609286 | 8,654,208 / 4,718,592 | 455.32 | 470.00 | [local](../records/ffn-competitive-1800-s461-competitive_256.json) |
| 8 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Competitive coding / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-competitive-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 5 | competitive_diagonal | 3.369100 | 8,654,208 / 4,718,592 | 455.32 | 470.00 | [local](../records/ffn-competitive-1800-s461-competitive_diagonal.json) |
| 6 | competitive_256 | 3.370112 | 8,654,208 / 4,718,592 | 455.32 | 470.00 | [local](../records/ffn-competitive-1800-s461-competitive_256.json) |
| 7 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 8 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Reflective mixing / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-reflective-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | reflective | 4.598324 | 8,654,208 / 4,718,592 | 442.00 | 460.00 | [local](../records/ffn-reflective-1800-s461-reflective.json) |
| 5 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 7 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Reflective mixing / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-reflective-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | reflective | 3.358345 | 8,654,208 / 4,718,592 | 442.00 | 460.00 | [local](../records/ffn-reflective-1800-s461-reflective.json) |
| 4 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 5 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 7 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |

### Local lattice / saved controls / text

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-lattice-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.590279 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 4.591732 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 4.595735 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | shifted_relu_squared | 4.603346 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 5 | relu_squared | 4.607443 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 6 | dense | 4.616679 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 7 | lattice | 4.634058 | 8,653,184 / 4,717,568 | 465.09 | 492.00 | [local](../records/ffn-lattice-1800-s461-lattice.json) |

### Local lattice / saved controls / chat

Seed 461; 1,800 updates; 6,028,850 matched targets. [Audit](../records/ffn-lattice-1800-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.352524 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-1800-s461-ridge_1.json) |
| 2 | convolution_raw | 3.354088 | 8,654,208 / 4,718,592 | 448.66 | 468.00 | [local](../records/ffn-convolution-1800-s461-convolution_raw.json) |
| 3 | hybrid_normalized | 3.362123 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-dictionary-1800-s461-hybrid_normalized.json) |
| 4 | relu_squared | 3.368997 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-dictionary-1800-s461-relu_squared.json) |
| 5 | shifted_relu_squared | 3.374200 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-dictionary-1800-s461-shifted_relu_squared.json) |
| 6 | dense | 3.375814 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-dictionary-1800-s461-dense.json) |
| 7 | lattice | 3.399442 | 8,653,184 / 4,717,568 | 465.09 | 492.00 | [local](../records/ffn-lattice-1800-s461-lattice.json) |

### Ridge confirmation seed293 / saved controls / text

Seed 293; 1,800 updates; 6,031,843 matched targets. [Audit](../records/ffn-ridge-confirm-1800-s293-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 4.591870 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-confirm-1800-s293-ridge_1.json) |
| 2 | hybrid_normalized | 4.596793 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 3 | shifted_relu_squared | 4.611920 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 4 | relu_squared | 4.617303 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 5 | dense | 4.645015 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### Ridge confirmation seed293 / saved controls / chat

Seed 293; 1,800 updates; 6,031,843 matched targets. [Audit](../records/ffn-ridge-confirm-1800-s293-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.344238 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-confirm-1800-s293-ridge_1.json) |
| 2 | hybrid_normalized | 3.351981 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 3 | shifted_relu_squared | 3.371116 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 4 | relu_squared | 3.375245 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 5 | dense | 3.387184 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### Ridge confirmation seed311 / saved controls / text

Seed 311; 1,800 updates; 6,017,388 matched targets. [Audit](../records/ffn-ridge-confirm-1800-s311-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.594062 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 2 | ridge_1 | 4.594346 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-confirm-1800-s311-ridge_1.json) |
| 3 | relu_squared | 4.606856 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 4 | shifted_relu_squared | 4.610709 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 5 | dense | 4.628676 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### Ridge confirmation seed311 / saved controls / chat

Seed 311; 1,800 updates; 6,017,388 matched targets. [Audit](../records/ffn-ridge-confirm-1800-s311-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | ridge_1 | 3.339692 | 8,654,208 / 4,718,592 | 459.38 | 482.00 | [local](../records/ffn-ridge-confirm-1800-s311-ridge_1.json) |
| 2 | hybrid_normalized | 3.340158 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 3 | relu_squared | 3.357382 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 4 | shifted_relu_squared | 3.361548 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 5 | dense | 3.368282 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### Two-axis confirmation seed293 / saved controls / text

Seed 293; 1,800 updates; 6,031,843 matched targets. [Audit](../records/ffn-two-axis-confirm-1800-s293-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.596409 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-confirm-1800-s293-dynamic_filter.json) |
| 2 | hybrid_normalized | 4.596793 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 3 | axis_parallel | 4.600753 | 8,654,208 / 4,718,592 | 460.00 | 492.00 | [local](../records/ffn-two-axis-confirm-1800-s293-axis_parallel.json) |
| 4 | shifted_relu_squared | 4.611920 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 5 | relu_squared | 4.617303 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 6 | dense | 4.645015 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### Two-axis confirmation seed293 / saved controls / chat

Seed 293; 1,800 updates; 6,031,843 matched targets. [Audit](../records/ffn-two-axis-confirm-1800-s293-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.347085 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-confirm-1800-s293-dynamic_filter.json) |
| 2 | axis_parallel | 3.348746 | 8,654,208 / 4,718,592 | 460.00 | 492.00 | [local](../records/ffn-two-axis-confirm-1800-s293-axis_parallel.json) |
| 3 | hybrid_normalized | 3.351981 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 4 | shifted_relu_squared | 3.371116 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 5 | relu_squared | 3.375245 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 6 | dense | 3.387184 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### Two-axis confirmation seed311 / saved controls / text

Seed 311; 1,800 updates; 6,017,388 matched targets. [Audit](../records/ffn-two-axis-confirm-1800-s311-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.574550 | 8,654,208 / 4,718,592 | 447.38 | 468.00 | [local](../records/ffn-dynamic-filter-confirm-1800-s311-dynamic_filter.json) |
| 2 | axis_parallel | 4.589981 | 8,654,208 / 4,718,592 | 464.38 | 492.00 | [local](../records/ffn-two-axis-confirm-1800-s311-axis_parallel.json) |
| 3 | hybrid_normalized | 4.594062 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 4 | relu_squared | 4.606856 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 5 | shifted_relu_squared | 4.610709 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 6 | dense | 4.628676 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### Two-axis confirmation seed311 / saved controls / chat

Seed 311; 1,800 updates; 6,017,388 matched targets. [Audit](../records/ffn-two-axis-confirm-1800-s311-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.324885 | 8,654,208 / 4,718,592 | 447.38 | 468.00 | [local](../records/ffn-dynamic-filter-confirm-1800-s311-dynamic_filter.json) |
| 2 | axis_parallel | 3.328431 | 8,654,208 / 4,718,592 | 464.38 | 492.00 | [local](../records/ffn-two-axis-confirm-1800-s311-axis_parallel.json) |
| 3 | hybrid_normalized | 3.340158 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 4 | relu_squared | 3.357382 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 5 | shifted_relu_squared | 3.361548 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 6 | dense | 3.368282 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### Dynamic filter confirmation seed293 / saved controls / text

Seed 293; 1,800 updates; 6,031,843 matched targets. [Audit](../records/ffn-dynamic-filter-confirm-1800-s293-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.596409 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-confirm-1800-s293-dynamic_filter.json) |
| 2 | hybrid_normalized | 4.596793 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 3 | shifted_relu_squared | 4.611920 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 4 | relu_squared | 4.617303 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 5 | dense | 4.645015 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### Dynamic filter confirmation seed293 / saved controls / chat

Seed 293; 1,800 updates; 6,031,843 matched targets. [Audit](../records/ffn-dynamic-filter-confirm-1800-s293-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.347085 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-dynamic-filter-confirm-1800-s293-dynamic_filter.json) |
| 2 | hybrid_normalized | 3.351981 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 3 | shifted_relu_squared | 3.371116 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 4 | relu_squared | 3.375245 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 5 | dense | 3.387184 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### Dynamic filter confirmation seed311 / saved controls / text

Seed 311; 1,800 updates; 6,017,388 matched targets. [Audit](../records/ffn-dynamic-filter-confirm-1800-s311-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.574550 | 8,654,208 / 4,718,592 | 447.38 | 468.00 | [local](../records/ffn-dynamic-filter-confirm-1800-s311-dynamic_filter.json) |
| 2 | hybrid_normalized | 4.594062 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 3 | relu_squared | 4.606856 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 4 | shifted_relu_squared | 4.610709 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 5 | dense | 4.628676 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### Dynamic filter confirmation seed311 / saved controls / chat

Seed 311; 1,800 updates; 6,017,388 matched targets. [Audit](../records/ffn-dynamic-filter-confirm-1800-s311-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 3.324885 | 8,654,208 / 4,718,592 | 447.38 | 468.00 | [local](../records/ffn-dynamic-filter-confirm-1800-s311-dynamic_filter.json) |
| 2 | hybrid_normalized | 3.340158 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 3 | relu_squared | 3.357382 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 4 | shifted_relu_squared | 3.361548 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 5 | dense | 3.368282 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### Normalized pair confirmation seed293 / saved controls / text

Seed 293; 1,800 updates; 6,031,843 matched targets. [Audit](../records/ffn-pair-confirm-1800-s293-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.596793 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 2 | pair_normalized | 4.601568 | 8,654,208 / 4,718,592 | 430.81 | 462.00 | [local](../records/ffn-pair-confirm-1800-s293-pair_normalized.json) |
| 3 | shifted_relu_squared | 4.611920 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 4 | relu_squared | 4.617303 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 5 | dense | 4.645015 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### Normalized pair confirmation seed293 / saved controls / chat

Seed 293; 1,800 updates; 6,031,843 matched targets. [Audit](../records/ffn-pair-confirm-1800-s293-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.351981 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s293-hybrid_normalized.json) |
| 2 | pair_normalized | 3.353779 | 8,654,208 / 4,718,592 | 430.81 | 462.00 | [local](../records/ffn-pair-confirm-1800-s293-pair_normalized.json) |
| 3 | shifted_relu_squared | 3.371116 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-shifted_relu_squared.json) |
| 4 | relu_squared | 3.375245 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s293-relu_squared.json) |
| 5 | dense | 3.387184 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s293-dense.json) |

### Normalized pair confirmation seed311 / saved controls / text

Seed 311; 1,800 updates; 6,017,388 matched targets. [Audit](../records/ffn-pair-confirm-1800-s311-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 4.594062 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 2 | pair_normalized | 4.603144 | 8,654,208 / 4,718,592 | 435.63 | 462.00 | [local](../records/ffn-pair-confirm-1800-s311-pair_normalized.json) |
| 3 | relu_squared | 4.606856 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 4 | shifted_relu_squared | 4.610709 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 5 | dense | 4.628676 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### Normalized pair confirmation seed311 / saved controls / chat

Seed 311; 1,800 updates; 6,017,388 matched targets. [Audit](../records/ffn-pair-confirm-1800-s311-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_normalized | 3.340158 | 8,654,208 / 4,718,592 | 453.56 | 474.00 | [local](../records/ffn-hybrid-confirm-1800-s311-hybrid_normalized.json) |
| 2 | pair_normalized | 3.349054 | 8,654,208 / 4,718,592 | 435.63 | 462.00 | [local](../records/ffn-pair-confirm-1800-s311-pair_normalized.json) |
| 3 | relu_squared | 3.357382 | 8,654,208 / 4,718,592 | 418.56 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-relu_squared.json) |
| 4 | shifted_relu_squared | 3.361548 | 8,654,200 / 4,718,584 | 421.39 | 448.00 | [local](../records/ffn-hybrid-confirm-1800-s311-shifted_relu_squared.json) |
| 5 | dense | 3.368282 | 8,654,208 / 4,718,592 | 435.56 | 454.00 | [local](../records/ffn-hybrid-confirm-1800-s311-dense.json) |

### Larger hybrid transfer seed71 / saved controls / text

Seed 71; 1,800 updates; 6,009,808 matched targets. [Audit](../records/ffn-hybrid-transfer-1800-s71-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_transfer | 4.442687 | 21,076,480 / 12,681,216 | 850.37 | 878.00 | [local](../records/ffn-hybrid-transfer-1800-s71-hybrid_transfer.json) |
| 2 | shifted_relu_squared | 4.476021 | 21,076,564 / 12,681,300 | 786.75 | 862.00 | [local](../records/ffn-single-scale-1800-s71-shifted_relu_squared.json) |
| 3 | dense | 4.481850 | 21,076,480 / 12,681,216 | 824.06 | 848.00 | [local](../records/ffn-single-scale-1800-s71-dense.json) |

### Larger hybrid transfer seed71 / saved controls / chat

Seed 71; 1,800 updates; 6,009,808 matched targets. [Audit](../records/ffn-hybrid-transfer-1800-s71-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_transfer | 3.140375 | 21,076,480 / 12,681,216 | 850.37 | 878.00 | [local](../records/ffn-hybrid-transfer-1800-s71-hybrid_transfer.json) |
| 2 | dense | 3.169490 | 21,076,480 / 12,681,216 | 824.06 | 848.00 | [local](../records/ffn-single-scale-1800-s71-dense.json) |
| 3 | shifted_relu_squared | 3.182611 | 21,076,564 / 12,681,300 | 786.75 | 862.00 | [local](../records/ffn-single-scale-1800-s71-shifted_relu_squared.json) |

### Larger dynamic filter transfer seed71 / saved controls / text

Seed 71; 1,800 updates; 6,009,808 matched targets. [Audit](../records/ffn-dynamic-filter-transfer-1800-s71-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_transfer | 4.442687 | 21,076,480 / 12,681,216 | 850.37 | 878.00 | [local](../records/ffn-hybrid-transfer-1800-s71-hybrid_transfer.json) |
| 2 | dynamic_filter_transfer | 4.450644 | 21,076,480 / 12,681,216 | 833.68 | 852.00 | [local](../records/ffn-dynamic-filter-transfer-1800-s71-dynamic_filter_transfer.json) |
| 3 | shifted_relu_squared | 4.476021 | 21,076,564 / 12,681,300 | 786.75 | 862.00 | [local](../records/ffn-single-scale-1800-s71-shifted_relu_squared.json) |
| 4 | dense | 4.481850 | 21,076,480 / 12,681,216 | 824.06 | 848.00 | [local](../records/ffn-single-scale-1800-s71-dense.json) |

### Larger dynamic filter transfer seed71 / saved controls / chat

Seed 71; 1,800 updates; 6,009,808 matched targets. [Audit](../records/ffn-dynamic-filter-transfer-1800-s71-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | hybrid_transfer | 3.140375 | 21,076,480 / 12,681,216 | 850.37 | 878.00 | [local](../records/ffn-hybrid-transfer-1800-s71-hybrid_transfer.json) |
| 2 | dynamic_filter_transfer | 3.144704 | 21,076,480 / 12,681,216 | 833.68 | 852.00 | [local](../records/ffn-dynamic-filter-transfer-1800-s71-dynamic_filter_transfer.json) |
| 3 | dense | 3.169490 | 21,076,480 / 12,681,216 | 824.06 | 848.00 | [local](../records/ffn-single-scale-1800-s71-dense.json) |
| 4 | shifted_relu_squared | 3.182611 | 21,076,564 / 12,681,300 | 786.75 | 862.00 | [local](../records/ffn-single-scale-1800-s71-shifted_relu_squared.json) |

### Matched continuation seed461: 1800 parent + 3600 added updates (not fresh) / text

Seed 461; 5,400 updates; 18,067,829 matched targets. [Audit](../records/ffn-filter-continuation-5400-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 4.060270 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-filter-continuation-5400-s461-dynamic_filter.json) |
| 2 | dense | 4.166560 | 8,654,208 / 4,718,592 | 436.44 | 466.00 | [local](../records/ffn-filter-continuation-5400-s461-dense.json) |

### Matched continuation seed461: 1800 parent + 3600 added updates (not fresh) / chat

Seed 461; 5,400 updates; 18,067,829 matched targets. [Audit](../records/ffn-filter-continuation-5400-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | dynamic_filter | 2.712704 | 8,654,208 / 4,718,592 | 446.88 | 468.00 | [local](../records/ffn-filter-continuation-5400-s461-dynamic_filter.json) |
| 2 | dense | 2.801623 | 8,654,208 / 4,718,592 | 436.44 | 466.00 | [local](../records/ffn-filter-continuation-5400-s461-dense.json) |

### Evolution seed461: 10,000 fresh updates with three saved matched controls / text

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-evolution-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 3.820069 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | gate16 | 3.826844 | 8,654,208 / 4,718,592 | 446.92 | 472.00 | [local](../records/ffn-evolution-10000-s461-gate16.json) |
| 3 | gate_sine | 3.827993 | 8,654,208 / 4,718,592 | 447.97 | 468.00 | [local](../records/ffn-evolution-10000-s461-gate_sine.json) |
| 4 | partial_rational | 3.829353 | 8,654,208 / 4,718,592 | 457.49 | 484.00 | [local](../records/ffn-evolution-10000-s461-partial_rational.json) |
| 5 | partial_positive_sine | 3.830455 | 8,654,208 / 4,718,592 | 457.49 | 484.00 | [local](../records/ffn-evolution-10000-s461-partial_positive_sine.json) |
| 6 | partial_sine | 3.833127 | 8,654,208 / 4,718,592 | 457.49 | 482.00 | [local](../records/ffn-evolution-10000-s461-partial_sine.json) |
| 7 | gate8 | 3.837566 | 8,654,208 / 4,718,592 | 446.62 | 472.00 | [local](../records/ffn-evolution-10000-s461-gate8.json) |
| 8 | dynamic_filter | 3.842274 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 9 | filter_updategate | 3.855924 | 8,654,208 / 4,718,592 | 472.30 | 508.00 | [local](../records/ffn-evolution-10000-s461-filter_updategate.json) |
| 10 | filter_postgate | 3.865842 | 8,654,208 / 4,718,592 | 469.68 | 508.00 | [local](../records/ffn-evolution-10000-s461-filter_postgate.json) |
| 11 | dense | 3.980314 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |

### Evolution seed461: 10,000 fresh updates with three saved matched controls / chat

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-evolution-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 2.411506 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | gate16 | 2.412582 | 8,654,208 / 4,718,592 | 446.92 | 472.00 | [local](../records/ffn-evolution-10000-s461-gate16.json) |
| 3 | partial_positive_sine | 2.413345 | 8,654,208 / 4,718,592 | 457.49 | 484.00 | [local](../records/ffn-evolution-10000-s461-partial_positive_sine.json) |
| 4 | partial_rational | 2.413616 | 8,654,208 / 4,718,592 | 457.49 | 484.00 | [local](../records/ffn-evolution-10000-s461-partial_rational.json) |
| 5 | gate_sine | 2.418575 | 8,654,208 / 4,718,592 | 447.97 | 468.00 | [local](../records/ffn-evolution-10000-s461-gate_sine.json) |
| 6 | partial_sine | 2.422035 | 8,654,208 / 4,718,592 | 457.49 | 482.00 | [local](../records/ffn-evolution-10000-s461-partial_sine.json) |
| 7 | gate8 | 2.426850 | 8,654,208 / 4,718,592 | 446.62 | 472.00 | [local](../records/ffn-evolution-10000-s461-gate8.json) |
| 8 | dynamic_filter | 2.432455 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 9 | filter_updategate | 2.440036 | 8,654,208 / 4,718,592 | 472.30 | 508.00 | [local](../records/ffn-evolution-10000-s461-filter_updategate.json) |
| 10 | filter_postgate | 2.451138 | 8,654,208 / 4,718,592 | 469.68 | 508.00 | [local](../records/ffn-evolution-10000-s461-filter_postgate.json) |
| 11 | dense | 2.549469 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |

### Evidence seed461: 10,000 fresh updates with three saved matched controls / text

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-evidence-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 3.820069 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | evidence_balance | 3.820250 | 8,654,208 / 4,718,592 | 448.13 | 468.00 | [local](../records/ffn-evidence-10000-s461-evidence_balance.json) |
| 3 | reaction_catalyst | 3.820372 | 8,654,208 / 4,718,592 | 470.10 | 490.00 | [local](../records/ffn-evidence-10000-s461-reaction_catalyst.json) |
| 4 | attention_energy | 3.820679 | 8,654,208 / 4,718,592 | 460.50 | 488.00 | [local](../records/ffn-evidence-10000-s461-attention_energy.json) |
| 5 | evidence_support | 3.820992 | 8,654,208 / 4,718,592 | 448.13 | 468.00 | [local](../records/ffn-evidence-10000-s461-evidence_support.json) |
| 6 | reaction_post | 3.823193 | 8,654,208 / 4,718,592 | 446.86 | 468.00 | [local](../records/ffn-evidence-10000-s461-reaction_post.json) |
| 7 | attention_alignment | 3.825272 | 8,654,208 / 4,718,592 | 463.11 | 488.00 | [local](../records/ffn-evidence-10000-s461-attention_alignment.json) |
| 8 | cross_evidence | 3.827581 | 8,654,208 / 4,718,592 | 454.27 | 472.00 | [local](../records/ffn-evidence-10000-s461-cross_evidence.json) |
| 9 | learned_evidence | 3.841977 | 8,654,208 / 4,718,592 | 450.94 | 468.00 | [local](../records/ffn-evidence-10000-s461-learned_evidence.json) |
| 10 | dynamic_filter | 3.842274 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 11 | dense | 3.980314 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |

### Evidence seed461: 10,000 fresh updates with three saved matched controls / chat

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-evidence-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | attention_energy | 2.411464 | 8,654,208 / 4,718,592 | 460.50 | 488.00 | [local](../records/ffn-evidence-10000-s461-attention_energy.json) |
| 2 | branch_sigmoid | 2.411506 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 3 | evidence_balance | 2.411573 | 8,654,208 / 4,718,592 | 448.13 | 468.00 | [local](../records/ffn-evidence-10000-s461-evidence_balance.json) |
| 4 | reaction_catalyst | 2.412067 | 8,654,208 / 4,718,592 | 470.10 | 490.00 | [local](../records/ffn-evidence-10000-s461-reaction_catalyst.json) |
| 5 | evidence_support | 2.412164 | 8,654,208 / 4,718,592 | 448.13 | 468.00 | [local](../records/ffn-evidence-10000-s461-evidence_support.json) |
| 6 | attention_alignment | 2.412334 | 8,654,208 / 4,718,592 | 463.11 | 488.00 | [local](../records/ffn-evidence-10000-s461-attention_alignment.json) |
| 7 | reaction_post | 2.413226 | 8,654,208 / 4,718,592 | 446.86 | 468.00 | [local](../records/ffn-evidence-10000-s461-reaction_post.json) |
| 8 | cross_evidence | 2.413500 | 8,654,208 / 4,718,592 | 454.27 | 472.00 | [local](../records/ffn-evidence-10000-s461-cross_evidence.json) |
| 9 | dynamic_filter | 2.432455 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 10 | learned_evidence | 2.433106 | 8,654,208 / 4,718,592 | 450.94 | 468.00 | [local](../records/ffn-evidence-10000-s461-learned_evidence.json) |
| 11 | dense | 2.549469 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |

### Branch refinement seed461: 10,000 fresh updates with three saved matched controls / text

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-branch-refinement-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 3.820069 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | router_mlp | 3.828213 | 8,654,208 / 4,718,592 | 454.39 | 472.00 | [local](../records/ffn-branch-refinement-10000-s461-router_mlp.json) |
| 3 | branch2 | 3.831924 | 8,654,208 / 4,718,592 | 446.97 | 468.00 | [local](../records/ffn-branch-refinement-10000-s461-branch2.json) |
| 4 | ungated | 3.831977 | 8,654,208 / 4,718,592 | 425.63 | 448.00 | [local](../records/ffn-branch-refinement-10000-s461-ungated.json) |
| 5 | dynamic_filter | 3.842274 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 6 | matrix4x4 | 3.847200 | 8,654,208 / 4,718,592 | 446.99 | 472.00 | [local](../records/ffn-branch-refinement-10000-s461-matrix4x4.json) |
| 7 | dense | 3.980314 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |

### Branch refinement seed461: 10,000 fresh updates with three saved matched controls / chat

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-branch-refinement-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | router_mlp | 2.410576 | 8,654,208 / 4,718,592 | 454.39 | 472.00 | [local](../records/ffn-branch-refinement-10000-s461-router_mlp.json) |
| 2 | branch_sigmoid | 2.411506 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 3 | branch2 | 2.411672 | 8,654,208 / 4,718,592 | 446.97 | 468.00 | [local](../records/ffn-branch-refinement-10000-s461-branch2.json) |
| 4 | ungated | 2.415453 | 8,654,208 / 4,718,592 | 425.63 | 448.00 | [local](../records/ffn-branch-refinement-10000-s461-ungated.json) |
| 5 | dynamic_filter | 2.432455 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 6 | matrix4x4 | 2.436932 | 8,654,208 / 4,718,592 | 446.99 | 472.00 | [local](../records/ffn-branch-refinement-10000-s461-matrix4x4.json) |
| 7 | dense | 2.549469 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |

### Independent dual-path seed461: 10,000 fresh updates with three saved matched controls / text

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-dual-path-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 3.820069 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | dual_plain_quarter | 3.821241 | 8,654,208 / 4,718,592 | 441.80 | 468.00 | [local](../records/ffn-dual-path-10000-s461-dual_plain_quarter.json) |
| 3 | dual_router_small | 3.833562 | 8,654,208 / 4,718,592 | 452.70 | 470.00 | [local](../records/ffn-dual-path-10000-s461-dual_router_small.json) |
| 4 | dual_filter_quarter | 3.833604 | 8,654,208 / 4,718,592 | 448.92 | 478.00 | [local](../records/ffn-dual-path-10000-s461-dual_filter_quarter.json) |
| 5 | dual_filter_half | 3.840192 | 8,654,208 / 4,718,592 | 450.75 | 474.00 | [local](../records/ffn-dual-path-10000-s461-dual_filter_half.json) |
| 6 | dynamic_filter | 3.842274 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 7 | dual_filter_small | 3.843426 | 8,654,208 / 4,718,592 | 451.54 | 468.00 | [local](../records/ffn-dual-path-10000-s461-dual_filter_small.json) |
| 8 | dense | 3.980314 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |

### Independent dual-path seed461: 10,000 fresh updates with three saved matched controls / chat

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-dual-path-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 2.411506 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | dual_plain_quarter | 2.415987 | 8,654,208 / 4,718,592 | 441.80 | 468.00 | [local](../records/ffn-dual-path-10000-s461-dual_plain_quarter.json) |
| 3 | dual_router_small | 2.416228 | 8,654,208 / 4,718,592 | 452.70 | 470.00 | [local](../records/ffn-dual-path-10000-s461-dual_router_small.json) |
| 4 | dual_filter_small | 2.423266 | 8,654,208 / 4,718,592 | 451.54 | 468.00 | [local](../records/ffn-dual-path-10000-s461-dual_filter_small.json) |
| 5 | dual_filter_quarter | 2.423454 | 8,654,208 / 4,718,592 | 448.92 | 478.00 | [local](../records/ffn-dual-path-10000-s461-dual_filter_quarter.json) |
| 6 | dual_filter_half | 2.424349 | 8,654,208 / 4,718,592 | 450.75 | 474.00 | [local](../records/ffn-dual-path-10000-s461-dual_filter_half.json) |
| 7 | dynamic_filter | 2.432455 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 8 | dense | 2.549469 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |

### Branch-next seed461: 10,000 fresh updates with three saved matched controls / text

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-branch-next-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 3.820069 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | branch_hierarchical | 3.823591 | 8,654,208 / 4,718,592 | 444.61 | 482.00 | [local](../records/ffn-branch-next-10000-s461-branch_hierarchical.json) |
| 3 | branch_rich_router | 3.829860 | 8,654,208 / 4,718,592 | 450.41 | 474.00 | [local](../records/ffn-branch-next-10000-s461-branch_rich_router.json) |
| 4 | branch_threshold | 3.832266 | 8,654,208 / 4,718,592 | 451.13 | 472.00 | [local](../records/ffn-branch-next-10000-s461-branch_threshold.json) |
| 5 | dynamic_filter | 3.842274 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 6 | branch_temperature | 3.844234 | 8,654,208 / 4,718,592 | 450.01 | 472.00 | [local](../records/ffn-branch-next-10000-s461-branch_temperature.json) |
| 7 | branch_bipolar | 3.897990 | 8,654,208 / 4,718,592 | 471.55 | 508.00 | [local](../records/ffn-branch-next-10000-s461-branch_bipolar.json) |
| 8 | dense | 3.980314 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |

### Branch-next seed461: 10,000 fresh updates with three saved matched controls / chat

Seed 461; 10,000 updates; 33,440,772 matched targets. [Audit](../records/ffn-branch-next-10000-s461-audit.json).

| Rank | FFN | Full final NLL | Core / FFN params | Allocated MiB | Reserved MiB | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | branch_sigmoid | 2.411506 | 8,654,208 / 4,718,592 | 454.54 | 480.00 | [local](../records/ffn-final-10000-s461-branch_sigmoid.json) |
| 2 | branch_rich_router | 2.411784 | 8,654,208 / 4,718,592 | 450.41 | 474.00 | [local](../records/ffn-branch-next-10000-s461-branch_rich_router.json) |
| 3 | branch_hierarchical | 2.412127 | 8,654,208 / 4,718,592 | 444.61 | 482.00 | [local](../records/ffn-branch-next-10000-s461-branch_hierarchical.json) |
| 4 | branch_threshold | 2.425312 | 8,654,208 / 4,718,592 | 451.13 | 472.00 | [local](../records/ffn-branch-next-10000-s461-branch_threshold.json) |
| 5 | branch_temperature | 2.431398 | 8,654,208 / 4,718,592 | 450.01 | 472.00 | [local](../records/ffn-branch-next-10000-s461-branch_temperature.json) |
| 6 | dynamic_filter | 2.432455 | 8,654,208 / 4,718,592 | 448.38 | 480.00 | [local](../records/ffn-final-10000-s461-dynamic_filter.json) |
| 7 | branch_bipolar | 2.497739 | 8,654,208 / 4,718,592 | 471.55 | 508.00 | [local](../records/ffn-branch-next-10000-s461-branch_bipolar.json) |
| 8 | dense | 2.549469 | 8,654,208 / 4,718,592 | 436.88 | 462.00 | [local](../records/ffn-final-10000-s461-dense.json) |
