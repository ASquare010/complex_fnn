# Confirmation resource review

All 60 profiles are retained. Each median covers three alternating rounds of 100 measured updates after 20 warmup updates.

WikiText full-baseline times vary by more than 2x. The cause is unresolved; these measurements do not support a reliable speedup claim. Only the profiling process appeared in the CUDA process snapshot. No clocks or power settings were changed.

| Corpus | Recipe | Median seconds | Min–max seconds | Peak allocated MiB |
| --- | --- | ---: | ---: | ---: |
| tinystories | compact-gelu-h518 | 4.651 | 4.483–4.718 | 413.1 |
| tinystories | compact-gelu-h576 | 4.543 | 4.543–4.751 | 415.9 |
| tinystories | compact-swiglu-kernel-h346 | 4.725 | 4.674–4.751 | 413.0 |
| tinystories | compact-swiglu-kernel-h384 | 4.701 | 4.645–4.849 | 411.5 |
| tinystories | curve-parent | 4.791 | 4.710–4.851 | 401.2 |
| tinystories | curve-wide | 4.973 | 4.857–4.994 | 403.2 |
| tinystories | full-gelu | 4.908 | 4.845–5.072 | 544.3 |
| tinystories | full-swiglu | 5.017 | 4.848–5.141 | 562.9 |
| tinystories | signed-fine | 4.933 | 4.766–4.982 | 409.9 |
| tinystories | signed-parent | 4.904 | 4.895–4.999 | 418.5 |
| wikitext | compact-gelu-h518 | 4.901 | 4.708–5.182 | 407.9 |
| wikitext | compact-gelu-h576 | 3.654 | 2.837–4.732 | 414.4 |
| wikitext | compact-swiglu-kernel-h346 | 4.935 | 4.789–5.104 | 404.7 |
| wikitext | compact-swiglu-kernel-h384 | 4.741 | 4.648–4.757 | 414.2 |
| wikitext | curve-parent | 5.020 | 4.826–5.254 | 405.2 |
| wikitext | curve-wide | 5.218 | 4.865–5.272 | 405.2 |
| wikitext | full-gelu | 2.760 | 2.224–5.050 | 538.0 |
| wikitext | full-swiglu | 2.980 | 2.457–5.218 | 563.4 |
| wikitext | signed-fine | 4.982 | 4.854–5.053 | 408.7 |
| wikitext | signed-parent | 4.947 | 4.920–5.460 | 415.6 |

Curve-Wide meets the registered parent-relative 5% limits using the observed medians and peak allocated memory. This is a point-estimate check, not proof of a stable timing difference. Its lower allocated memory than both full baselines is retained as the resource benefit; do not present these timings as a proven speedup.

[Raw measurements](../records/compact-confirmation-v1-resources.json).
