# H114 completion audit after replay recovery failed

The frozen numerical-replay recovery also stopped. Five probes passed before
`wikitext2_s61_loss_chunks__block_fp32` failed on its sixth backward: global
relative L2 0.0005106002, maximum tensor relative L2 0.0009818187, maximum absolute
element difference 3.051758e-5. Its tokens, targets, weights and scalar loss
matched. The difference is larger than the original six-pass diagnostic and
exceeds the explicitly frozen 1e-5/1e-4 audit tolerances. Its cause is unresolved.

**Do not relax those tolerances again or repeat training.** Preserve both failed
audits, original code/protocols, traces and the diagnostic. H114's 2,800 updates
and 2,856 profile backwards remain complete; this recovery performed six extra
backwards and no optimizer updates. The first failed audit's one-to-four
backward range remains unknown. No complete gradient replay audit has passed.

The completion audit is deliberately narrower: perform zero backwards and zero
updates, verify all original saved gradients by hash and compute the original
candidate-versus-FP32 comparisons from those tensors, independently rescore all
14 source and 56 final models through native BF16 forward, and verify every
training batch plus saved Adam state. Freeze its source before execution.
Write `completion_audit.json`, explicitly setting
`full_gradient_replay_qualified: false` and preserving both failed receipts.

Report whether the original *measured* loss/gradient/memory/time/NLL gates pass,
but hold any fresh LM allocation because the replay qualification is unresolved.
This is an additional audit hold, not a retrospective relaxation of any
scientific threshold. Native scoring and resource measurements may be reported
within their scope. Further backward determinism work requires a new prospective
diagnosis; do not attribute the discrepancy to a specific operator without it.
