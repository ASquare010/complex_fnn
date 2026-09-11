# H150: order-balanced confirmation of fused reconstruction timing

Previous turn: progress. H149 passed every memory/speed comparison, but one
candidate and one full-control timing window failed stability. Preserve that
result; do not drop those seeds or relabel their measurements as successful.
Freeze an independent confirmation protocol with longer windows and mirrored
arm order. No kernel, architecture, optimizer or acceptance-threshold changes.

Use H149's saved step12 model AND Adam states for each batch128/2048 and
seed223/239/251. Arms: full GELU checkpoint, full SwiGLU checkpoint, learned fused.
For each fixture rotate ABC by fixture index, then execute ABC CBA. Each segment
resets to the identical saved state for its arm and runs30 updates:10 warmups,
20 timed complete updates.36 segments,1080 updates/backwards,37 clean allocator
boundaries. This is two independent30-update segments, not60 continuous updates.
Reuse the exact H149 forward/backward kernels, data and optimizer recipe; input
gradients enabled, gradient reset outside timing. Save every segment, no retries.

All gates retain H149 thresholds. For BOTH full controls in every fixture:
parameter ratio<=.8, max candidate peak/max control peak<=.9, pooled40-update
median CUDA/wall candidate/control<=1.15. Additionally EVERY segment's timed
half-window median ratio<=1.15 AND repeat-segment median ratio<=1.15 for each
arm/fixture. These guard within-segment and across-repeat drift. All failures
remain in the report. No power/clock changes or data-dependent warmup extension.

This tests short timing reproducibility, not new-seed generalization, sustained
throughput, quality or a new architecture. State/data reuse is deliberate to
separate execution effects from initialization/training differences. Two source
states originate in unstable H149 windows; their timing is not reused.

Audit exact source/restored model/optimizer states,30-step histories/counter42,
finite diagnostics, fixed-buffer invariance, six bitwise-regenerated datasets,
and36 independent saved-state CPU FP64 scores (batch257,relative tolerance1e-5).
No audit backwards or GPU work. Report matched repeat final-state differences,
all timings, mean/median/variance, setup costs and source hashes. Keep H149 receipts,
maintained modules/defaults and cached kernel code unchanged. Resource pass only
earns a separate quality study with full activation and fixed/affine controls.
