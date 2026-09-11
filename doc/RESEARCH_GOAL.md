# Research goal

Discover the simplest trainable FFN primitive that improves nonlinear
representation per learned parameter while preserving optimization and GPU utility.

**Current primary objective (user clarification, 2026-09-10): reduce actual VRAM
while preserving quality and practical runtime.** Parameter reduction is one
possible means. Execution scheduling, activation storage and optimizer memory
are also eligible; each must be measured against equally optimized controls.
Report training and inference memory separately. Mathematical equivalence alone
does not establish floating-point training fidelity.

The original architectural gold target remains: >=70% fewer FFN parameters with <=1% relative validation NLL
degradation versus full dense GELU and SwiGLU. This is a target, not a finding.
Report total-model parameters, memory, throughput, compute and stability.
The smaller primitive must also beat a narrow conventional parameter control.
An execution-only memory improvement does not satisfy that architectural target
and is not evidence of a novel activation or a breakthrough.

Use progressively stronger evidence: mathematical analysis → sanity/overfit →
function fitting → TinyStories → multiple seeds/ablations → larger models and a
broader corpus. No single synthetic benchmark establishes general superiority.
Negative results and a quantitatively measured frontier are valid outcomes.

Start at [current state](../research/CURRENT_STATE.md), then read the
[frozen protocol](../research/experimental_plan.md), [theory](../research/theory.md)
and [literature audit](../research/literature.md). The research is not limited to
Bézier curves; follow the evidence toward the smallest useful mechanism.
