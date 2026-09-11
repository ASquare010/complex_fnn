# Paused by user — H160 checkpoint and research handoff

Training and its telemetry processes were stopped on request. **Do not resume
training or start another experiment until the user explicitly resumes.**

## Preserved state

- H160 cases00,01,02 completed: 2,400 optimizer updates. This includes the first
  native/FP16 pair and the second seed's FP16 run; no full H160 audit has run.
- Case03 (WikiText2 seed113 native) was interrupted after207 recorded updates.
  Its latest resumable checkpoint is step200; updates201-207 are not checkpointed.
  Keep the partial history and checkpoint. No completed run should be rerun.
- Remaining cases04-11 have not trained. The original H160 result is incomplete.
- `results/checkpoint_fp16_long_v1/pause.json` records exact progress and hashes.
- `src/experimental/checkpoints/manifest.json` indexes three selected complete
  model/Adam/sampler checkpoints copied into `src/experimental/checkpoints/`.
  Binary weights remain local and git-ignored; code and manifest are committed.
- `src/core/training_memory.py` contains the qualified exact-memory helpers.
  `src/experimental/checkpoint_compression.py` is a byte-identical copy of the
  tested FP16 prototype, with its restrictions in the adjacent README.

Do not blindly invoke the original H160 `train` launcher: its schedule starts at
case00. A later recovery must preserve the original pause, choose either a clearly
labelled continuation from step200 or a documented restart of only interrupted
case03, and account for the interruption when interpreting uninterrupted timing.
No such recovery has been run or approved as a new experiment during this pause.

## What actually improved

H156 qualified the exact buffered-loss/four-block-offload helper over12 fresh
800-update runs: about24% less allocated GPU memory, with per-seed quality and
timing gates passed. H159's FP16 storage variant saved23.77-23.99% GPU allocation
at complete-update times within0.61% of native in1,080 short updates. It also avoids
the large pinned offload allocation in isolated H158 probes. H160 is testing the
remaining fresh-learning question. These are useful memory results, not new
activations, fewer parameters, cross-domain superiority or a proven breakthrough.

## Why the architectural goal remains unresolved

1. Fewer coefficients do not automatically preserve learnable feature directions.
   H154's narrow coupling used74.24% fewer parameters but had roughly31% higher
   MSE than wide GELU on both qualified synthetic tasks. Expressivity, optimization
   and parameter counts are different constraints.
2. Smaller parameter sets can still need more saved activations, temporary tensors,
   recomputation or kernel launches. Real peak VRAM and complete-update timing have
   rejected candidates that looked attractive from equations or parameter counts.
3. The current evidence is narrow: three seeds, small models and repeatedly used
   development corpora. Passing a short continuation cannot establish convergence
   from initialization, untouched test performance or unrelated-domain generality.
4. Our process became too fragmented. Too many small variants and repeated
   qualification studies consumed time and tokens without enough promotion into
   a small usable set of models. Windows import crashes and allocator accounting
   mistakes added overhead, but they are not the fundamental scientific barrier.

The project is not technically blocked: the paused study can continue. The missing
result is evidence that a substantially smaller architecture retains capability
and real memory efficiency. Neither persistence nor a more exotic activation
alone guarantees that result.

## How to improve after resuming

- Finish or formally close H160 first, with one documented interruption recovery.
  Decide whether the FP16 codec merits a scoped maintained API; do not start another
  chain of near-identical memory microbenchmarks.
- Treat the exact helper as the current practical result. If FP16 passes long
  training, test one larger memory-bound workload and one unrelated benchmark.
- Limit architectural work to at most two hypotheses per round. Give each a fixed
  compute budget and a preregistered stop rule. Use the failure log to prevent repeats.
- Screen mathematical capacity and trainability cheaply before language-model runs.
  Always include a parameter-matched conventional model and a strong wide control.
- Promote only a real Pareto improvement: meaningful measured VRAM savings,
  controlled quality and complete-update cost, all seeds reported. Separate useful
  engineering from novelty claims, and reserve an untouched test for the finalists.
- Keep one short current-state document and concise progress reports; spend tokens
  on experiments and interpretation rather than repeated protocol restatements.

No further research runs were started after the stop request. The research goal
is incomplete and paused, not achieved or scientifically blocked.

Packaging checks: the copied codec matches its tested source byte-for-byte; new
package/handoff checks pass. Historical generated SVG whitespace and two existing
EOF-blank findings are preserved to avoid changing frozen evidence artifacts.
