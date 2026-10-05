# Experiment results

After every new experiment, keep `docs/leaderboard.md` current. Export completed
language runs to compact `records/` evidence, then run
`uv run python -m main leaderboard`. Shared-trainer full validation and export
already refresh it automatically; frozen/staged launchers must refresh explicitly.

Rank full last-checkpoint validation NLL, best first, separately for TinyStories
and WikiText and for each comparison group. Include all baselines and trained
controls; never substitute sampled validation or synthetic-task scores. Preserve
record links, model/FFN weights, training memory, seed and budget. List experiments
stopped before language as unranked, updating that section when a new family or
execution revision is tested. Keep each study's result.md and removal evidence.

Do not modify `docs/research_goal.md` when documenting Step 1 experiments.

Current selection: Curve-Wide (`channel_curve_transformer`, `self_curve_wide`).
Keep only this model and the dense baseline active in `src/models`. The shortlist
for future research lives in root `ffn_experiments/`; other implementations are
retired, with their reports retained under `docs/retired_models/`. Do not register
archived candidates again without a new research task. Preserve historical
evidence and the trainer's source-provenance checks. No new kernel optimization.

Use the configured GPU environment without dependency resynchronization when
running checks: `uv run --no-sync ...`, or set `UV_NO_SYNC=1` before the required
leaderboard command. A plain sync can replace the measured CUDA PyTorch build.
