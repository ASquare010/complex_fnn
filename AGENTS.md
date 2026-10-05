# Current repository scope

Keep only Curve-Wide (`channel_curve_transformer`, `self_curve_wide`) and the
dense baseline active in `src/models`. The latest bounded refinement is complete.
Root `ffn_experiments/` retains the explicitly requested shortlist for later
research. Do not restore retired code, old research trees, scripts or historical
study configurations to the active tree. No new kernel optimization.

Do not modify `docs/research_goal.md` when documenting Step 1.

# Experiment results

After every new experiment, export completed language runs into local `records/`
and refresh `docs/leaderboard.md`. **Keep records ignored and untracked; do not
force-add them.** Raw runs, checkpoints and old documentation remain under
ignored `dump/`. Current result summaries and the leaderboard stay tracked.
Links to local evidence are expected to require those files.

Rank full last-checkpoint validation NLL, best first, separately by dataset and
comparison group. Include baselines and trained controls. Never substitute
sampled validation or synthetic scores. Preserve model/FFN counts, memory, seed,
budget and local evidence. Keep stopped-before-language experiments unranked.

Use `uv run --no-sync ...` to preserve the configured GPU environment, or set
`UV_NO_SYNC=1` before `uv run python -m main leaderboard`. Plain dependency sync
can replace the measured CUDA PyTorch build. Preserve source-provenance checks;
historical runs require their original sources for exact resumption.
