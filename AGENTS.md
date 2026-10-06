# Current repository scope

Keep Curve-Wide (`channel_curve_transformer`, `self_curve_wide`) and the dense
baseline as the active language models. `position_compressor` is the selected
Step 2 reconstruction model. The failed `paragraph_compressor` is retired. The latest bounded FFN refinement is complete.
Root `ffn_experiments/` retains the explicitly requested shortlist for later
research. Do not restore retired code, old research trees, scripts or historical
study configurations to the active tree. No new kernel optimization.

Do not modify `docs/research_goal.md` when documenting Step 1.

# Experiment results

Step 2 is officially complete as a bounded reconstruction milestone (2026-10-05).
The selected integration handoff is the span-32 `position_compressor`; preserve the
original 8-token checkpoint and inference notebook default. Keep reconstruction
metrics in `docs/step_2_results.md`, separate from language-model NLL. Preserve all
checkpoints, source-provenance checks, failed controls and ignored evidence.
Historical source cleanup snapshots remain in `dump/step2-cleanup-v1/`.

Current work is the Step 3 design draft in `docs/step_3_memory_llm.md`: a tiny causal
language model, external compressed memory with CREATE/READ only, bounded sliding
encoding, and at most four shared-weight reasoning loops per output token.
No UPDATE/DELETE memory operations, new kernels or unbounded training runs.
The current target is <=10M total unique learned system parameters, counting frozen
encoder, embeddings, core, generator, adapters and any key/query networks. The older
1M charter is preserved as history; do not claim a 10M core is a 10M complete system.
Resolve the causal cache contract, dataset and bounded experiment protocol before
starting integration training. Design preferences in the draft are not proven wins.
No downstream reasoning, semantic retrieval, universal lossless recovery, raw-byte
compression or whole-LLM speedup follows from Step 2 alone.
Preserve the main
research charter. The active model uses ordinary PyTorch autograd on CPU and GPU;
custom execution code stays in the archive. Do not restore custom kernels to the
active model. Historical optimized memory/timing results do not describe this
plain PyTorch execution revision.

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
