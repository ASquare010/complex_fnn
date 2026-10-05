# Atlas / complex_fnn

The active code contains **Curve-Wide**, our selected compact FFN, and a
**dense Transformer baseline**. Both retain embeddings, full causal attention,
normalization, residual connections and the output head. Only the FFN differs.

Curve-Wide uses 74.9% fewer FFN parameters, 42.9% fewer total parameters and about
28% less allocated training memory than full SwiGLU in our measured four-layer
models. It beats its parent and compact SwiGLU on both held-out datasets.
At the longer single-seed budget, full SwiGLU wins TinyStories and Curve-Wide wins
WikiText. Reliable speedup, generalization to large LLMs and research novelty
remain unproven.

[Final results](docs/compact_refinement_summary.md) ·
[Dataset leaderboards](docs/leaderboard.md) ·
[Step 1](docs/step_1_ffn.md) · [Main research goal](docs/research_goal.md)

## Code map

```text
src/
  models/
    channel_curve_transformer/  Curve-Wide only; name preserves checkpoint compatibility
      transformer.py            complete model and learned curves
      kernels.py                existing execution implementation, unchanged
      result.md                 measured results
    base_transformer/           conventional dense baseline
    components.py               attention, normalization and dense FFN
    activation_kernels.py       existing dense activation implementation
  config/                       winner/baseline presets and data recipes
  dataset/                      preparation, loading and synthetic smoke data
  trainer.py                    training, checkpointing and evaluation
  main.py                       command-line entry point
  experiments.py                planning, comparison and evidence export
  leaderboard.py                automatic full-validation rankings
ffn_experiments/                 archived shortlist for future research
docs/                           current research goal, selected study and leaderboard
records/                        local experiment evidence; ignored by Git
dump/                           ignored local data, checkpoints and source snapshots
```

Start reading [CurveFFN](src/models/channel_curve_transformer/transformer.py), then
`Block` and `Model` in the same file. Its computation is **mix → learned curves →
mix**. The selected mode is `self_curve_wide`; other curve modes are archived.

## Run the selected model

Use the configured Python environment. `--no-sync` preserves its existing GPU
PyTorch installation instead of replacing it from the general dependency lock.
A fresh environment needs the dependencies in `pyproject.toml` and a compatible
CUDA-enabled PyTorch installation for the CUDA presets.

```powershell
uv run --no-sync python -m main inspect src/config/curve_wide_tinystories.json
uv run --no-sync python -m main prepare src/config/data/tinystories.json
uv run --no-sync python -m main run src/config/curve_wide_tinystories.json
```

The matching control is `src/config/baseline_tinystories.json`. WikiText has
`curve_wide_wikitext.json`, `baseline_wikitext.json`, and `data/wikitext.json`.
These presets retain the measured 512-wide, four-layer, 8000-update comparison.
The `smoke.json` preset checks plumbing with the synthetic smoke data recipe.

```powershell
uv run --no-sync python -m main evaluate dump/runs/<run-id>
uv run --no-sync python -m main export dump/runs/<run-id> <unique-record-name>
uv run --no-sync python -m main leaderboard
```

Validation and export refresh the leaderboard automatically. Rank full
last-checkpoint validation within each dataset and comparison group; keep test
scores separate. Held-out evaluation requires `--split test --final` after the
recipe is frozen. Recorded evaluations are never overwritten.

## Archived research

[ffn_experiments](ffn_experiments/README.md) retains Signed Linear, Self-Curve and
Basis Readout, including their family controls and original kernels where needed.
They are not registered in the active trainer. Older documentation is kept only
in ignored `dump/old-research-docs/`; retired implementations are absent.
`records/` is local evidence and is no longer tracked by Git. Links to raw records,
checkpoints and old reports therefore work only where those local files exist.
The current study tables and conclusions remain in `docs/`.

Recorded runs enforce exact source provenance. Load old weights directly for
inspection, or use their original local source snapshots to resume/re-evaluate
historical experiments; do not bypass the trainer's source checks. This cleanup
does not change the recorded experiments or start a new research campaign.
