# Parameter-efficient FFN research

**The research goal remains unmet.** We are testing whether a smaller FFN can
match full GELU and SwiGLU with at least 70% fewer FFN weights, while beating
calibrated narrow controls at practical memory and runtime costs.

Plain BlockShuffle achieves **70.31% fewer FFN weights** and **42.17% fewer total
model weights**. In the latest one-seed 3,200-update WikiText-2 comparison its
validation loss is 4.1443 versus 4.1054 for full SwiGLU. Earlier three-seed evidence
misses the 1% loss allowance, and native updates are slower. These are promising
compression measurements, not a completed breakthrough.

The latest [nonlinear-residual study](research/nonlinear_residual_results.md)
tests a new four-parameter rational mixture in 264 runs. It reduces aggregate
synthetic error 12.39% versus narrow GELU, but only 0.060% versus StarReLU and
costs 40.49% more native update time than narrow GELU. It fails the frozen gates
and is rejected at this budget. All 528 checkpoint scores reproduce exactly.

The new data isolates nonlinear structure: three tasks are learned, while cubic
remains unresolved for ordinary models. Known-feature controls learn all four.
This narrows the next question to feature discovery and optimization, without
promoting another small activation improvement. The earlier
[affine-control finding](research/affine_falsification_results.md) is preserved.

## Start here

- [Progress overview](research/PROGRESS_OVERVIEW.md): results, data, batches,
  duration and activation experiments.
- [Current state](research/CURRENT_STATE.md): decisions and outstanding evidence.
- [New research direction](research/research_direction_2026_09_10.md): coupled
  feature-pair geometry, falsifiable comparisons and prior work.
- [Candidate ledger](research/idea_bank.md): hypotheses and eliminated branches.
- [Goal](doc/RESEARCH_GOAL.md) and [rules](doc/RESEARCH_RULES.md).

## Maintained code

Only **two model folders, five variants and eight recipes** remain active.

| Folder | Role |
|---|---|
| [dense_ffn](src/dense_ffn/README.md) | Full and narrow GELU/SwiGLU controls |
| [blockshuffle_ffn](src/blockshuffle_ffn/README.md) | Compressed comparison operator; its limitations remain documented |

Shared data, training, evaluation and diagnostics live in src/core. Rejected
architectures, including rational BlockShuffle after repeated memory failures,
are [archived](research/archive/README.md). Isolated prototypes stay with their
experimental source until evidence earns integration.

## Run

Use Python 3.12+, UV and a CUDA-capable PyTorch installation. Existing measurements
use one RTX4070 Laptop GPU with 8 GiB VRAM. Keep GPU experiments sequential.

~~~powershell
uv sync --extra compile --extra data
uv run --extra compile --extra data python -m src.core.cli hardware
uv run --extra compile --extra data python -m src.core.cli counts
uv run --extra compile --extra data python -X faulthandler -m pytest -p no:anyio -q
uv run ruff check src tests scripts main.py
~~~

With the local WikiText-2 cache prepared, a new reference run can be started with:

~~~powershell
uv run --extra compile --extra data python -m src.core.cli train --config configs/wikitext2_blockshuffle_stronger_800.json --cache data/wikitext2_v1 --seed 17
~~~

Read the [recipe notes](configs/README.md) before comparing selected learning rates.
Training creates new artifacts and refuses to overwrite completed runs.

## Evidence without large Git history

This repository keeps readable research, source, compact results and essential
source snapshots. Datasets, checkpoints, raw gradient tensors, caches and repeated
run archives remain local and are ignored. They have not been deleted.
See [artifact policy and inventory](research/ARTIFACTS.md) for exact scope and the
limits of reproducing historical measurements from a small clone.

Local mathematical proofs, controlled fitting, language quality and GPU speed
are separate claims. Neither a better synthetic score nor a passed gradient test
establishes general superiority, novelty, convergence or a breakthrough.
