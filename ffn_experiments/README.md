# Archived FFN shortlist

Curve-Wide is the selected active model in `src/models/channel_curve_transformer`.
This archive keeps the strongest alternatives available for later research,
outside the active trainer and installed `src` package.

## Current decision and limits of the selection

Continue with the current plain-PyTorch Curve-Wide implementation. The notes
below preserve research limitations for later; they do not reopen experiments
or promote any archived model into the active trainer.

Curve-Wide was selected from the experiments that completed our screening
process. Some alternatives stopped on speed or memory before language training,
so their language quality is unknown. A resource failure is not evidence that
the architecture cannot learn well. Several received custom optimization;
we have not established equal optimization effort or that further work would
make them better than Curve-Wide.

| Unresolved candidate | Why it stopped before language | Evidence |
| --- | --- | --- |
| Learned basis templates | Roughly 25–28% memory savings, but missed speed limits | [Report](../dump/old-research-docs/basis_readout_result.md) |
| Paired readout v1 | 19.8% memory savings versus GELU missed the 20% threshold; also missed a speed comparison | [Report](../dump/old-research-docs/paired_readout_result.md) |
| Readout reuse v5 | Memory savings, but excessive slowdown after multiple execution revisions | [Report](../dump/old-research-docs/readout_reuse_v5_result.md) |
| Gate transport v2 | 17.9% memory savings versus GELU missed the threshold | [Report](../dump/old-research-docs/retired_models/shared_basis_transformer/result.md) |

Channel-curve execution v1 itself failed the speed screen; a later execution
revision passed and enabled language testing, from which the self-curve control
eventually led to Curve-Wide. [Initial failure](../dump/old-research-docs/channel_curve_result.md).
Other families, including Group Product and Feature Flow, eventually completed
language tests and failed quality comparisons. Do not classify all retired
models as merely underoptimized. These links preserve local evidence; this
archive does not contain every retired implementation.

If this research is reopened, separate language-quality screening at a small
parameter budget from execution optimization. Consider a modest language test
before rejecting a mathematically valid idea solely on resource thresholds.
This is a proposed future protocol, not a new result or a training instruction.

## What quality gain is established?

In the historical three-seed, 2000-update comparison, Curve-Wide achieved mean
validation NLL 2.592300/4.272862 on TinyStories/WikiText, versus compact SwiGLU
h384 at 2.607348/4.282197 (lower is better). Curve-Wide used 2,121,728 FFN weights
versus that control's 2,359,296. It also beat that compact control on both
held-out datasets in the longer single-seed comparison. This supports a useful
quality/parameter tradeoff under the tested settings, not universal superiority.

We have **not tested Curve-Wide enlarged to the full dense model's parameter
budget**. Its winning that comparison is a hypothesis, not an established
result. Full SwiGLU won both datasets in the short three-seed comparison; in
the longer single-seed test it won TinyStories while Curve-Wide won WikiText.
The active plain-PyTorch revision passed numerical and checkpoint checks but
has no new language-training result. Historical optimized memory savings do
not apply to it. [Current execution report](../src/models/channel_curve_transformer/result.md).

## Retained implementations

| Family | Starting recipe to revisit | Why keep it? |
| --- | --- | --- |
| [Signed Linear](signed_context_transformer/transformer.py) | `signed_linear`, hidden576/groups64 | Strong compact alternative; slightly better WikiText mean in the short three-seed comparison. |
| [Self-Curve](channel_curve_transformer/transformer.py) | `self_curve`, width512 | Same size as Curve-Wide; original parent for studying learned curve initialization. |
| [Basis Readout](basis_readout_transformer/transformer.py) | `fixed_basis`, width512/groups16 | Different grouped readout mechanism; smaller groups did not improve both corpora. |

Each folder retains `result.md` and a representative `config.json`. The archived
implementations also retain their family controls and removal switches for
comparison. The curve archive includes the pre-cleanup Curve-Wide implementation
as a numerical reference. `settings.py` and `components.py` preserve supporting
definitions; `context_activation.py` is a dependency of the signed family.
Existing kernel code is retained without optimization.

From the repository root, load an archived model with:

```python
import json
from pathlib import Path
from ffn_experiments.settings import ModelConfig
from ffn_experiments.signed_context_transformer.transformer import Model

recipe = json.loads(Path("ffn_experiments/signed_context_transformer/config.json").read_text())
model = Model(ModelConfig(**recipe["model"]), seed=recipe["training"]["seed"])
```

Run this with the repository environment, for example `uv run --no-sync python`.
To resume research, explicitly promote the chosen recipe into the active registry,
register a new study, and compare against Curve-Wide and the dense baseline.
The current trainer intentionally rejects archived model configurations.
Historical replay requires the original source snapshots; archive imports have
changed, so this directory is not a byte-identical replay environment.

[Selection and evidence](../docs/compact_refinement_summary.md) ·
[All ranked results](../docs/leaderboard.md)
