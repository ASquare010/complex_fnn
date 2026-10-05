# Archived FFN shortlist

Curve-Wide is the selected active model in `src/models/channel_curve_transformer`.
This archive keeps the strongest alternatives available for later research,
outside the active trainer and installed `src` package.

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
