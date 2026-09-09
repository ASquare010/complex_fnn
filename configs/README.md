# Retained experiment recipes

Eight recipes remain after H087 rational retirement. They are starting settings;
CLI arguments explicitly override selected rates and budgets.

| Corpus | Forms | Cache |
|---|---|---|
| TinyStories | Full GELU, full SwiGLU, calibrated narrow, BlockShuffle | data/tinystories_v1 |
| WikiText-2 | Full GELU, full SwiGLU, calibrated narrow, BlockShuffle | data/wikitext2_v1 |

The retained plain WikiText recipe uses learning rate 0.0012 for its matched-rate
800-step comparison. Other studies select different rates; use the configuration
and final selection recorded by that study. A200-step result is not directly
comparable with an 800-step or 3,200-step endpoint.

Every command creates a new run. Read [current evidence](../research/CURRENT_STATE.md)
before allocating compute. The rejected rational recipe and historical grids
are [archived](../research/archive/README.md); use original source snapshots to
reproduce historical experiments. Do not overwrite completed result directories.
