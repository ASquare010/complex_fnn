# Position compressor

Start with [the inference notebook](../../../notebooks/paragraph_compressor_inference.ipynb).
It loads the selected model, reconstructs your text, shows differences and saves vectors.

- `transformer.py`: configuration, attention, encoder, ordered span compression and decoder.
- `codec.py`: text inference, persisted memories and encoder-only export.
- `data.py`: reversible tokenizer, preparation and batches.
- `training.py`: optimizer, continuation and exact checkpoint recovery.
- `evaluation.py`: full reconstruction metrics and source fingerprints.
- `result.md`: measured results and limitations.

The decoder uses only vectors and exact token lengths. Architecture and trained
weights are unchanged by cleanup. The current source-compatible checkpoint and
encoder export are in `dump/position-pattern-5000-clean-v1/`. Original weights and
all study sources remain preserved. Original training runs require original
source snapshots for exact resumption; no provenance checks were disabled.

The old pooled model and experiment launchers are no longer active. Archived
recipes, including the synthetic pattern training that produced the winner,
are available locally under `dump/step2-cleanup-v1/src/models/position_compressor/`.
