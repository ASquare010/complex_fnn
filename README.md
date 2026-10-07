> Branch Sigmoid + RoPE encoder/decoder training is complete (7,000 updates; 1,000/1,000 exact validation reconstructions): [protocol and status paths](docs/branch_rope_training.md). See the recorded result and its limits.

# Selected models

The active repository contains two models:

- **Step 1 — Branch Sigmoid:** 8,654,208 parameters; four gated squared-ReLU feature groups. Saved 10,000-update model: web NLL 3.820069, chat NLL 2.411506.
- **Step 2 — span-32 Branch Sigmoid compressor:** updated encoder and reconstruction decoder. New weights are required; the old CurveFFN reconstruction results do not describe this architecture.

## Layout

```text
src/
  config/
    branch_sigmoid.json
    branch_data_identity.json
    position_compressor.json
  models/
    branch_sigmoid/       # LM, loading, verified data, training
    position_compressor/  # encoder, decoder, codec, training
    components.py        # shared attention/norm/initialization
  main.py
  settings.py
  storage.py
notebooks/
  branch_sigmoid_inference.ipynb
  context_encoder_inference.ipynb
```

Both models now use Branch Sigmoid. See [migration notes](docs/branch_compressor_migration.md). Version 0.3.0 removes the legacy CurveFFN implementation.

## Use

Run from the repository root. Always use `--no-sync` to preserve CUDA PyTorch.

```powershell
uv run --no-sync python -m main inspect
uv run --no-sync python -m main generate "Once upon a time" --tokens 32
uv run --no-sync python -m main encode "Text to remember" --output dump/my-memory.pt
uv run --no-sync python -m main reconstruct "Text to remember"
```

The `encode` and `reconstruct` commands require new weights in `dump/compression-rope-v1/`; no trained Branch Sigmoid compressor is supplied.

Generation is a small-model smoke/demo interface, not a claim of chatbot ability.
Step 1 uses at most 256 recent tokens. The two selected models are independent;
there is no automatic history retrieval or memory integration.

The retained training implementation supports explicit new Branch runs with
`python -m main train --config ...`; cleanup does not launch training. Historical
optimizer resumption requires original sources and recipes. The current loader
supports inference from the exact saved winner and checks its checkpoint hash.

The Step 1 checkpoint remains usable. The two historical CurveFFN checkpoint paths below have been deleted at the user's request; historical scores remain recorded:

- `dump/ffn-final-10000-s461/branch_sigmoid/last.pt`
- `dump/compression-span32-v1/encoder.pt`
- `dump/compression-span32-v1/last.pt`

[Cleanup details](docs/selected_models_cleanup.md) ·
[Historical language results](docs/leaderboard.md) ·
[Step 2 evidence](docs/step_2_results.md)

Datasets, records and historical checkpoints are local/ignored. Historical source
snapshots and result documents are evidence, not active implementations. Some
older weights were explicitly pruned in a previous storage cleanup.
