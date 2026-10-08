# Selected models — Steps1 and2 complete

Step1: **Branch Sigmoid**,8.65M parameters,webNLL3.820069/chat2.411506.
Step2: **plain residual attention + Branch Sigmoid + RoPE**,64tokens/vector,
context512,7.33M encoder/10.46M total parameters. Saved final model achieves
999/1000 short and202/206 packed exact reconstruction on reused validation.
This closes the selected-model handoff,not the original ambitious research targets.

- [Handoff and limitations](docs/selected_handoff.md)
- [Latest compressor results](docs/residual64_long.md)
- `notebooks/branch_sigmoid_inference.ipynb`:saved Step1 inference.
- `notebooks/context_encoder_inference.ipynb`:saved Step2 inference.
- `notebooks/residual64_training.ipynb`:optional15000-update continuation in
  D:/Git/latent-text-compressor,configured but not launched.

```powershell
uv run --no-sync python -m main inspect
uv run --no-sync python -m main reconstruct "Text to remember"
uv run --no-sync python -m main encode "Text to remember" --output dump/memory.pt
```

Defaults live insrc/config. Weights/data/records stay local and ignored;fresh clones
do not include trained weights. Historical source snapshots and negative results
are retained. No CurveFFN/custom-kernel/Step3 integration is active. Base transformer
modules remain because the selected model uses them. No automatic model training.
