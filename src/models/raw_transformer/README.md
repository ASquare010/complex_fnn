# Raw-token baseline

Standard causal GELU transformer,23,077,376 trainable parameters. Up to2048
raw history tokens plus512 current positions, matching source text for the
compressed-memory experiment. Frozen encoder is not loaded. FP32 activation
checkpointing is disabled for speed. Direct entry point: training.py.

See docs/three_model_comparison.md for matched data, commands and limitations.
