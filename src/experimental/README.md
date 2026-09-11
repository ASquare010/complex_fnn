# Promising memory-efficient models

- `src/core/training_memory.py`: exact buffered classifier loss and four-block
  checkpoint-input offload. H156 passed fresh 800-update training across six fixtures.
- `checkpoint_compression.py`: unchanged H157 FP16 checkpoint-input codec.
  H158 passed isolated memory/gradient checks; H159 passed 1,080-update paired timing.
  About24% less allocated GPU memory at approximately native update speed in H159.
  **Experimental approximate gradients. Fresh-training qualification is incomplete.**

`checkpoints/manifest.json` indexes local complete model/Adam/sampler checkpoints,
with source paths and SHA256 hashes. Binary checkpoints remain local and git-ignored;
the manifest, code and essential research documentation are committed. Files include
full training state, not only model weights. Load only trusted checkpoint files.

The compression module is a byte-for-byte copy of the tested H157 codec. Its API is
`compress_inputs(model, half=True)` with `buffer_model_loss(model, tokens, targets)`.
It supports the ordinary Transformer with whole-block checkpointing and FP32 inputs.
Forward remains FP32; saved inputs round through FP16 for backward recomputation.
Do not describe this as exact autograd. FP16 overflow is possible at large scales;
this prototype is qualified only for the documented finite-range workloads.

See `research/PAUSED_HANDOFF.md` before resuming. No background training is intended
while paused. This is a memory optimization, not a novel activation or parameter win.
