# Promising memory-efficient models

- `src/core/training_memory.py`: exact buffered classifier loss and four-block
  checkpoint-input offload. H156 passed fresh 800-update training across six fixtures.
  [H163](../../research/exact_offload_long_scale_results.md) also passed 800 updates on a
  22.2M-parameter model across three WikiText2 seeds: 18.1% lower allocated GPU peak,
  1.2–5.9% slower updates, maximum validation NLL increase 0.24%.
- `checkpoint_compression.py`: unchanged H157 FP16 checkpoint-input codec.
  H158 passed isolated memory/gradient checks; H159 passed 1,080-update paired timing.
  About24% less allocated GPU memory at approximately native update speed in H159.
  **Experimental approximate gradients. H160 failed fresh-training qualification:** WikiText2 seed113 had 2.26% higher final validation NLL; seed101 failed timing stability. Do not promote as quality-preserving. See [H160 report](../../research/checkpoint_fp16_long_results.md).

[Current qualification](checkpoints/qualification.json) supersedes the historical pause status in the manifest. [Latest exact-helper result](../../research/exact_offload_long_scale_results.md).

`checkpoints/manifest.json` indexes local complete model/Adam/sampler checkpoints,
with source paths and SHA256 hashes. Binary checkpoints remain local and git-ignored;
the original pause manifest and code are committed; newer qualifications are in the worktree. Files include
full training state, not only model weights. Load only trusted checkpoint files.

The compression module is a byte-for-byte copy of the tested H157 codec. Its API is
`compress_inputs(model, half=True)` with `buffer_model_loss(model, tokens, targets)`.
It supports the ordinary Transformer with whole-block checkpointing and FP32 inputs.
Forward remains FP32; saved inputs round through FP16 for backward recomputation.
Do not describe this as exact autograd. FP16 overflow is possible at large scales;
this prototype is qualified only for the documented finite-range workloads.

[Current larger-model checkpoints](checkpoints/h163.json) reference verified result files without duplicating binaries. All three helper seeds and their native controls are indexed. The pause handoff is historical; see [current state](../../research/CURRENT_STATE.md). This is a memory optimization, not a novel activation or parameter win.
