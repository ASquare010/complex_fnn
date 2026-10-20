# Native-256 reader experiment

## Question

The frozen compressor emits one 256-dimensional code for each completed
64-token block. The completed raw64 reader normalizes that code, projects it
from256 to512 dimensions, adds a memory-type vector, then applies its512-wide
attention stack. This experiment tests whether that adapter is a harmful
interface: a fresh reader runs at width256, so both a frozen old-memory code and
each fresh token embedding enter attention at the same native width.

## Reader

- One frozen code remains one attention position for each old64-token block.
- The code enters the first attention block directly: no code RMSNorm,
  256-to-512 projection, or additive memory-type vector.
- Fresh token IDs use a new learned256-wide embedding table.
- The reader has21 causal layers, width256, four64-wide heads, tied input/output
  embeddings, and the existing raw64 packed causal mask.
- Its FFN widths sum to32,467. The resulting trainable reader count is exactly
 23,209,216, matching the completed512-wide reader.
- The frozen codec and the original FP32 code cache are reused read-only. This
  test does not retrain the encoder or claim that a code carries64 separately
  addressable token positions.

The frozen encoder already receives relative positions while encoding its64
tokens and can store positional distinctions in code dimensions. A single code
still supplies only one attention address. Adding a block-position vector would
identify the block in the sequence; it would not create64 independently
selectable addresses within that vector. A future multi-code codec is a separate
capacity experiment.

## Training contract

A full run, if started, uses fresh seed47 and the original two-epoch corpus,
order,32-segment effective batch, assistant-token CE, AdamW schedule and frozen
codes. It is not resumable from the512-wide checkpoints. The completed models,
data and cache contents remain unchanged.

## Preflight

`dump/token-memory-64-native256-preflight-1792164038079915100.json` passed with
zero optimizer updates. It re-encoded checked frozen blocks, verified packed
raw64 logits against isolated raw64 calls (max2.3842e-6) and all parameter
gradients (max3.3528e-8), and confirmed the initial reader tensors unchanged.
The native256 packed body samples were1.120,1.018 and1.022 seconds per effective
batch32 at microbatch4, excluding optimizer, cache creation, validation and
checkpointing. Peak heavy-batch allocation was1.917GiB, with3.719GiB reserved
including Adam-sized moments. This is not full-training throughput or a quality
result.
