# Canonical native256 token memory

This is the single active compressed-memory reader. It uses256-wide raw-token
embeddings, a256-wide eight-layer/four-head transformer, and the existing256-float
memory carriers. There is no dense memory-width projection. Two local reads run
after blocks2 and5 (zero based). Total trainable reader parameters:23,209,216.
The frozen9,426,432-parameter encoder remains separate and unchanged.

A completed64-token block becomes one global memory position. BOS plus at most63
preceding raw tokens form the active window. The next generated token completes
the block, the frozen encoder encodes it once, existing low-bit packing inserts
the64 literal IDs, and raw context resets to BOS. The payload is read before
normalization/arithmetic and reinserted into existing V slots at the read layers.
There is no second persistent history bank. The local reader still scores
individual old-word candidates;64:1 global positions does not mean64x total compute.

Memory input uses non-affine RMS normalization followed by256 learned channel
scales (initial .02) and256 learned type offsets (initial zero). These operations
preserve width and do not mix coordinates. This avoids feeding high-amplitude
codec vectors directly alongside small fresh token embeddings. This is not a
learned256-to256 or256-to512 matrix. The two narrow read layers retain the same
query-selection, pooled lexical context and copy mixture used by the completed
wide512 study. Literal IDs are explicit metadata; their exact recovery does not
prove that semantic neural compression alone is lossless.

Keeping eight blocks and the exact reader budget requires total FFN hidden width
38,062, distributed across the eight blocks. This is a new architecture, not a
continuation of the completed512-wide weights. Quality and speed equivalence are
unproven. No inference vocabulary-table cache is added in this change.

## Train in PowerShell

```powershell
Set-Location D:\Git\complex_fnn
& .\.venv-cuda64\Scripts\python.exe .\src\models\token_memory\training.py `
  --name token-memory-native256-v1 `
  --data .\dump\data\unified-chat-v4 `
  --cache-dir .\dump\token-memory-64-cache-v4 `
  --epochs 2 --batch-size 32 --microbatch 4 --eval-every 1000
```

Fresh seed47; exactly48,762 original updates /361,357,104 assistant-label exposures.
The original effective32 sample orders and assistant targets are retained.
Length-first microbatch grouping changes only physical grouping. The objective
remains original assistant CE + .05 query-slot CE + .02 critical-source CE, using
only pre-existing TRAIN labels. AdamW, peak3e-4/minimum3e-5, warmup200, decay .01,
clip1 and FP32 match the completed comparison. Cache and corpus are read only;
there is no per-update encoder work. TF32 and reduced precision are disabled.

Full original development evaluation (7,831 segments /1,859,388 labels) runs every
1,000 updates and at completion. Checkpoints save every250 updates. Console output
is readable CE/accuracy/progress/ETA; detailed JSON stays in files. Ctrl+C finishes
the current update and checkpoints. Resume without resetting optimizer/schedule:

```powershell
& .\.venv-cuda64\Scripts\python.exe .\src\models\token_memory\training.py `
  --resume .\dump\token-memory-native256-v1 --microbatch 4 --eval-every 1000
```

Resume verifies format, recipe, source hashes, input hashes, sampler-file hash and
cursor. Old wide512/native256 checkpoints require their archived implementations.
The new run refuses existing output folders unless explicitly resuming. Use
`--inspect` for architecture only, `--check-data --name unused` for CPU input
verification, or `--stop-after N` for N new updates without shortening the schedule.
No new experiment is launched by the cleanup itself.

## Generation

```powershell
& .\.venv-cuda64\Scripts\python.exe .\src\models\token_memory\generation.py `
  --run .\dump\token-memory-native256-v1 `
  --prompt "User: Where is the notebook?`nAssistant:" --max-new-tokens 128
```

The model prompt is serialized text; use the existing User/Assistant format.
Generation uses the original frozen encoder, genuine64-token rollovers and
existing-size KV arrays. FP32 carrier bits must be retained.

## Records

See [cleanup and new-model protocol](../../../docs/token_memory_native256_cleanup_20261020.md)
and [completed wide512 results](../../../docs/wide512_completed_comparison_20261020.md).
The completed study had a large KV saving and better familiar fact recovery, but
lower language accuracy and no latency win. Its results do not apply automatically
to this fresh256-wide model.
