# Higher compression and external latent memory

Status: **bounded sweep complete; Step 2 closed**, 2026-10-05. Span 32 is the selected
handoff to [Step 3](step_3_memory_llm.md). Earlier prospective sections are the
registered historical protocol; completed measurements appear below.

The intended system is a small reasoning model plus external information storage:
text -> encoder -> stored latent bundles -> query/retrieval -> reasoning model.
The reasoning model fetches relevant information instead of placing every stored
document into attention. This is the user's CPU/RAM analogy, not a claim that
FFNs only store facts or that retrieval eliminates the need for learned reasoning.

The autoencoder analogy is useful: latent diffusion performs its generative work
in a pretrained autoencoder's latent space. Our encoder is deterministic, not a
probabilistic VAE. Reconstruction demonstrates recoverability, not that a separate
reasoning model already understands the code. See the primary
[latent diffusion paper](https://arxiv.org/abs/2112.10752).
External-memory language modeling also has relevant precedent in
[RETRO](https://arxiv.org/abs/2112.04426); the broad idea is not a novelty claim.

## Capacity and attention accounting

At eight source tokens per memory position, one million memory positions could
represent approximately eight million source-token positions, subject to rounding,
width, positional support and learned quality. This capacity extrapolation has
not been tested at that scale. Full self-attention over those memory positions
has roughly 1/64 of the pair count of eight million token positions. A single
output query attending to memory has roughly 1/8 as many positions to visit.
Neither number is a measured whole-system speedup. Encoding, FFNs, retrieval,
projections, memory traffic and generation still cost computation.

Current reconstruction expands memories back to token positions. The eventual
reasoning core must consume compact memories directly to obtain the intended
attention savings. Model weights and stored payload bytes remain separate costs.

## Registered first compression pilot

Preserve the selected 8-token model and inference notebook. Test span 16 first,
with span 32 and 64 as later candidates. Keep width 256, four encoder blocks,
one parallel decoder block, context 256 and vocabulary 4096. Do not weaken the
decoder simultaneously; changing both factors would obscure the result.

Transfer shared weights from `dump/position-pattern-5000-clean-v1/last.pt`;
reinitialize both span-dependent compression/expansion maps and AdamW. Train
2,000 additional updates, seed 17, microbatch four with four-step accumulation,
ordinary CE, cosine learning rate 3e-4 to 3e-5, weight decay 0.01 and clipping one.
Use 50k original natural paragraphs, 25k uniform-token rows (seed 219), and 25k
small-alphabet/repetition rows (seed 217). Pattern/uniform validation uses seeds
218/220, removing exact training overlaps. Full natural development every 500
updates and at the final checkpoint. Preserve original tests as historical
confirmation; do not repeatedly use them for model selection.

This is a transfer feasibility pilot, not a matched-total-compute comparison or
a proof of the maximum attainable compression. The parent already received
5,000 updates; the new span maps must relearn reconstruction. Failure at this
budget can reflect optimization as well as information capacity.

| Tokens per vector | Vectors for 256 tokens | Memory self-attention pairs | Full model parameters |
| ---: | ---: | ---: | ---: |
| 8 (selected) | 32 | 1,024 | 4,147,200 |
| 16 | 16 | 256 | 5,195,776 |
| 32 | 8 | 64 | 7,292,928 |
| 64 | 4 | 16 | 11,487,232 |

These parameters increase because the current span maps scale with span length.
Their matrix-multiply work per full input is approximately constant: fewer groups
offset larger group maps. Report actual timing and peak allocated memory, alongside
latent features/bytes and theoretical attention pairs. This pilot does not include
a consuming LLM or establish its memory/quality gains.

Use at least 99.9% token accuracy and 95% exact sequences across the measured
development suites as the practical recovery gate; preserve failures. Only a
candidate passing recovery checks should replace the current notebook model.

## External memory details for later implementation

- Do not rely on exact floating-point vector equality for finding repeated facts.
  Context and position affect the code; the same statement elsewhere can differ.
- Deduplicate identical source chunks using content hashes and encoder version.
  Semantic deduplication must preserve different names, numbers, negation and dates.
- Keep retrieval keys separate from reconstruction payloads unless a retrieval
  objective establishes that distances in the latent space reflect relevance.
- Store ordered bundles with source identity, offsets, token lengths and model
  version. A single vector need not be independently decodable from its neighbors.
- Train query selection and direct latent consumption; measure retrieval recall,
  answer quality, end-to-end latency and GPU memory. The ability to decode source
  text is a prerequisite, not proof of successful reasoning or retrieval.

The research target is reliable information per active attention position under
a measured total-resource budget, not the largest nominal compression ratio.

## Initial measurements

The completed span-16 transfer pilot passes all measured development recovery
checks at 2,000 updates. Span 32 is the next same-budget pilot, independently
initialized from the same span-8 shared weights rather than span-16 weights.

An isolated CUDA BF16 attention benchmark (batch 1, four heads, head width 64;
20 warmups and five groups of 100 calls) measured median event times of 0.7412 ms
at 4,096 positions and 0.0304 ms at 256 positions: about 24.4x, not the theoretical
256x pair reduction. Smaller 512/256/128/64-position calls were all near 0.03 ms,
where dispatch and execution overhead limit measurable gains. This is not a
consuming LLM, and excludes encoding, retrieval, projections and FFNs.
Evidence: `records/compression-attention-kernel-v1.json`.

## Completed bounded study

All three transfer pilots ran for 2,000 additional updates from the same frozen
8-token shared weights, with freshly initialized span maps and optimizer.

| Span | Development exact | Uniform exact | Pattern exact | Model parameters | Peak allocated MiB | Update seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1,000/1,000 | 256/256 | 512/512 | 5,195,776 | 260.17 | 388.20 |
| 32 | 1,000/1,000 | 256/256 | 512/512 | 7,292,928 | 296.62 | 412.81 |
| 64 | 31/1,000 | 0/256 | 39/512 | 11,487,232 | 367.61 | 451.36 |

Both passing candidates also reconstruct all 256 existing Unicode/formatting
stress inputs exactly. Span 32 additionally recovers 1,000 freshly selected
natural paragraphs exactly, with no overlap in document hashes against previous
selected data. The checkpoint was frozen before this confirmation. Its encoder
has 4,798,720 parameters; full model has 7,292,928. No search beyond these ratios
was performed, and failure at 64 does not establish an information-theoretic limit.

At span 32, a 256-token window becomes eight 256-feature vectors: 4,096 BF16
payload bytes plus length metadata, versus 131,072 bytes for the width-256 token
embedding tensor. The fresh corpus averages 27.17 source tokens per vector after
rounding. Its latent payload plus lengths is 1,742,144 bytes, compared with 328,357
UTF-8 bytes. This is feature/position compression, not raw-string byte compression.

Verified artifacts: `dump/compression-span16-v1/` and `dump/compression-span32-v1/`
contain checkpoints and encoder exports. The single inference notebook supports
8, 16 and 32; the default remains 8. Sources and protocols are frozen in each run.

Final span-32 memory controls: correct 64/64 exact; zeroed/shuffled 0/64.
An encoder-only timing check (batch four, 256 tokens/example, GPU-resident inputs,
BF16, rotated model order) measured medians 9.90/9.39/9.61 ms for spans 8/16/32.
Individual groups ranged roughly 4.2–10.9 ms, so small differences are inconclusive.
The first timing run overlapped notebook activity and is explicitly marked invalid;
only the later non-overlapping run is reported. No encoder speedup is claimed.

Records: `compression-span16-v1.json`, `compression-span32-v1.json`,
`compression-span64-v1.json`, `compression-span32-confirmation-v1.json`,
`compression-span32-controls-v1.json`, `compression-export-check-v1.json`,
`compression-encoder-timing-v1.json` and notebook checks under ignored `records/`.
