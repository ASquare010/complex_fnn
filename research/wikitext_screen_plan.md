# WikiText-2: frozen equal-budget screen

Hypothesis H032. The selected wide structured FFN can retain its parameter
advantage on Wikipedia text when every recipe receives the same new global
learning-rate search. Freeze this protocol before any WikiText model training.
This is a transfer screen, not a convergence or final test-set result.

## Verified cache

Use data/wikitext2_v1, official Salesforce/wikitext revision
f776294184f13b8ff2337b3841cf9269a6216d1e, wikitext-2-raw-v1. Both downloaded
Parquet files match published LFS SHA256 values. The preparation script identifies
629 train and 60 validation units by top-level article headings. Subsection
headings stay inside the article. Concatenated original row text is preserved;
these are parser-defined units, not an assertion about independent samples.
Normalized exact deduplication, validation first, removes five train duplicates,
leaving 624 train units and all 60 validation units. No near-duplicate claim.

A shared train-only byte BPE has 4,096 entries and zero unknown-token occurrences
in either cache. Append one EOS per article. Train tokens: 3,083,650; validation
tokens: 322,802. All eight stored file hashes match a second independent cache
construction using the archived tokenizer. Source, manifest, tokenizers and
reconstruction details are under results/data and results/verification.

Evaluate all 2,521 complete, nonoverlapping context-128 validation windows,
target positions 1 through 322,688 inclusive. Use 158 batches (157 of 16 and
one of 9); 113 trailing tokens have no complete window. The official test split
has not been downloaded or scored and remains excluded from selection. New
NLLs are not comparable to TinyStories or published word-level WikiText scores.

## Models, optimizer and resource budget

Train four recipes from scratch: full SwiGLU, full GELU, calibrated narrow
SwiGLU, native BlockShuffle. Dimensions: width 384, eight layers, six attention
heads, context 128, tied 4,096-token embedding. FFN widths are 1,024 / 1,536 /
304 / 2,048; BlockShuffle uses eight groups. Full FFN weights: 9,437,184;
compressed weights: 2,801,664 (70.3125% reduction). Total weights are 15,735,168
versus 9,099,648. Exact recipes retain the previous documented FFN-specific
initialization and optimizer multipliers. No post-training condition floor or
custom inference kernels are added to training.

Every recipe receives peak global rates 0.0003, 0.0006, 0.0012; seed 17;
200 steps of batch 16 = 409,600 training tokens per trial. Twelve trials total,
4,915,200 training tokens. Each trial samples about 0.133 training-cache exposures.
Use the unchanged trainer, data window sampler, attention, optimizer and LR
schedule: AdamW (.9,.95), decay .1, clip 1, 20-step warmup then cosine to .1 peak.
Use CUDA BF16/FP32 master weights, TF32 off, four CPU threads. Log every 50 steps
and evaluate the complete frozen validation region using token-weighted loss.

Run rates in ascending order, rotating recipe order by one position per rate.
Use one fresh GPU worker at a time, with a 600-second per-worker limit. No
additional trial or grid expansion is authorized by this frozen screen. An
infrastructure error stops the cohort for diagnosis; preserve all evidence.
A numerical failure is an unsuccessful grid trial, not an omitted observation;
continue only after its cause is verified as numerical rather than infrastructure.
No inference-speed claim comes from sequential training logs.

## Preflight and frozen decisions

Verify source identity for the existing trainer/model/optimizer/data components
against the completed larger cohort. Compare actual common initial tensors,
actual FFN/total parameter counts, optimizer groups and finite backward on real
WikiText tokens for all four recipes. Cache/data hashes and selected windows
must match the frozen reconstruction. Check partial-batch loss weighting.

For each recipe select the finite final checkpoint with the lowest complete
validation NLL. Exact ties choose the lower peak rate. Do not select an earlier
checkpoint within a trial. Record a grid-boundary winner explicitly; it means
the optimum was not bracketed. Equal new tuning does not erase unequal historical
architecture search or prove that any recipe is fully tuned.

BlockShuffle earns a separately specified 800-step cohort with all controls only
if its selected NLL is within 1% of both selected full controls, below selected
calibrated narrow, and its peak allocated memory is no more than 1.1 times each
selected full control. Require >=70% fewer unique FFN weights and finite recorded
losses, gradients and activations. Passing this screen earns further evidence;
it is not a breakthrough. Failure rejects the frozen transfer recipe from this
promotion; inspect learning curves and diagnostics before any changed hypothesis.

Preserve every run's source archive/hash, config, data/tokenizer hashes, history,
checkpoint, gradients, activation diagnostics, memory and timing. Retain a cohort
archive and this plan hash. Report all 12 trials, selected controls, quality/compute
accounting and every failure. The full goal additionally requires multiple seeds,
convergence, appropriately scoped stability, efficient-FFN comparators and verified
novelty. Do not infer those from this screen.
