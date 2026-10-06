# Step 2: paragraph reconstruction results

**Step 2 officially complete (2026-10-05).** The selected handoff is span 32:
1,000/1,000 exact development paragraphs and 1,000/1,000 fresh confirmation
paragraphs; 256/256 uniform, 512/512 pattern and 256/256 Unicode/formatting checks.
Encoder: 4,798,720 parameters; full reconstruction model: 7,292,928. The span-32
model used 5,000 parent updates plus 2,000 transfer updates. Full tables and failures
remain below and in [the compression study](latent_memory_compression.md).

This closes bounded reconstruction, not universal losslessness, byte compression
or downstream reasoning. [Step 3](step_3_memory_llm.md) is the new design-stage goal.


These are reconstruction experiments, separate from the language-model leaderboard.
The encoder has seen the paragraph being reconstructed. Lower reconstruction NLL
does not establish better next-token language modeling.

| Study | Data / split | Updates | Reconstruction NLL | Exact text recovery | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Memorization gate | Eight short training examples | 300 | 0.016710 | 8/8 | Passed; training fit only |
| Paragraph pilot v1 | FineWeb-Edu, full 1,000-paragraph validation | 2,000 | 2.327738 | 0/1,000 | Complete; exact-recovery objective not achieved |
| Position compressor v1 | Same full validation; explicit length metadata | 2,000 | 0.058251 (no EOS target) | 911/1,000 | Complete; 99.8854% token accuracy; near-lossless gate not met |
| Position continuation | Same full validation; explicit length metadata | 3,000 | 0.018055 (no EOS target) | 979/1,000 | 99.9730% token accuracy; validation gate passed |
| Position continuation, reserved test | Full 1,000 test paragraphs | 3,000 | 0.036940 (no EOS target) | 975/1,000 | 99.5841% token accuracy; token threshold missed |
| Token-coverage continuation | Same full development set; 50/50 natural/synthetic training rows | 4,000 | 0.005134 | 1,000/1,000 | 100% token accuracy; no new independent corpus test |
| Pattern-coverage continuation | Same full development set; broader synthetic patterns | 5,000 | 0.004100 | 1,000/1,000 | 100% token accuracy; independent synthetic checks below |
| Frozen pattern model, fresh corpus confirmation | 1,000 unique paragraphs from new source documents | 5,000 | 0.004085 | 1,000/1,000 | 100% token accuracy; zero character edits |

The new [position-preserving candidate](step_2_improvement.md) predicts all token
positions directly. Its NLL excludes EOS and uses exact length metadata, unlike
the autoregressive baseline, so NLL values are not identical-objective comparisons.
Compare actual reconstruction and disclose all architectural differences.

The pilot has 3,227,136 parameters and used 221.30 MiB peak allocated memory
sampled during training (418.00 MiB reserved). Seed 17; 32,000 paragraph draws
with replacement across 2,000 updates; microbatch four with four-step accumulation.
Token/character edit rates were 0.666447/0.580567. Correct latent memories beat
zeroed and shuffled memories in the separate 64-example diagnostic, but do not
recover unseen paragraphs exactly. See the model report for all controls,
fact probes, payload sizes, and protocol limitations. No storage-compression
advantage or advantage over a trained uncompressed control has been established.

[Model report](paragraph_baseline_results.md) |
[Memorization evidence](../records/paragraph-memorization-v1.json)

[Pilot evidence](../records/paragraph-pilot-v1.json) |
[Notebook verification](../records/paragraph-notebook-check-v1.json)

The original pilot left the 1,000-paragraph test split reserved; the frozen continuation has now been evaluated on it. Checkpoints, source data and raw
evidence are local and ignored by Git. No universal compression or downstream LLM
compatibility claim follows from this pilot.

The position candidate has 4,147,200 parameters and measured 242.41 MiB peak allocated GPU memory. Its vectors occupy more bytes than raw text. [Position report](../src/models/position_compressor/result.md) | [Interactive notebook](../notebooks/paragraph_compressor_inference.ipynb).

The 1,000-update continuation took 280.95 seconds of training updates and measured 242.11 MiB peak allocated GPU memory. It used the same loss with a declining learning rate. [Continuation record](../records/position-continuation-3000-v1-ce.json) | [Reserved test](../records/position-continuation-3000-test-v1.json). Reserved test data is now evaluated for this frozen checkpoint; it must not become a model-selection set.

The coverage revision adds 1,000 updates (237.77 seconds, 241.76 MiB peak allocated GPU memory). It passed 256/256 held-out random token sequences and 10/10 original diagnostic strings, but only 73/128 new Unicode/formatting stress strings. All improvements remain development results; arbitrary-text near-lossless recovery is not established. [Coverage evidence](../records/position-coverage-4000-v1.json) | [Unicode stress](../records/position-stress-4000-v1.json).

The 5,000-update model recovers 128/128 earlier Unicode stress inputs and 256/256 freshly generated confirmation inputs exactly. It recovers 511/512 independent repeated-pattern sequences (99.9569% token accuracy), and 256/256 new uniform-token sequences. These support near-lossless recovery on evaluated inputs; they do not establish universal losslessness or replace a fresh natural-language corpus test for these weights. [Pattern study](../records/position-pattern-5000-v1.json) | [Fresh synthetic confirmation](../records/position-confirmation-5000-v1.json).

Fresh confirmation now evaluates the frozen 5,000-update weights on 1,000 unique new documents from the same pinned corpus, beyond the original preparation range. One eligible paragraph per document; exact duplicates against all original splits removed; tokenizer unchanged. All 96,245 tokens and 341,281 characters recovered exactly. This is a filtered English-web corpus confirmation, not evidence for every possible string. [Fresh corpus evidence](../records/position-fresh-confirmation-v1.json) | [Final export and controls](../records/position-final-verification-v1.json).

## Higher-compression transfer pilots

These pilots reuse the selected 8-token model's shared weights but reinitialize
span-dependent maps and optimizer; they are not matched-total-compute comparisons.
The selected inference notebook remains on the original 8-token model.

| Tokens per vector | Additional updates | Full development exact | Token accuracy | Parameters | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 2,000 | 1,000/1,000 | 100% | 5,195,776 | Recovery checks passed; transfer pilot |
| 32 | 2,000 | 1,000/1,000 | 100% | 7,292,928 | All four development recovery suites exact |
| 64 | 2,000 | 31/1,000 | 76.3425% | 11,487,232 | Failed recovery gate; not offered in notebook |

The 16-token candidate also recovered 512/512 repetition sequences, 256/256
uniform-token sequences and 256/256 existing Unicode/formatting development
strings exactly. These sets are now development checks, not fresh final tests.
[Protocol](latent_memory_compression.md) -
[16-token evidence](../records/compression-span16-v1.json) -
[Unicode check](../records/compression-span16-stress-v1.json).

The 32-token candidate also passes 512/512 patterns, 256/256 uniform sequences and 256/256 Unicode/formatting strings exactly. [32-token evidence](../records/compression-span32-v1.json).

The frozen 32-token candidate then passed a new 1,000-document corpus confirmation
with 100% exact/token recovery and NLL 0.003116. These documents are disjoint from
all original splits and the previous 8-token confirmation. The new input totals
92,016 tokens -> 3,387 vectors -> 867,072 features (27.17 tokens/vector after rounding).
[Fresh 32-token confirmation](../records/compression-span32-confirmation-v1.json).
The 64-token pilot also failed synthetic suites: 0/256 exact uniform sequences
and 39/512 exact patterns. This is a failure at the registered transfer budget,
not proof that 64-token memories are impossible. The 32-token candidate is the
strongest tested passing ratio, not a proven maximum.

Encoder exports for 16/32 and both notebook workflows were verified. The original
8-token notebook default remains available; choose `TOKENS_PER_VECTOR = 32` to
try the strongest passing pilot. No reasoning-model or retrieval system has been
trained on these memories yet.
