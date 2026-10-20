# Wide512 fresh full-budget comparison

This is a new two-arm experiment. It does not resume or extend any completed pilot,
and it does not alter the native256 run or original models, checkpoints, corpus,
encoder, or caches. The reader budget is exactly 23,209,216 trainable parameters;
the frozen 9,426,432-parameter encoder is separate, as confirmed by the user.

## Architecture and evidence limits

The separate `src/models/token_memory_64_wide512` model uses width512, eight global
transformer layers/eight heads, and two width256 local query readers. The compressed
arm keeps one global attention position per64 old tokens and a raw64 window.
The raw1 control uses one global position per old token, the same parameter tensors,
the same local reader, the full512 embedding residual, and the shared adapter. Both
are fresh seed47 with identical initial tensors and RNG. Neither starts from a
trained checkpoint. The adapter's initial scale is calibrated to raw embeddings;
this is a hypothesis, not an established explanation of the old quality gap.

The existing wordbits cache adds exact original token IDs to low mantissa bits of
each256-number code. This is literal metadata, not evidence that the learned codec
is lossless or that storage is compressed64:1. Local dictionary operations inspect
64 payload words per code. The claimed compression concerns global attention
positions/KV entries, and must not be described as a64-fold total compute saving.
No additional history bank or KV fields are introduced. Prior preflight records
and their source bytes remain immutable. The original512 reference includes older
compressed memory and is not a pure uncompressed baseline.

## Fixed data, objective, and optimizer

Each arm receives two ORIGINAL epochs:780,186 segments/epoch,180,678,552 assistant
labels/epoch,24,381 updates/epoch,48,762 updates and361,357,104 assistant labels total.
Effective batch32, with26 examples in each epoch's last update. Original seed47/48
orders, assistant masks, global label denominators, LR schedule and cursor are
preserved. There are no extra QA examples, training texts, or substituted targets.

Objective: original global assistant CE +0.05 query-slot CE +0.02 critical-source CE.
Auxiliaries come only from the original TRAIN labels. Their core, embedding and old
pointer features are detached; they train the selector/entity gates. Full exposure
totals per arm:78,540 query segments/157,080 query terms;62,826 source segments/
125,644 source terms, including62,822 critical name and62,822 critical value terms.
The source-segment count differs from the name/value count because some targets
cross original segment boundaries. Generic absolute word coordinates are checked
separately for1-word and64-word memory positions.

FP32, TF32 disabled, AdamW(lr3e-4, betas.9/.999, eps1e-8, weight_decay.01), gradient
clip1, warmup200 original updates and cosine minimum3e-5 over48,762 updates.
The GPU process has a70% allocator limit and uses the single-worker lock.

Compressed arm: microbatch8, stable length-first physical grouping inside each
original effective update. Raw1: microbatch4, stable memory-count-first grouping.
This changes within-update execution order, not the update's example multiset,
loss denominators or sampler cursor. No dropout is used. Actual effective32
preflight gradients across these layouts agree within2.24e-8. Each original and
physical order is logged and independently checked; neither layout is represented
as the original physical execution order. Cached-body timings exclude Adam,
encoder/cache I/O, validation and saving, and are not end-to-end speed claims.

## Gates and immutable provenance

1. Complete CPU/GPU/layout preflights with zero production updates.
2. Export this NEW run's manifest/source archive, original orders and BOTH fresh
   initial states. Freeze all training, helper, evaluation, verification, test and
   protocol source bytes included in the manifest.
3. Independently check every original segment/mask, every literal payload/cache
   block, all48,762 original baseline orders/labels/LRs, both fresh initial states,
   auxiliary-label audit, both coordinate modes and protected-file hashes.
4. Run exactly ONE actual update per arm, save immutable `first.pt` snapshots,
   then independently check original/physical orders, auxiliary denominators,
   original assistant-label count, recipe/cursor/LR and every Adam step/moment.
5. Only after both first-update audits pass, resume each for its remaining48,761
   NEW updates. Resume requires exact format, recipe, sources, saved RNG and cursor.
   Completed checkpoints cannot be resumed through this entry point.

Checkpoints save every250 updates and at evaluation/stop/final boundaries. The
initial step-one safety run skips initial validation; periodic full original-dev
validation runs every2,000 updates and at completion. BOTH arms are scored with
their actual copy/generator distribution, never the old raw1 generator-only path.
No source edits are permitted during a live worker or after this run is exported.
Any source correction requires explicit provenance handling before another update.

## Prespecified evaluation

After checkpoint audits, score full original development set (7,831 segments/
1,859,388 labels), natural free answers, changed-value/query pairs, missing and
blocked controls, conditional completions, all seven legacy counterfactual groups,
historical cases, reasoning and the notebook four-way binding suite. Compare strict
value-plus-EOS and all-four correctness; lower CE or isolated token matches are not
factual recovery. Verify saved metrics, frozen-case derivation and public generation
replays independently. Include original/conditional/blocked/shuffled transfer suites
with exact donor replay. Paired document/unit intervals describe these fixed
checkpoints, not training-seed uncertainty.

Reserved novel-literal student TEST inputs remain closed until BOTH arms have
48,762 audited updates and361,357,104 original assistant-label exposures, all Adam
steps48,762 and cursor(epoch2,batch0). Then evaluate enabled and blocked conditions,
verify public outputs and compare each held-out group/form. Reserved scores cannot
select an early checkpoint or trigger automatic training extension.

No architecture adoption or quality-parity claim follows from a preflight, speed
sample, pilot or familiar-only retrieval. The goal remains unmet until the full
matched language/fact/reasoning comparison supports it.

## Commands

Use the CUDA environment Python executable with `tools/run_wide512.py`. Actions
`prepare`, `data`, `first`, `training`, `results`, verification actions and `compare`
are CPU-only. GPU actions use one worker. The sequence is:

```powershell
& .\.venv-cuda64\Scripts\python.exe .\tools\run_wide512.py prepare
& .\.venv-cuda64\Scripts\python.exe .\tools\run_wide512.py data
& .\.venv-cuda64\Scripts\python.exe .\tools\run_wide512.py train --arm both --stop-after 1 --skip-initial-eval
& .\.venv-cuda64\Scripts\python.exe .\tools\run_wide512.py first --arm compressed64
& .\.venv-cuda64\Scripts\python.exe .\tools\run_wide512.py first --arm raw1
& .\.venv-cuda64\Scripts\python.exe .\tools\run_wide512.py train --arm both --resume --stop-after 48761
```

After each arm completes, run `training --arm ARM`, then `evaluate`, `notebook`,
`results`, `transfer`, and `verify-transfer` with `--arm ARM --step48762` (spell the
flag as `--step 48762`). Only after BOTH full-training audits pass, run `reserved`
and `verify-reserved` for each arm, then `compare --step 48762`. Never launch a
second GPU action while the first worker is alive.
