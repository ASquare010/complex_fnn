# H112: fresh quality and whole-job memory replication

Status before execution: CONDITIONAL HYPOTHESIS. Allocation requires H111's
T512 evaluation component to pass its frozen score/memory/time gates and its
independent audit. If no policy qualifies, do not launch training. H110's
fidelity result stays stopped and its original endpoints remain untouched.

## Question and motivation

Does the loss-chunk training recipe preserve or improve language-model quality
while saving at least15% of the whole measured job's tensor allocation, when
both arms receive the same qualified memory-aware evaluation? H110 motivates
this with one T512 seed that improved NLL4.32% while reducing whole-job allocation
19.26%. That observation is hypothesis-generating only. This prospective study
uses a one-sided quality criterion and fresh seeds, not a retroactive exemption
from H110's two-sided execution-fidelity test.

## Frozen comparisons and budget

Use the unchanged H110 narrow GELU model/configuration: width384, hidden456,
eight layers, six heads, vocabulary4096, 9,099,648 total parameters and
2,801,664 FFN parameters. Two training policies: block checkpointing with native
mean CE, and the same block checkpointing with512-token classifier/CE chunks.
The maintained helper and frozen H101/H110 worker are reused without changing
their implementation. Explicit adapters select the corpus, output root and
H111-qualified evaluation policy; both arms get the identical evaluator.

Use B8/T512, training seeds61/73/89, and both hashed existing corpora:
`data/wikitext2_v1` and `data/tinystories_v1`. Each corpus retains its own frozen
train-only tokenizer and split, with vocabulary4096. Compare NLL only within
each corpus; do not pool raw losses across tokenizers. These are development
splits with prior project use, not a new held-out test. No test split access.

Twelve fresh trials of800 updates =9,600 updates and39,321,600 token
presentations. Each trial gets constant LR0.0006, AdamW(.9,.95), epsilon1e-8,
matrix decay0.1, global clip1, no width-calibrated initialization or rate scaling,
BF16 autocast/FP32 weights and Adam, TF32 off, and four CPU threads. No schedules,
rates, chunk sizes, widths or seed selection. Rotate policy order by seed and
corpus. Use matching named-tensor initial states and sampler seed=train_seed+10000
within each pair. Whole CUDA corpus caches count toward memory.

Use the H111 T512-selected evaluator for the full job in both arms, including
the fixed16-batch checks at0/100/200/400/800 and complete validation at800.
Retain all H110 instrumentation: every update's loss, preclip norm, synchronized
timing and allocation; fixed activation/gradient diagnostics; checkpoints at
100/200/400/800; final model/optimizer/sampler state. An explicit evaluator
adapter adds the200 checkpoint so every reported score can be independently
rescored; it checks the frozen six-call evaluation schedule. First20 updates are timing
warmup. Separate training, validation, preparation and whole-job allocation.
The inherited whole-job peak accounts for persistent state, sampler, diagnostics
and validation across peak resets. Report driver/context exclusion explicitly.

## Primary evidence and fixed decisions

Independent native unchunked rescoring of all saved states is outside the
measured job and performs no optimization. Every streamed evaluation score must
agree with its native score within0.01% relative. The primary quality metric is
the independently audited native full-validation NLL at800, not a selected
checkpoint or training loss. A score-adapter mismatch invalidates the affected
comparison until diagnosed; never use a favorable alternative evaluator.

For every seed in each corpus, require:

- candidate full native NLL <=1.01 times the matched reference;
- candidate whole-job allocated peak <=0.85 times the matched reference;
- candidate median synchronized update time <=1.25 times the reference;
- all training quantities, model weights and Adam states finite.

Report every seed, mean, median, sample variance, min/max and paired ratios
within each corpus. A corpus qualifies only if all three seeds pass all gates.
The two-corpus component qualifies only if both corpora qualify. Do not label
a three-seed mean a strong statistical confidence statement. Better NLL is
allowed without a lower bound; it is still not evidence of a new activation.

Finish the fixed12-trial grid despite ordinary quality, memory or speed gate
failures so both corpora and all seeds are visible. Stop immediately on a
runtime/source/data failure or nonfinite training, preserving all completed
work and the interrupted trial. No automatic retry loop. Do not expand the
budget, tune a failed policy or resume H110 under this label.

## Runtime and scope

Use the original absolute UV-managed Python3.12.9 with the same installed
packages, malloc allocator, hash seed107, bytecode bypass and dependency preload
before CUDA. Prior native failures remain unresolved; a source/runtime failure
gets explicit diagnosis. Only one active GPU worker at a time. Freeze this
plan, all adapters, source dependencies, selected H111 policy, corpus manifests
and prior evidence before any qualification or training.

This is an execution comparison of an existing dense FFN. No weight count,
activation, architecture or inference behavior is changed. A passing result
would be a scoped800-update, two-corpus memory/quality component. It would not
establish a new SOTA FFN, a novel mechanism, convergence-optimal training or the
separate70%-parameter architectural goal. Those broader requirements persist.

Preserve full histories/checkpoints locally and compact machine-readable
records, plots, failures, source and an independent verification receipt.
No changes to active model folders, variants, recipes or maintained training
defaults are required to conduct this isolated experiment.
