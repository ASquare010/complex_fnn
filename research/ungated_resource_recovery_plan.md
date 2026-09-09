# H075 - Bounded operational recovery of H074 resource qualification

Freeze before construction probes or recovered workers. The preceding turn made
PROGRESS: 17 completed H074 resource workers, independently audited exact partial
comparisons, and an explicit pre-update import failure. H073 fitting still earns
full-model resource qualification, which remains incomplete. This protocol
specifies one operational recovery; it does not alter H074's scientific design.

## Evidence and diagnostic scope

H074's same-width GELU block_inner worker failed in AdamW construction while
CPython tokenized an imported Torch configuration module. It produced no probe,
update or initial-state artifact. The cause is unknown. Its process exited with
code1; OS inspection found no remaining H074 worker. The other17 completed
workers and all failures remain byte-for-byte preserved. The three matched-width
workers were never launched. The prior standalone AST/tokenizer probe passed
but did not reproduce the failed import context.

Run exactly three new construction probes in separate sequential processes:
(1) CPU two-weight AdamW construction, then (2) and (3) the original full-size
same-width GELU seed17 initialization, CUDA placement and AdamW construction.
Reuse H074 make_model, parameter_groups and original import path. Do not preload
Dynamo, patch dependencies or change any environment/source. All probes have
zero forwards, backwards, optimizer updates or corpus targets. Check empty
optimizer state, finite initial weights, exact full-size initial weight/group
signatures versus the completed H074 same-width none worker, and original
software/hardware/source hashes. Record PID/UTC, exception and terminal logs.

All three must pass to dispatch recovery. Passing only shows these fresh process
constructions succeeded; it is not a cause diagnosis or a claimed repair. Stop
on any failure. No repeated probes, package reinstall, cache deletion, dependency
patch, machine setting change or scientific retry loop is allowed in this round.

## Four-cell recovery and unchanged allocation

If all three probes pass, run exactly four new sequential CUDA workers:

1. gelu_same_block_inner: one explicit retry of H074's pre-update failure.
2. gelu_matched_none: first launch of the missing native cell.
3. gelu_matched_block: first launch of the missing block-checkpoint cell.
4. gelu_matched_block_inner: first launch of the missing inner-checkpoint cell.

Call the byte-frozen H074 worker function directly, changing only its artifact
root binding. A new protocol contains all103 source hashes (100 prior and three
new orchestration/probe files), the unchanged H074 plan hash and this recovery
plan hash. The original token stream and environment must match. No architecture,
initialization, rate, precision, batch, sequence, optimizer, checkpoint option,
warmup, timing method, diagnostic, training step or decision gate is changed.
No completed H074 cell is rerun, copied over or overwritten. New files reside
under results/ungated_resource_recovery_v1; the old failed folder remains empty.
Stop immediately on another worker/runtime/finite failure; no second retry.

Four workers add80 updates,163,840 synthetic training targets and8,192 probe
targets. Together with H074's17 completed cells, the complete21-cell allocation
would be420 updates,860,160 training targets and43,008 probe targets, exactly the
original budget. H074's failed process contributed zero updates. Record22 total
worker launches if recovery completes:21 successful plus the original failure;
only one cell has a second launch. The three construction probes are separate
and add no scored targets or updates. No language data or cache is touched.

## Combined analysis and unchanged decisions

Map the17 old cells and four new cells explicitly to their artifact roots and
pinned process/checkpoint hashes. Recompute the original within-form comparisons
and apply the original gate function without edits. Same-width inner compares
with the old same-width none run; matched-width modes compare within the new
cohort. Require all21 completed cells and all14 comparisons to be exact. The
independent CPU audit reloads all21 initial/final checkpoint pairs, verifies
moments, signatures, data/RNG, actual counts and recomputes selection/gates.

Use each form's minimum allocated-memory mode, ties median time then original
mode order. Candidates require >=70% FFN reduction, finite evidence, <=110% of
EACH full control's minimum allocation and <=125% of plain's update time in the
same mode. These thresholds are exactly H074's. Show every old/new measurement
and its origin. A time gap separates cohorts; this is a short single-device
qualification, not a causal timing or statistical speed superiority claim.
Passing earns only a separately frozen short language screen. An incomplete
recovery grants no promotion. Prior negative recipes remain closed.

Preserve H074 final/failure/protocol/source/support archives, all17 complete
worker artifacts and the empty failure directory. Preserve all prior active
sources/configurations/tests, plans and798 fitting checkpoints. No active model
or recipe is introduced. The prior five isolated H074 tests and108 active tests
are not rerun solely for orchestration. Update current state and ledger with the
actual diagnosis/recovery outcome, including any failure. The full research goal
still requires language quality, convergence, three seeds, scaling, broader data,
strong prior-art comparisons and practical utility.

References: [H074 frozen design](ungated_resource_plan.md),
[H074 incomplete results](ungated_resource_results.md),
[H073 fitting](ungated_fit_results.md).
