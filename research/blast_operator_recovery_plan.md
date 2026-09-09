# H080 - One explicit recovery of BLAST operator qualification

Freeze before recovery implementation/execution. H079's original process records
PASS, but the 2026-09-09 integrity audit finds 25 entirely zero-filled files,
including the result/log. Only 18/34 observation JSON files parse, and 24/32
saved tensor archives pass ZIP integrity. The source archive and all 115 frozen
sources are intact; earlier H078 evidence matches. Original worker handles are
absent. The cause is unknown. H079 remains INCOMPLETE_ARTIFACT_INTEGRITY.

This is an explicit exception to H079's no-automatic-retry rule, justified by
unusable artifacts rather than a failed numerical comparison or learning result.
Repeat the SAME 34 checks ONCE in results/blast_operator_recovery_v1. Keep the
original directory and every damaged/intact byte. No source, seed, shape, dtype,
precision, comparison, tolerance, factorization, initialization or proof changes.
There are zero optimizer updates, corpus targets or full Transformer runs.
The previous goal work made PROGRESS by identifying and recording the integrity
failure. Recovery does not declare the research goal achieved.

## Reuse and storage changes

Import the six original test functions with their existing parametrization into
an isolated wrapper. Their bodies and the BLAST operator remain unchanged.
Rebind only the observation root and record writer. The replacement writer uses
binary exclusive-create handles, flush and os.fsync. Read back every JSON payload;
load every tensor file with weights_only=True and compare its full nested payload
with the CPU tensors intended for writing. Verify ZIP integrity and SHA256.
These changes affect artifact persistence, not the numerical computations.
A successful readback is not a diagnosis or a guarantee about the unknown cause.

A collection-only subprocess must report exactly 34 checks before the single
qualification subprocess. This performs no numerical model probes or updates.
Keep four CPU threads and TF32 off; the frozen harness retains six CUDA BF16
and six CPU FP32 FFN cases, twelve CPU FP64 projection cases, four embeddings,
four witnesses, one count/shape check and one finite-difference check.
The original 114 eager/checkpoint tensor comparisons and every frozen numerical
threshold remain required. Count finite-difference forward calls explicitly.

Use one GPU process at a time, UV's existing environment, hidden durable launch,
PID/UTC/return code records, source hashes and fsynced process logs. Stop on any
failure and preserve partial work. No second recovery, numerical repair or
training allocation follows automatically. Across H079/H080 this is two
qualification attempts, including one explicit repetition, not one clean run.

## Independent analysis and final decision

Only after all 34 checks and artifact readbacks pass, independently load all 32
tensor files. Reconstruct the canonical matrices with separate scalar/rank sums;
verify the stored output/gradient pairs, initializer Gram identities/row norms,
all exact embeddings and the Hadamard/rank-collapse certificates. Derive counts
from actual saved factors and require the original 70.3125% FFN reduction.
Check original observation JSON hashes where usable and compare readable prior
tensor payloads with recovered payloads; report this comparison separately.
It cannot reconstruct missing original output or erase H079's integrity failure.

Passing earns only a separately frozen learning comparison at equal parameter
and tuning budgets with full, calibrated narrow and plain controls. It supplies
no full-model optimizer calibration, resource qualification, corpus result,
convergence, arbitrary-data gradient guarantee, state-of-the-art or novelty.
No active model/variant/recipe addition is allowed in this stage.

Pin H079 protocol/source/process/integrity audit and its observed original file
hashes, H078 final evidence, 115 prior scientific sources, 68 prior frozen plans
and three current navigation documents. Add three isolated recovery sources,
for 118 total; archive the unchanged H079 theory and both qualification plans.
At completion verify all prior metadata/artifacts, including original damaged
H079 files, and preserve a duplicate metadata archive. Generate the readable
report, update README/current state/ledger, and distinguish local qualification
from the unmet broader research objective.

References: [H079 incomplete result](blast_operator_results.md),
[original immutable plan](blast_operator_plan.md),
[theory and source attribution](blast_operator_theory.md),
[integrity evidence](../results/blast_operator_v1/integrity_observation.json).
