# H083 - One explicit binding-only recovery of H082

Freeze before recovery implementation/execution. H082 is INCOMPLETE_PREFLIGHT:
seven checks pass, but patch.multiple(base, **replacements) receives its target
argument twice because the dictionary also replaces a function named target.
The counts/optimizer check cannot enter the binding context. The coordinator
is terminal and original worker handles are absent. No dataset or fitting cell
was created; there were zero optimizer updates.

This is one explicit exception to H082's no-automatic-retry rule, justified by
the diagnosed adapter error before any fitting outcome. Preserve every original
source, observation, tensor, log and process record. There are two preflight
attempts across H082/H083 and one explicit repetition. The fitting grid, if
qualified, runs for the first time. No H081 recipe is reopened.

## Permitted changes

Import the unchanged H082 model and study. Rebind only its artifact root and
plan to results/latent_activation_recovery_v1 and this document. Replace the
colliding patch.multiple context with ExitStack plus individual patch.object
contexts for the SAME replacement mapping, including target. Keep the separate
torch.save durability patch. All fitting/model/target/diagnostic/selection bodies,
parameter domains, initialization, seeds, rows, precision, comparison thresholds,
rates and gates remain exactly those in the [original plan](latent_activation_plan.md).

Alias all original eight checks into an isolated recovery test module; do not
change their bodies or parameterization. Require all eight and zero qualification
optimizer updates. Preserve the numerical settings and the six saved fidelity
tensor cases. Compare the seven readable prior observation results and tensor
payloads where applicable; path/hash fields can differ by artifact destination.
No tolerance relaxation, curve modification or additional shape search.

Only after preflight passes, execute the unchanged allocation once: four
permuted nonlinear tasks, eight forms, two rates, three seeds, 600 updates,
batch256,192 cells and115,200 updates/29,491,200 example presentations. Keep
the separate training/selection/reporting splits and36 selected reset ablations.
All H082 quality/positive-control/resource-promotion rules remain binding.

Use UV and one bounded GPU worker. Preserve PID/UTC/return/log and source
records, fsync and semantic artifact readback. Stop on any runtime, finite-value
or fidelity failure and preserve partial work. No further retry or repair follows
automatically. Inspect actual handles before considering recovery.

## Evidence and closeout

Pin the H082 preservation audit, source archive, protocol, preflight log and
all27 original non-cache files by SHA/size/mtime. Retain the H081 final/result/
source anchors. Inherit125 scientific sources and add three recovery sources
(adapter, test wrapper, launcher), for128 total. Archive both plans and the
unchanged scalar-theory note. Preserve the prior71 frozen plans and navigation
documents before editing.

After fitting, independently regenerate data/permutation/streams; verify every
checkpoint, optimizer state, history, selected rate and reset ablation; rescore
selected held-out predictions without updates and derive all gates separately.
Check original H082 files remain unchanged. Report the failed attempt and the
recovery distinctly. No active model, variant, recipe or publication is added.
The research goal still requires language, longer multi-seed training,
convergence, scale, broader data and useful GPU performance.

[H082 failure](latent_activation_results.md),
[preservation audit](../results/verification/latent_activation_preflight_failure_v1.json),
[unchanged local proofs](latent_activation_theory.md).
