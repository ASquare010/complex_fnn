# H082 - Incomplete latent-activation preflight

**INCOMPLETE PREFLIGHT; no fitting began.** Seven of eight checks pass, including
scalar derivatives, finite differences, exact initial identity and CPU/GPU
checkpoint-gradient comparisons. Six tensor archives pass integrity checks.

The counts/optimizer check fails before entering its context: the adapter calls
patch.multiple(base, **replacements) while the replacement dictionary contains
a key named target. Python therefore supplies the function's target argument
twice. This is an adapter error, not a failed curve measurement.

The process is terminal (return code 1). There are zero optimizer updates, zero
fitting cells and no generated fitting dataset. All original sources, logs,
observations and tensor artifacts remain preserved. H082 has no fitting result.

A separately frozen recovery may replace the colliding bulk patch with individual
patch.object contexts. It must preserve all numerical functions, initialization,
seeds, thresholds and allocated fitting cells. It does not reopen H081.

[Original plan](latent_activation_plan.md), [local proofs](latent_activation_theory.md),
[preservation audit](../results/verification/latent_activation_preflight_failure_v1.json).
