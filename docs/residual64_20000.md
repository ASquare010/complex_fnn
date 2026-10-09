# Residual64: 20,000 additional updates

User authorized one20,000-update continuation of the selected10,456,576-parameter
plain residual model. Same64tokens/vector,512context,256width,4encoder/1decoder,
Branch Sigmoid FFNs,RoPE,code125. Start from the saved999/1000short,202/206packed
checkpoint at44,000 cumulative updates;finish at64,000 if successful.

Restore AdamW moments/counters and Python sampler/CPU/CUDA RNG. Constant lr3e-5,
weight decay.01,clip1,micro2xaccum4. Same25%short/25%packed/25%uniform/25%patterns.
The previous15,000-step recipe was prepared only;this is ONE20,000-step run,not both.
Run-specific recipe dump/residual64-20000-v1/recipe.json;generic config remains unchanged.
Output D:/Git/latent-text-compressor/artifacts/runs/residual64-continue-20000-v1.
Notebook runner sources and recipe freeze at launch;copied into output/source.

Single CPU16-pinned worker;uv run --no-sync,no environment changes. Full independent
complete decoded identities,source/parent/model checks,finite gradients and memory
caps1300/1500MiB allocated/reserved retained. Separate resource probes/initial full
validation remain waived,not passed. No optimizer probes or extra models. No blind
new native/data-failure retries. Historical native issues remain unresolved.

Evaluate all206packed and1000short rows every500;retain final and packed-selected
best checkpoints. Full final/best replays,encoder parity,source/checkpoint/record
hashes and all20,000 updates required. Optimizer counter ends34,000 since last
reset,while total inherited training reaches64,000. Keep all earlier weights and
results. No automatic inference promotion. Same reused validation/one inherited
seed;no untouched test evaluation;100% is a target,not guaranteed.

Status: authorized,prepared for launch. No monitoring automation.

## Launch verified

Worker16392 advanced through its first optimizer update. Complete independent
data identities/source/parent checks passed. Full AdamW(step14000),sampler and
CPU/CUDA RNG restored. GPU627/8188MiB;first-update training404.30/448MiB allocated/
reserved,not final peaks. No launch failure. Sources frozen. Training is active.

## User-authorized recovery

The interrupted run ended with5545 logged updates and a verified5500 checkpoint.
The latest user explicitly authorized resuming training. Separate recovery run
executes remaining14500 updates (49500to64000 cumulative),restoring AdamW19500,
Python sampler and CPU/CUDA RNG. Same architecture,data,constant lr3e-5. The45
logged but unsaved updates are discarded/re-executed,not new schedule progress.
Original cause unknown;no controller/worker remains. Original evidence retained.

Sources dump/residual64-recovery-v1;output standalone artifacts/runs/
residual64-recovery-14500-v1. Step counts in recovery status are relative to restart;
add5500 for progress against original20000 budget. Previous999/1000 model remains
selected. No guarantee of100%;no untouched-test evaluation or extra experiment.
Pinned PowerShell crashed during prior documentation write;checkpoint verification
itself had completed successfully. No automatic retry of that failed write script.

Recovery launch verified:worker31944 at50/14500 new updates (5550/20000 original
schedule). Complete identities and parent/source/finite-state checks passed;
AdamW19500,sampler and CPU/CUDA RNG restored. Early425.20/458MiB training memory,
not final peaks. No launch failure. Sources frozen;previous best model preserved.
