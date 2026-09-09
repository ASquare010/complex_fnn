# H048: Equal extended learning-rate search at 800 steps

Frozen before any new H048 trial. H044 establishes a useful plain BlockShuffle
control at peak LR 0.0012, but every 800-step full/narrow recipe has not had an
independent extended rate search. H046/H047's new headwise candidates failed
quality and earn no longer training. This round tests the existing strong
candidate against stronger tuning, rather than introducing another architecture.

## Question and fixed design

Does the 70.3125%-compressed plain BlockShuffle recipe retain the gold local
quality/memory gates after equal per-recipe global-rate search at 800 steps?
Use four recipes: full SwiGLU, full GELU, calibrated narrow SwiGLU and plain
BlockShuffle. Primary rates are 0.0012, 0.0024, 0.0048. Reuse the four completed
seed-17 cells at 0.0012; run two new rates per recipe, eight new trials in total.
The earlier plain 0.0006 cell is secondary historical context, not an extra
candidate in this primary three-rate selection. No affine/headwise rate search
is added. Prior architecture and optimizer exploration effort is unequal and
must be disclosed. The same-rate affine conclusion remains a historical result
at 0.0012, not a claim about activations at untested higher rates.

Within each recipe, change only TrainConfig.learning_rate from its 800-step
reference. Preserve model dimensions, initialization, data, sampling order,
optimizer calibration, decay mode, recomputation, schedule length and precision.
All trials use width 384, eight layers, attention heads 6, context 128, vocabulary
4096, batch 16, seed 17, native BF16, AdamW betas (0.9,0.95), eps 1e-8, clip 1,
80-step warmup and cosine decay to 0.1 times peak. Full SwiGLU hidden is 1024,
full GELU hidden 1536, narrow hidden 304, BlockShuffle hidden 2048/groups 8.
Narrow retains width-calibrated initialization/down LR and product decay;
BlockShuffle retains factor LR calibration, parameter decay and native gate
recomputation. Full controls retain uniform LR and ordinary autograd.

Global LR multiplies every existing group scale, including AdamW's per-step
decay amount; it is not an isolated change in function-space learning speed.
Keep the common training implementation unchanged. Each completed trial trains
1,638,400 sampled tokens and evaluates all 322,688 WikiText-2 validation targets
at initialization and steps 1/200/400/600/800. New training budget is at most
13,107,200 tokens (eight trials); official test stays unfetched and unscored.

Run 0.0024 in order full SwiGLU, calibrated narrow, BlockShuffle, full GELU;
run 0.0048 in reverse order. One fresh GPU worker at a time, at most 2400 seconds
per trial. This balances recipe order but does not create a paired timing study.
Finish the frozen grid unless an infrastructure failure needs diagnosis.
A trainer-recorded nonfinite-gradient failure is a numerical failure of that
cell, preserved and excluded from selection; continue the other planned cells.
An unexplained process/import failure is not a numerical architecture result:
stop, preserve it, diagnose and explicitly qualify any unchanged continuation.
No automatic retries, overwriting run artifacts or adaptive extra rates.

## Preconditions and artifact verification

Verify all current computation against the passing 220-test snapshot and H047
final audit. Existing NN, training, optimizer, data and diagnostic code must be
unchanged in this round. Verify all four reference metrics/checkpoint/archive
hashes against H039/H043 records, their configurations, data and sampling state.
Historical registry/factory/count/diagnostic additions are explicitly qualified
by the 30-model exact CPU snapshots and unchanged old execution branches.
Compare shared literal sources and training/attention/forward ASTs against
reference archives; reproduce reference matrix counts and optimizer groups.
The four current initial full-validation NLLs must match their reference values
within 1e-7 in a fresh GPU worker before new training.

Check all eight immutable data hashes and reconstruct the complete validation
stream, including the last nine-window batch. New initial NLL, parameter counts,
optimizer metadata, source archives, checkpoints, histories, finite layer/gradient
statistics and final sampling RNG must match their own frozen reference contracts.
Record clipping, pre-clip gradients, allocation and descriptive timing. Source
hashes and archived plan accompany the audit and every training run.

A resumed audit requires a separate written qualification identifying its
failure stage and confirming unchanged computation/plan and preserved completed
runs. Resume only unfinished cells whose expected run directory does not exist;
never rerun a completed or numerically failed cell. A partial run directory
requires diagnosis and a separate explicit protocol, not silent reuse.

## Selection, gates and next action

Select lowest final validation NLL independently per recipe among its three
primary rates; ties choose lower rate. Failed numerical cells cannot win.
Report the entire twelve-cell matrix, same-rate candidate/control differences,
selected-recipe differences, all failures, initial losses and learning curves.
Report whether each winner is an interior grid point or a lower/upper boundary;
three discrete points do not establish a global continuous optimum.

Apply the existing gold local gates to selected plain BlockShuffle: >=70% FFN
weight reduction, <=1% relative NLL cost against BOTH selected full controls,
strictly better NLL than selected calibrated narrow, and peak training allocation
<=1.1 times EACH selected full control, with all recorded diagnostics finite.
Report separately whether the narrow margin is >=0.2%. These are engineering
gates, not significance tests. Report parameter counts and logical matrix FLOPs;
fewer weights do not imply lower latency.

If quality fails after expanded tuning, qualify the earlier same-rate gold
claim and do not promote that recipe on the uncorrected comparison. If quality
passes but memory fails, require a separate execution investigation. If all
pass, freeze an independent-seed comparison of selected rates before concluding
replication, then test a longer budget for convergence. Rate rankings can change
with duration: no automatic long run, same-rate causal claim, convergence,
published-scale superiority or new primitive is inferred from this screen.
