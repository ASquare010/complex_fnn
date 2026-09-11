# Current research state

## Latest: shared-latent modulation capacity obstruction

[H144](modulated_capacity_results.md): **capacity obstruction verified**.
For Gaussian linear targets, arbitrary output modulation of one h-dimensional
observation reduces to diagonal-plus-rank-h approximation. An explicit rotation
has exact relaxed normalized error max(1-2h/d,0): one third at d384/h128.
Eight integer/SVD witnesses, four conditional-risk checks and candidate gradcheck
pass, with zero GPU work. [Proof](modulated_capacity_theory.md).

Do not allocate h128 fitting merely because its output rank is full. A new
hypothesis must alter the shared observation bottleneck or accept a larger latent
width and measure its resource cost. This is not a language-model bound or a
rejection of all modulation. The broader research goal remains open.

## Previous: conjugated shared-FFN falsification

[H143](conjugated_ffn_results.md): **REJECT fixed shared-permuted recipe.**
54 fits, three tasks/seeds, 16,200 updates; rank/gradient preflight and independent
data/score audit pass. No active model/default changes.

Do not insert this recipe into language models, tune kernels or claim a better FFN. The rank witness survived; the fixed learning/resource recipe did not. Use the linear and teacher-task controls to distinguish optimization or tying restrictions from the rank obstruction. Cubic nonlearning is inconclusive when wide controls also fail. A future change must identify a distinct mechanism or a specific failure diagnosis, not silently extend this failed budget.
The broader VRAM/parameter-efficient FFN research goal remains open.

## Previous: doubled-batch memory-helper qualification

[H142](batch_scale_training_results.md): **PASS scoped batch-16 qualification gate.**
720 updates across all six saved corpus/seed states; independent audit: True.

The maintained helper now has bounded evidence at batch8 and batch16. Preserve ordinary defaults and H138's failed long-run result. Return to the unresolved structural FFN/parameter-efficiency question using the repository's prior candidate eliminations; do not infer novel neuron geometry or sustained throughput from these memory-only results.
The broader VRAM/parameter-efficient FFN goal remains open.

## Previous: maintained opt-in memory helper

[H141](training_memory_integration_results.md): **integration verification passed**.
Explicit helper, unchanged classifier arithmetic, 16 new tests and 12 GPU replay
updates across all six source states. All 132 tests have passing coverage after
a documented compiler-path rerun of six existing Triton tests. Old maintained
sources/defaults remain unchanged. [Usage](training_memory_usage.md).

Next: qualify another workload scale through the maintained API with ordinary
controls and memory/compute/quality checks. H138 remains failed; sustained
throughput and the broader parameter-efficient FFN objective remain open.

## Previous: interleaved complete-training timing

[H140](interleaved_training_results.md): **PASS scoped interleaved timing gate.**
720 updates across all six saved corpus/seed states; independent audit: True.

Design an explicit opt-in maintained implementation with source-equivalence tests, preserving defaults, then qualify another workload scale. This pass concerns only balanced short segments at saved states; H138 remains failed and sustained deployment throughput remains unproven.
The broader VRAM/parameter-efficient FFN goal remains open.

## Previous: timing-drift audit

[H139](timing_drift_audit_results.md): verified all12 H138 runs and9,360 measured
updates with zero GPU training. The final ordinary control transitions P0 to P4
around updates541-566; its last timing block rises to142ms from78-80ms. The first
WikiText pair also has different clock distributions. Cause remains unknown;
H138 remains failed and no candidate is promoted.

Next: prospectively freeze balanced, temporally interleaved complete-update
comparisons at saved states, retaining every round and the15% runtime limit.
Do not repeat long convergence runs to chase favorable timing. This must remain
separate from H138 and cannot by itself prove deployment throughput or a new FFN.
The broader VRAM/parameter-efficiency research goal remains open.

## Previous: four-block fresh-training validation

[H138](partial_offload_long_results.md): **FAIL four-block fresh-training gate.**
12 fresh 800-update runs, three seeds and two corpora; independent audit: True.

Do not promote the configuration. Retain every seed and inspect the failed quality/runtime/resource gate. Choose a narrower falsifiable follow-up without relaxing limits, reordering completed runs or using aggregate means to rescue failures.
Historical failures and the full research goal remain open.

## Previous: partial checkpoint-input offload

[H137](partial_offload_training_results.md): **PASS short qualification: buffer4.**
480 updates; zero/four/eight-block transfer comparison and independent audit.

Validate buffer4 from fresh initialization against ordinary native training on both corpora and three seeds, using complete-update timing and native scoring. Preserve H136's failed full-offload result; this short continuation does not prove fresh convergence or robust long-run efficiency. The broader FFN/activation parameter-efficiency objective remains open.
Historical failures and the full research goal remain open.

## Previous: fresh three-seed ordinary-policy training

[H136](ordinary_long_training_results.md): **FAIL three-seed long-training gate.**
18 fresh 800-update runs, three seeds, two corpora; independent audit:True.

Do not promote the configuration. Identify the failing seed/gate from the table, retain all three seeds and choose one narrower follow-up. Do not change limits or substitute aggregate means for a failed individual run.
Historical failures and the full research goal remain open.

## Previous: ordinary-policy complete training

[H135](ordinary_complete_training_results.md): **PASS ordinary short-training gate.**
360 updates; all arms share ordinary execution and default workspace.
Independent numerical/native-score/noise audit: True.

Run fresh longer training with at least three independent seeds on both corpora, paired ordinary/native-offload/buffer-offload controls, native validation and complete-update resource accounting. The short result earns that test; it does not prove fresh convergence or long-run quality. Preserve the existing structural FFN failures while pursuing the broader parameter-efficiency goal.
Earlier failures and the broad research goal remain open.

## Previous: paired complete-update training

[H134](paired_complete_training_results.md): **FAIL complete short-training gate.**
360 updates, mirrored run order, complete-update timing, passive telemetry.
Independent numerical/native-score audit: True.

Test the same candidate under ordinary/default attention policy, alongside native loss with and without checkpoint-input offload. Calibrate full-model numerical tolerances against repeated ordinary native controls, retaining the existing exact operator qualification. This isolates whether strict deterministic execution is the remaining practical runtime cost; do not assume it is the cause. Keep paired complete-update timing, native scoring and memory gates, and do not extend long training yet.
Earlier failures remain recorded; the full research goal remains open.

## Previous: balanced paired workspace timing

[H133](paired_workspace_timing_results.md): **PASS paired timing gate.**
All 32 bursts match fresh native gradient references bitwise.
260 backwards, zero updates, passive GPU telemetry; independent CPU audit verified.

Run a separately frozen complete-update comparison at the explicit 8-MiB capacity, using alternating candidate/control windows with telemetry. Include both ordinary execution and matched deterministic native controls, original clipping/Adam, native validation and peak GPU/host memory. Do not combine H132 memory with H133 timing to claim a complete-training win.
Earlier gates and the broad research goal remain open.

## Previous: explicit intermediate-workspace comparison

[H132](blas_workspace_mid_results.md): **FAIL combined diagnostic gate.**
8-MiB external workspaces save 48 MiB versus 32-MiB workspaces while keeping
the deterministic environment fixed. Native replay: True; accounting verified.
88 backwards, zero updates, eight artifacts; all 61 maintained files unchanged.

Do not expand to longer training. Use a separately frozen timing experiment with alternating candidate/control probes and GPU clock/power telemetry to distinguish a persistent runtime penalty from execution-order drift. No threshold or old-result changes; the present gate remains failed.
The full research goal and all earlier failed gates remain unchanged.

## Previous: deterministic workspace isolation

[H131](blas_workspace_results.md): **FAIL combined diagnostic gate.**
Workspace release accounting verifies the 63.75 MiB high-low difference.
Native replay audit: True. 88 backwards, no updates, eight gradients;
all 61 maintained files unchanged. No long-run quality or novelty claim.

Do not extend this 128-KiB workspace to longer training. Test an intermediate explicit workspace size through the installed PyTorch workspace API, keeping the deterministic environment and algorithm policy fixed. That isolates external workspace capacity without accepting this speed penalty. Require fresh exactness/resource comparisons; no threshold or old gate changes.
Prior failed gates and the full research goal remain unchanged.

## Previous: complete native-buffer training comparison

[H130](native_buffer_training_results.md): **FAIL combined short-training gate.**
Eight 30-update continuations, 240 updates, 24 tensor artifacts. Independent
numerical/data/score and deterministic-equality audit: True. All 61
maintained files unchanged; no parameter reduction or default change.

Do not expand the training matrix yet. The deterministic native arm adds exactly 47.75 MiB over ordinary native on both corpora. Isolate the deterministic policy bundle, especially cuBLAS workspace allocation, before more training: test a lower-workspace deterministic configuration with matched controls, exactness checks, and full resource accounting. This is a causal hypothesis, not an attribution established by the present experiment. Preserve the current failures and thresholds.
The full research goal remains open.

## Previous: explicit-policy reproducibility test

[H129](attention_reproducibility_results.md): **PASS scoped reproducibility: deterministic_default, deterministic_math.**
36 backwards, zero updates; 18 gradient artifacts, CPU verification
complete, all 61 maintained files unchanged. No training/resource claim.

Run a separately frozen complete-update and resource comparison under `deterministic_default`, including all transfers, original clipping/Adam and native validation. Match the policy in both arms. Compare its cost with ordinary execution separately before claiming practical efficiency.
Prior failed gates remain unchanged; the broad research goal remains open.

## Previous: classifier resource gain, replay gate unresolved

[H128](native_buffer_layout_results.md): layout-aware native-buffer reuse saves
14.50–14.75% diagnostic GPU allocation with near-equal runtime. Nine operator
cases pass bitwise checks, but independent full-model gradient replay fails
for candidate AND native controls. All loss values match. Combined gate fails.
62 backwards, zero updates, four artifacts; H127's earlier 16-backward failure
is preserved. No defaults or parameter counts change. Next: native reference
reproducibility and explicit attention-backward policy before any longer run.
The full research goal remains open.

## Previous: native-buffer exactness failure

[H127](native_buffer_loss_results.md): native-buffer reuse fails bitwise input-gradient
checks in five of eight operator cases. Loss/weight gradients are exact; finite
differences pass. Sixteen backwards, zero updates; all model profiling stopped.
Next: test PyTorch's layout-dependent input-gradient multiplication explicitly.
No VRAM/quality claim or maintained default change. Full goal remains open.

## Previous: long compact-training comparison

[H126](compact_long_training_results.md): **FAIL: the candidate does not pass every fixed long-run gate.** Six matched runs from
original initialization, 800 updates each, cover native/chunked/combined arms
on both corpora. Independent audits pass. Total 4,800 updates/4,812 backwards;
24 tensor artifacts, no repeated runs, all 61 maintained file hashes unchanged.

Expansion to more seeds is not earned under this protocol. Investigate the failed dimensions before allocating another training matrix. Memory savings remain scoped observations, not a rescued quality/runtime gate.
One seed per corpus does not establish broad quality or architectural success.
The full research goal remains open. See the report for every comparison.

## Previous: compact storage passes short complete-training validation

[H125](compact_training_results.md): four 30-update continuations pass all
fixed gates. Complete-job VRAM falls 13.82–14.42%, with 4.67–5.57% event-time
overhead and about 0.0006% final NLL difference. Original AdamW/global clipping,
BF16 evaluation fallback and all transfers are included. Independent update,
batch and native-score audits pass. Total120 updates/122 backwards; no retries.

Next: longer training from original initialization, conventional native-loss
controls, multiple seeds and scale. No parameter reduction or default change;
H117/H119 failures and the broader research goal remain unresolved.

## Previous: compact RMSNorm plus gradient staging qualifies a training test

[H124](compact_rmsnorm_results.md) passes the two-corpus diagnostic gate:
13.73–14.33% lower peak VRAM, with 4.36–6.41% event-time overhead, relative
to ordinary RMSNorm/resident gradients with checkpoint-input offload. Compact
RMSNorm alone saves 12 MiB but fails 10%; gradient staging alone still fails.
The combined arm passes memory, time and host-memory gates. Gradient checks,
finite differences and bitwise staging restoration pass. No updates yet.

86 backwards, eight gradient artifacts, zero repeated cases; 61 maintained
file hashes unchanged. This is an FP32 first-derivative storage implementation,
not a new activation or parameter reduction. Next: frozen complete-training
validation with original AdamW/clipping, BF16 evaluation fallback, actual job
VRAM and endpoint quality. Earlier quality/numerical failures remain failed.
The broader research goal is still open.

## Previous: gradient staging narrowly misses the memory gate

[H123](gradient_staging_results.md) completes 42 backwards, zero updates.
Completed gradients move to pinned CPU buffers and are restored to CUDA;
round-trip cost is included. Peak allocation falls 26.60–27.53 MiB, but
9.4605%/9.9778% savings both fail the fixed 10% requirement. Gradient checks
pass; event overhead is 11.3–13.0%, within the 15% limit. Combined pinned-host
allocation is 112.034 MiB. No training extension, default change or repeated
case. Next hypothesis: reduce remaining temporary storage, such as RMSNorm
backward intermediates, under a separate correctness/resource protocol.
The broader research goal remains open; H121/H122 gates remain failed.

## Previous: backward payload peak shifts after checkpoint-input offload

[H122](backward_allocation_results.md) completes 12 backwards, zero updates.
Requested-payload peaks move from block 7 backward to block 0 backward on both
corpora after input offloading. All four gradient comparisons pass. Exact
allocated-peak attribution remains **FAILED**: event requested sizes differ
from allocator block sizes. Offline requested-byte accounting reconciles;
it does not rescue that gate. No GPU case was repeated or default changed.
Next test: inventory completed parameter gradients at the late backward peak
before designing another storage change. The broad research goal remains open.

## Previous: checkpoint-input offload preserves gradients but misses the combined gate

[H121](checkpoint_input_offload_results.md) temporarily places eight block
checkpoint inputs in pinned CPU memory using an existing PyTorch API. With
native FP32 classifier loss, complete diagnostic-case peak falls by exactly
48 MiB (11.81–11.99%) on WikiText and TinyStories, with 1.21–2.34% event-time
overhead. With chunked loss it falls by only 14.47–15.40 MiB (4.98–5.19%).
**The all-four-fixture 10% memory gate fails.** No training extension is
earned under this protocol; no maintained default changes.

All traces confirm eight pinned [8,512,384] FP32 inputs, exact copy hashes,
one unpack each and zero logical payload after cleanup. Payload is 48 MiB;
rounded pinned-host allocation is about 64 MiB and its cache persists. FP64
finite differences and 24 NumPy gradient comparisons pass; maximum global
gradient relative L2 is 9.344e-8 and checked losses match exactly. GPU allocator
boundaries are zero and model/moment states remain unchanged.

The first audit process crashed during SymPy import before PyTorch/replay.
Its failure is preserved; a prospective recovery runs only the eight remaining
replays through the unchanged audit function. Total work: 262 backwards,
zero optimizer/training updates, 1,048,576 diagnostic target evaluations,
24 gradient artifacts and 824 full-model memory intervals. Repeated fixed-state
targets are not training exposure. The native import issue remains undiagnosed.

All 162 frozen sources and 61 maintained files stay intact. The previous
116-test pass is historical, not a new test execution. No parameter reduction,
endpoint quality improvement or novelty is claimed. H117's quality and H119's
active-clipping failures remain. A separately budgeted allocation timeline
could identify the remaining chunked backward peak before another change;
phase-boundary inventories alone do not uniquely explain it.
[Plan](checkpoint_input_offload_plan.md),
[recovery](checkpoint_input_offload_audit_recovery.md),
[receipt](../results/verification/checkpoint_input_offload_final_v1.json).

## Previous: optimizer temporary savings do not reduce the complete-job peak

[H120](optimizer_memory_results.md) compares default, per-tensor and fused
native AdamW on the same update800 states for WikiText/TinyStories, using both
native and chunked FP32 classifier losses. Twelve cases × 30 updates complete.
Backward sets all 12 job peaks. Fused saves 35–36 MiB within the optimizer phase,
but complete-job allocation rises 25 KiB; per-tensor mode leaves it identical.
**Both are ELIMINATED for the fixed 10% whole-job memory gate.**

Fused event-sum timing is 2.7–3.6% lower in the instrumented screen; this is not
an uninstrumented throughput claim. Per-tensor mode fails several timing gates.
All short NLL, stability, numerical, batch and scoring gates pass. These are
correlated continuations of one seed per corpus, not a broad quality result.
First-step clipping is inactive (norm<1), so its zero error does not resolve
H119's active CPU clipping failure.

The audit independently verifies first-step AdamW with NumPy FP64, every
first/final model/moment state, all 360 batches/samplers and 12 native final
scores. Total work is 360 updates/backwards / 1,474,560 targets, plus 24 study
and 12 audit scores. All 1,884 memory intervals and 26 zero CUDA boundaries
are retained. No scientific stage or case was repeated. A later read-only
PowerShell display failure is recorded separately.

All 152 frozen sources and 61 maintained files stay intact. No optimizer,
layer, parameter count or default is promoted; the historical 116-test result
is not represented as a new test run. H117's old quality gate remains failed.
Move the memory hypothesis to backward storage: inspect which checkpoint
inputs remain resident and test selective host storage only under a separate
budget with gradient/lifetime/transfer-cost checks. The eight input tensors
contain at most 48 MiB of payload; that is an estimate, not a measured saving.
The broad VRAM/quality and architectural goal remains open.
[Plan](optimizer_memory_plan.md),
[receipt](../results/verification/optimizer_memory_final_v1.json).

## Previous: first-step sensitivity supports a mechanism, not a training cure

[H119](adam_update_sensitivity_results.md) uses six frozen initial probe gradients
from the selected WikiText seed and 12 disposable native AdamW steps. Across all
nine cross-policy pairs, ideal first-update direction differences are amplified
**61.21–63.51x**; at least **99.888%** of squared difference comes from gradients
within 10 epsilon of zero. The direction difference is still only about 0.0017%.
This does not prove the cause of the later 800-step quality gap.

The independent NumPy calculation verifies all 45 direction pairs, 12 distortions,
15 native pairs and six device comparisons. All six CUDA cases pass the fixed
numerical checks. **The overall numerical audit fails**: CPU clipping differs
from FP64 by up to 6.872e-6, above the fixed 1e-6 bound. All parameter/moment
checks pass. Larger epsilons reduce discrepancy but change directions by
10.79%/32.76%, so both fixed low-distortion remedies are rejected.

The original process completed six CPU steps and then failed its CUDA-init
guard. Its outputs and exit are preserved; a frozen continuation runs only the
remaining six GPU cases through the unchanged function. Total is 12 disposable
optimizer steps, zero language-training updates/forwards/backwards/scores,
24 tensor artifacts and seven zero CUDA allocator boundaries. No tolerance or
case is repeated. All 145 original frozen sources and 61 maintained files remain
intact; recovery sources have their own manifest. The prior 116-test pass is
historical, with no new maintained-suite run.

H117's failed two-corpus gate stays failed, H118 stays inconclusive, and no
optimizer or model default is promoted. Return the memory investigation to
measuring optimizer temporaries versus persistent moments at the complete-update
peak under a separate plan; no new long training sweep is allocated. The broad
VRAM/quality and architectural goals remain open.
[Plan](adam_update_sensitivity_plan.md),
[recovery](adam_update_sensitivity_recovery_plan.md),
[evidence receipt](../results/verification/adam_update_sensitivity_final_v1.json).

## Previous: repeated execution exposes material trajectory variation

[H118](training_variability_results.md) adds four 800-update executions of the
failed WikiText seed101, using the same initial state, batches and unchanged
H117 runner. New chunk/native NLL gaps are **+1.143% and +0.940%**, versus the
original +1.352%. All three chunked endpoints are worse than all three native
endpoints, but the smallest cross-comparison gap is only 0.630%, below the fixed
1% rule. The preset diagnostic verdict is **INCONCLUSIVE**. These are repeats
of one deliberately selected seed, not three independent training seeds.

The native NLL range is 0.035284, smaller than the original 0.066509 pair gap.
Observed ranges do not overlap, so repeat variability does not span the entire
original gap. It still affects interpretation, and no unique cause is proved.
Within-policy initial gradients differ by at most 5.904e-8 symmetric relative L2;
median model distances reach 36–37% by step 800. First recorded scalar-loss
differences appear at steps 14–18. This does not establish catastrophic gradients.

All 12 new trained checkpoints, 13 native scores, four FP32 backward replays and
all 3,200 new sampled batches verify. The new runs preserve 26.816% lower allocated
memory versus native FP32, costing 12.65–17.65% more warmed update time. New
budget is 3,200 updates / 13,107,200 targets / 3,208 backwards, with no failed stage
or scientific retry. All 137 frozen sources and 61 maintained hashes are retained.

H117's failed WikiText/two-corpus gate stays failed; its scoped TinyStories
component remains separate. No new default, layer or parameter reduction is
claimed. Do not allocate more identical long repeats automatically. A bounded
first-Adam-update comparison on the six saved initial gradients is the next
specific diagnostic question and needs a separate plan. The broad goal remains
open; the prior 116-test result is historical, with no maintained-suite rerun.
[Plan](training_variability_plan.md),
[receipt](../results/verification/training_variability_final_v1.json).

## Previous: fresh replication retains memory savings but requires every seed's quality

[H117](fp32_training_replication_results.md) completes 18 fresh 800-update runs
on seeds 101/113/127 for both WikiText-2 and TinyStories. Full-job tensor allocation
falls 26.98–27.75% versus BF16 native, with practical warmed update time.
wikitext2 fails the fixed gates; tinystories qualifies in this scope. Failed gate names: final_quality_bf16, final_quality_native.
The two-corpus recipe does not qualify.

The candidate and both references have identical 9,099,648 parameters. Initial
FP32 gradient fidelity does not guarantee the endpoint after 800 updates.
Independent verification covers six exact initial-state regenerations, 54
trained model/Adam/sampler checkpoints, 60 complete native validation scores,
12 FP32 initial-gradient replays and all 14,400 sampled batches. Three seeds,
reused development validation and no complete-trajectory repeats limit the claim.

The original startup failed after 24 CPU qualification backwards and before
any training update because environment metadata initialized CUDA too early.
Its frozen recovery reorders CPU initialization before metadata; no training,
scientific recipe or tolerance is repeated or changed. Total backwards across
attempts: 14,478; recorded memory intervals: 28,998. Original failure is retained.
The initial figure process later hit a Matplotlib import access violation;
one frozen-input CPU recovery ran the unchanged plot successfully. No training
or audit was repeated and no runtime cure is claimed.

No default or maintained model is promoted. All 61 maintained files remain
unchanged from the earlier 116-test pass; that suite is not rerun. Keep the
scoped memory evidence and close failed fixed quality claims. Paired versus
within-policy trajectory variability is the next bounded diagnostic question,
before any further broad training sweep. The broad goal remains unmet.
[Plan](fp32_training_replication_plan.md),
[recovery](fp32_training_replication_recovery_plan.md),
[receipt](../results/verification/fp32_training_replication_final_v1.json).

## Previous: default FP32 chunks pass the narrow-model full-job resource gates

[H116](fp32_decoder_resource_results.md) completes 30 matched continuations /
1,500 updates in 291.67 seconds. Default-attention FP32 decoder/classifier
chunking reduces job allocation from 407.25 to 297.43 MiB on WikiText and
400.22 to 291.25 MiB on TinyStories: **26.97% / 27.23% saved**. Median warmed
update time changes -0.12% / +0.33% versus BF16 native. Same-backend native
FP32 controls show 26.81% / 27.23% saving with 16.03% / 17.70% slower updates.

All original gates pass for these two narrow scopes. Full gradient error is
below 9.354e-7. The independent audit passes all 24 FP32 gradient replays,
36 native scores and 1,500 batch/state checks. Default-chunk NLL changes stay
within -0.00159% to +0.00170% versus BF16 after 50 updates. This does not prove
long-run quality: the four narrow states are correlated seed61 pairs, with
single-seed full controls. No architectural or parameter-count change occurs.

Forced math attention fails memory and BF16-relative time on both narrow
corpora. Full GELU/SwiGLU fail the 15% job-memory gate under either backend.
Retain only default FP32 chunks on narrow context512 for broader replication
and duration testing. No fresh training grid or default change occurs here.

CPU analysis initially hits a Windows access violation. Its failure is retained;
a fresh process runs the unchanged analyzer successfully, with no scientific
retry. The 61 maintained files remain unchanged from the earlier 116-test pass.
The broad goal stays open. [Plan](fp32_decoder_resource_plan.md),
[receipt](../results/verification/fp32_decoder_resource_final_v1.json).

## Previous: FP32 decoder modes pass the fixed numerical diagnosis

[H115](decoder_gradient_transport_results.md) completes 30 conditions on six
saved fixtures: correlated WikiText/TinyStories seed61 pairs and full GELU/SwiGLU
seed17 controls. This is 300 main plus 20 qualification backwards, zero optimizer
updates, with 189.85 seconds for the main study. No new model is introduced.

BF16/default's median paired total-gradient error is 0.003181. Disabling
checkpointing does not help. BF16/math removes the observed repeat variability
but still gives median paired error 0.003030. Both FP32 modes reduce paired
error below 9.354e-7 in every fixture/repetition and pass the original transport
and numerical replay gates. FP32/default still has tiny bitwise replay changes;
neither mode establishes global determinism. An exact CPU/CUDA scalar witness
demonstrates 32,768-fold perturbation amplification at a BF16 rounding boundary.

All same-mode forward boundaries match. Precision changes forward values too:
FP32 normalized hidden states differ from BF16 by 0.249-0.384% relative L2.
The independent audit verifies 30 native forward captures and saved gradient
arithmetic, with zero backward replays. H114's failed audits remain failures.

Retain FP32/default/block and FP32/math/block for an **uninstrumented whole-model
resource screen** only. Diagnostic memory excludes optimizer state and hooks
affect timing. No practical VRAM win, fresh LM allocation or default change is
earned here. The broad goal remains open. All 61 maintained files remain
unchanged from the earlier 116-test pass; that suite was not rerun.
[Plan](decoder_gradient_transport_plan.md),
[scoped receipt](../results/verification/decoder_gradient_transport_final_v1.json).


## Previous: full-model precision gate fails despite useful narrow-model memory savings

H114 completes **56 checkpoint continuations / 2,800 updates** in 502.12 seconds,
using all twelve H112 final states plus full GELU/SwiGLU H078 controls. Each
case receives 20 warmup and 30 measured updates from the same paired weights
and Adam state. This is not fresh LM training. FP32 classifier chunks save
**32.36% / 33.06%** whole-job allocation on WikiText/TinyStories narrow T512
models, costing **12.84% / 12.64%** more median update time. Native validation
NLL differs by at most 0.0051% after 50 updates; this does not prove long-run quality.

Every scope fails the original <=0.002 global-gradient relative-L2 gate versus
the native FP32 classifier. Median candidate error is 0.00225 / 0.00337 in the
two narrow corpora. Full T128 GELU/SwiGLU additionally fail memory: only 0.53%
/ 1.08% whole-job saving. Final diagnostics set both full-GELU peaks and the
FP32-chunk SwiGLU peak. All 5,992 memory intervals are logged; warmed timing
block ratios remain <=1.0479, without outlier removal.

Independent exact gradient replay fails, and an explicit 1e-5 global / 1e-4
per-tensor numerical recovery also fails on a different fixture (0.0005106
global error). The cause is unresolved; no further tolerance relaxation or
training retry occurs. A narrower completion audit verifies **70 native scores,
56 saved gradient hashes, final model/Adam states and all 2,800 batches** with
zero backwards. It does not qualify independent gradient replay.

Close the fixed full-model precision-fidelity claim. Retain the scoped memory
observation; hold fresh LM allocation pending a new matched-state backward
diagnosis. Two model folders, five variants, eight recipes and all 61 maintained
files remain unchanged from the prior 116-test pass. The broad goal remains open.
[Report](fp32_classifier_profile_results.md), [plan](fp32_classifier_profile_plan.md),
[verification and explicit limitations](../results/verification/fp32_classifier_profile_final_v1.json).

## Previous: classifier cache error is verified; FP32 chunks earn resource screening

H113 completes 120 local comparisons on 24 saved H112 states, with 504
reference/warmup/measured backward passes and **zero optimizer updates**.
One shared BF16 classifier-weight cast versus eight separate casts explains
a reproducible difference: every cached/uncached loss and hidden gradient is
bitwise identical, while every weight gradient differs. A CPU/CUDA witness
gives cached gradient 256 versus the exact 260, recovered by separate casts.
This is a scoped numerical mechanism, not proof of H112's final NLL cause.

Uncaching reduces paired weight-gradient error by median 35.24% / 22.89% on
WikiText/TinyStories, but fails the fixed improvement/runtime gates. FP32
classifier chunks pass the local gates: approximately 3.7-3.9e-7 weight-gradient
error and 72.02% lower classifier allocation than native BF16. Median paired
time ratios are 1.378 / 1.415, but individual ratios span 0.335-15.862x. Timing
evidence is weak; a full-model screen needs sustained warmup, wall time and
GPU-event measurements. No language trial is allocated by this result.

All 24 native recaptures, 144 gradient reruns and 24 independent closed-form
FP64 derivative checks pass. The study takes 170.33 seconds and has no native
failure or scientific retry. A planned overall diagnostic-process maximum was
not fully recorded because warmup peaks were not serialized; only measured
phase peaks are reported. This limitation does not change the local gates.

Retain the cache diagnosis, eliminate the fixed uncaching-only candidate and
advance FP32 chunks solely to whole-model gradient/resource qualification.
The active model tree and 61 maintained files are unchanged from the previous
116-test pass. The broad goal remains unmet.
[Report and scoped proof](classifier_precision_results.md),
[final receipt](../results/verification/classifier_precision_final_v1.json).

## Previous: whole-job saving qualifies on TinyStories; WikiText quality fails

H112 completes **12 fresh trials / 9,600 updates** across seeds 61/73/89 on two
corpora. Both training arms get H111's qualified classifier-streaming evaluator.
Measured job allocation falls **32.43% on WikiText-2 and 33.06% on TinyStories**,
with 13.65-14.52% slower median updates. These actual per-trial tensor peaks
include validation, diagnostics and CUDA corpus caches; they exclude driver/
context memory. All training, model and Adam quantities remain finite.

TinyStories passes every seed's <=1% NLL degradation, >=15% allocation saving
and <=25% timing-cost gates. WikiText-2 loses 1.465% / 0.042% / 1.954% NLL and
fails two seeds. **Retain the scoped TinyStories component; reject the fixed
two-corpus recipe.** H110's favorable one-seed long-context observation does
not replicate on these fresh WikiText seeds. Its old fidelity stop stays intact.

Independent audit verifies 72 native scores, 48 checkpoints, twelve exact
initializations and all 9,600 training batches. Maximum streamed/native score
drift is 3.40e-8, so evaluation drift does not explain the quality changes.
The grid runs once in 922.36 seconds, using fresh models in one process with
25 verified zero-allocation boundaries. cuBLAS workspaces are cleared only
between trials and fully charged within trials. Original startup failures,
the workspace diagnosis and postprocessing recoveries remain preserved. Native
failure root cause remains unknown; process-isolation changes are explicit.

H111 separately qualifies classifier streaming at T128/T512 on 24 old states
with zero updates: 16-31% lower evaluation allocation and roughly 1% median
time overhead. Sequence microbatching fails cost at 3.71x / 7.07x. H111 uses
persistent-state fixtures; H112 supplies the separate actual-job measurements.

The next discriminator is a bounded matched-state gradient/precision comparison
before allocating any changed training policy. No causal mechanism has been
proved and no further training grid is allocated by these results. Models,
parameters and defaults remain unchanged: two model folders, five variants,
eight recipes, with the prior 116-test pass kept separate. The broader goal
remains unmet. [H112 report](whole_job_memory_results.md),
[H111 report](streamed_evaluation_results.md),
[final audit](../results/verification/whole_job_memory_final_v1.json).

## Previous: training memory improves; whole-job gains remain limited or unreplicated

H110 completes **eight fresh 800-update trials / 6,400 updates**. At B16/T128,
loss chunking passes all frozen training-memory, runtime and fidelity gates
across seeds 17/29/43: **20.32% less training allocation**, 6.91-8.18% slower
updates, and at most 0.745% final NLL change. But unchunked validation raises
the candidate's whole-job peak from 222.3 to 266.6 MiB, so the overall saving is
only **4.45%** versus 279.0 MiB for block checkpointing.

At B8/T512, seed 17 saves **32.83% training / 19.26% whole-job allocation** and
improves final NLL **4.32%**, with 13.05% slower updates. The lower loss is a
positive one-seed quality observation. It fails the frozen +/-1% execution
fidelity requirement, so the four remaining trials are not allocated. Close
that fixed faithful-execution claim, not the possible performance lead. Do
not relabel improved loss as regression or grant a retrospective exception.

Independent audit verifies **40 exact NLL scores, 24 checkpoints, eight initial
states and all 6,400 training batches**. Both original pretraining import
failures and a separate post-training UV-launcher panic are retained. The
recorded continuation uses the original UV-managed Python 3.12.9 with bytecode
bypass, reuses the exact completed control, and repeats no scientific training.
Twelve CPU probes and 1,572 matching SymPy source hashes do not establish the
root cause. All model/Adam states remain finite; clipping occurs on 1.25-3.375%
of updates. Finite gradients do not prove optimal conditioning or convergence.

Retain the B16/T128 training component. The next memory question is streamed
evaluation with independently verified scores; a long-context quality claim
needs a separately frozen replication. No parameter reduction, inference
benefit, new activation or architectural breakthrough is established here.
Two model folders, five variants and eight recipes remain. Maintained code/
tests are unchanged from the previous 116-test pass; current lint passes.
[Full report](token_memory_duration_results.md),
[runtime investigation](runtime_import_diagnosis.md).

## Previous: derivative-aware initialization does not survive value-only learning

H109 completes **144 fresh fits and 86,400 updates** in 363.28 seconds at the
original 70.8% FFN-parameter reduction. Both derivative-aware initializers fail
the fixed learning gate. Before training, their paired derivative advantage is
6.06-9.81% over value-only; after 600 value-only updates it is just **0.41-0.73%**.
All 72 selections choose LR0.001, so the result is also an equal-rate comparison.

Relative to their own initialization, candidate output error improves about
35%, but directional-derivative error **rises 7-8% for GELU and 51-52% for SwiGLU**.
This is measured local sensitivity degradation, not proof of deep-network
exploding/vanishing gradients. All training quantities remain finite and no
update clips. H108's zero-SGD component result remains valid in its narrower scope.

Mean learned output error is about 0.073 / 0.152 and derivative error about
0.359 / 0.864 for GELU / SwiGLU, far above the fixed 0.05 / 0.10 gates. Narrow
fitting peaks near 24.2 MiB, saving 41-43% versus full profiles, but the value-only
control has the same memory. The complete pipeline reaches 117.908 MiB with
teacher capture included. Fresh-process inference separately saves 31-35%
allocation; no language experiment or complete quality/VRAM result is earned.

Independent audit verifies **648 metrics, 216 endpoint states, 72 selections**,
144 training records and exact input/target recapture for all 18 fresh datasets.
The datasets use 960 windows disjoint from H105/H107, still from the teacher
training corpus and the same two teachers, both trained with seed17. The three
sampling seeds do not establish teacher-training replication. An initial
pre-protocol path-type failure is preserved; no scientific training was repeated.

Close both fixed initializer-plus-value-learning recipes. Do not add steps,
rates or derivative terms automatically. Any different learning objective or
upstream-sensitivity diagnostic needs its own hypothesis and cost qualification.
The active code, tests, two model folders, five variants and eight recipes remain
unchanged; the previous 116-test pass is referenced separately. The full goal
remains unmet. [Report](sobolev_learning_results.md), [plan](sobolev_learning_plan.md).

## Previous: derivative-aware calibration qualifies as a component

H108 produces **432 compressed FFNs with zero SGD updates** in 231.87 seconds.
It keeps selected teacher neurons and fits a dense output projection to both
values and directional derivatives. All three widths pass the frozen comparative
component gates on GELU and SwiGLU, including every sampling-seed aggregate.
Paired derivative-error reductions are **6.83-16.15%** versus value-only selection
and fitting, with improved output error. The crossed ablations attribute about
**88-90%** of the combined absolute derivative gain to the readout objective alone.
Retain derivative-aware calibration as **PROMISING COMPONENT**.

None of the complete recipes meets the absolute quality/memory gate. The primary
models use **70.80% / 70.87% fewer FFN parameters** and save **33.45% / 35.44%** of
local inference allocation, but output error is 0.1062 / 0.2355 and derivative
error is 0.3293 / 0.5218, above the fixed 0.05 / 0.10 limits. Half width also
fails quality. Three-quarter width saves only 6.6-6.8% inference allocation and
SwiGLU derivative error remains 0.1171. No language insertion is allocated.

These are local FP32 FFN reconstruction results using 18 reused development
input sets from two seed-17 teachers, three depths and three sampling seeds.
They are not independently trained teacher replications or NLL measurements.
Calibration peaks at 36.125 MiB; including prerequisite full-teacher capture
raises pipeline allocation to 117.908 MiB. No optimizer or training-memory
saving is tested. The export is an ordinary dense FFN with an output bias.

Independent auditing verifies **864 scores, 432 exports/readouts**, all 18
statistics sets and 432 sampled actual greedy marginal gains. Thirteen isolated
qualification groups pass. An initial teacher-reference batch-partition mismatch
and its source/logs are preserved; recovery restores only the teacher's original
batch size, keeps all tolerances and student rescoring unchanged, and passes.
No maintained code/test changes; the previous 116-test pass remains separate.

The next justified question is whether derivative-aware readout initialization
improves actual narrow-FFN learning on fresh data versus value-only initialization,
with preparation and training costs charged. This needs a new frozen protocol.
Do not assume the selector is necessary, relax the current gates or claim a new
activation/theorem. The active tree remains two model folders, five variants and
eight recipes; the broader VRAM/parameter/quality goal is unmet.
[Full report](sobolev_selection_results.md), [frozen plan](sobolev_selection_plan.md).

## Previous: affine residuals pass capacity screening but fail learned compression

H106 checks 126 finite-data residual-rank bounds before training. An explicit
full-rank affine path qualifies GELU h256 and SwiGLU h170 corrections; smaller
h128/h85 recipes fail the fixed allocation gates. This follows from projection
and truncated SVD, not a new theorem or a learned-model result.

H107 then completes **288 fresh fits, 172,800 updates and 144 selections** in
12.98 minutes. Both candidates are rejected against six parameter-matched
conventional controls. They save **70.78% / 70.84% of FFN parameters**, but their
paired reconstruction error is 35.55% / 56.61% above SiLU on the GELU teacher and
38.78% / 23.52% above ordinary SwiGLU on the SwiGLU teacher. Both lose the strongest
control in every sampling-seed aggregate and every tested depth. Native raw-input
inference costs **1.64-2.32x narrow GELU**. No language trial or integration is earned.

Student training peaks near 23.95 MiB versus 43.15 / 41.30 MiB full-reference
profiles, but ordinary narrow controls already use 24.17-24.28 MiB. Raw inference
needs about 10.94 MiB for the candidates versus 10.69 MiB for ordinary GELU.
The entire compression pipeline peaks at **117.91 MiB** because it captures the
full teacher. These are PyTorch tensor allocations, not whole-process VRAM or
full Transformer training. Local memory savings do not establish preserved quality.

Independent audit recaptures every row in all 18 datasets, checks **864 scores**,
144 ridge initializations and all 144 exports/selections. The largest score
discrepancy is 2.64e-8. **116 maintained tests pass.** Separate native Windows
audit-import and plotting failures are preserved; recorded postprocessing recovery
passes without retraining or changing frozen sources/thresholds. Both teachers
have one training seed; three fresh sampling seeds are not teacher replications.

Close these fixed affine recipes. A capacity bound can eliminate inadequate
classes, but cannot certify learnability, useful memory savings or NLL. The
active tree remains two model folders, five variants and eight recipes. The
broader VRAM/parameter/quality goal remains unmet.
[Fitting report](affine_residual_fit_results.md),
[capacity proof and result](affine_residual_capacity_results.md).

## Previous: real FFN inputs reject energy projection; a rank limit explains part of the failure

H105 completes 378 local comparisons in 82.18 seconds with zero SGD updates.
All three energy-based projection recipes fail against simpler input PCA and
activation-aware linear compression on existing full GELU/SwiGLU checkpoints.
At rank 64, unwhitened energy has normalized output MSE 0.6146 / 0.6492, versus
0.1596 / 0.4279 for the linear-response control. Whitening and affine-residual
energy do not repair the signal. No fitting budget is allocated to these recipes.

A post-screen capacity diagnosis finds a stronger constraint: a final global
rank-64 output bottleneck has mean normalized error floors **0.1098 for GELU
and 0.2125 for SwiGLU** on these datasets, irrespective of the input nonlinearity.
Every one of the 18 rank-64 cases exceeds the 5% gate. Thus that absolute gate
was unattainable for the whole tested output-rank class. This was learned after
the screen; it must not be portrayed as an initializer-specific failure.
Energy's additional regression versus the simpler controls is independently
measured. Check rank capacity before the next compression grid.

The empirical bound uses centered output singular-value tails plus a conservative
BF16-to-FP32 triangle-inequality correction. All 54 bounds match a separate direct
SVD calculation. It constrains local teacher-function reconstruction, not language
NLL, a surrounding Transformer, structured full-rank maps or training from scratch.

Independent audit recaptures every token in all 18 pair datasets through the full
decoder and verifies 378 scores, bases and counts. **116 tests pass.** Data varies
across three calibration seeds, not three independently trained teachers. Only
training-cache windows are used. About 79% fewer FFN parameters does not qualify
a quality/VRAM result; collection retains the full teacher at 107-109 MiB peak.

Close this fixed global projection branch. The next architecture should retain
adequate output rank, or reduce memory without this rank restriction. The active
factory remains two model folders and five variants. The full goal is unmet.
[Report and proof](real_subspace_results.md), [frozen screen](real_subspace_plan.md).

## Previous: feature discovery helps a small cubic FFN, but promotion fails

H103/H104 completes 36 spectral estimates and 408 fresh synthetic fits, across
four Gaussian target families, three independent dataset seeds and two rates.
All 816 checkpoint scores and 204 rate selections pass independent verification.
The feature-discovery population calculation and oracle capacity checks are
scoped proofs, not new convergence or universal stability theorems.

The bounded-initialized cubic factorization uses 66,048 parameters, **94.41% fewer**
than full GELU/SwiGLU. It improves paired aggregate MSE by 41.01% / 15.97% versus
those equally initialized full controls and wins each task's mean. It also beats
every tiny control in each seed's aggregate, but **loses cubic to tiny cubic by
47.82%** and takes **2.12 times narrow GELU's inference time**. Its fixed gates fail.
The GELU factorization additionally fails cubic-learning and full-model quality.

Initialization uses a training-label energy covariance to estimate a rank-32
input subspace. Including its GPU workspace, the cubic candidate's peak tensor
allocation is **30.77 MiB**, saving **25.02% / 22.68%** versus full GELU/SwiGLU.
Training-only peaks save roughly 51-52%, but that is not the pipeline saving.
Equally initialized narrow/tiny controls share the 30.77 MiB pipeline peak.

Neither recipe earns automatic language training or integration. Constant-norm
one-hot language labels give zero energy signal, and the Gaussian assumptions
favor this synthetic screen. The next distinct question is whether useful
training-only subspace information exists on real FFN inputs/teacher outputs,
with all teacher, preprocessing, memory and inference costs charged. That needs
its own qualification before allocation. No long-run or broad task win is known.

Report/test runtime failures are preserved separately; no training or audited
score is replaced. The active tree remains two model folders and five variants.
[Full report](spectral_fitting_results.md), [H103 proof/plan](spectral_discovery_plan.md),
[frozen H104 fitting plan](spectral_fitting_plan.md).

## Previous: VRAM is primary; a scoped execution result qualifies

The user's September 10 clarification makes measured VRAM the primary objective;
the original parameter/quality goal remains separate and unmet. H101/H102 test
token chunking with equally checkpointed models, without changing parameters.

Narrow GELU uses **21.01% less training VRAM at context 128 and 32.23% less at
context 512**, across three seeds. Updates cost 7.45-22.51% more; final NLL changes
by at most +0.189% after 12 updates. Both shapes pass the frozen scoped gates.
Retain loss chunking as an experimental option; it is not a default or an
architectural promotion. Long-run learning and inference benefits are untested.

H101 stops after 22 cases: full GELU chunking fails training fidelity despite
local gradient checks. Fresh-process replay and checkpoint rescoring reproduce
the failure. H102 completes 36 cases; all 38 saved endpoints, including diagnostic
replays, rescore exactly. Full SwiGLU fails all-seed timing gates. FFN chunking
adds cost with little memory benefit and is closed at this fixed setting.

[Full report](token_memory_results.md), [initial plan](token_memory_plan.md),
[scoped replication](token_memory_followup_plan.md),
[machine-readable evidence](../results/token_memory_replication_v1/result.json).
The active factory still has two model folders and five variants. The next
architecture must beat memory-aware controls and undergo actual trajectory
checks; local algebra cannot certify numerical training stability.

## Previous architectural state

**The gold target remains unmet. H078 fails its single-seed duration gates.**
The latest completed longer-budget language experiment is matched-budget BlockShuffle GELU h3264, with
2,801,664 FFN / 9,099,648 total weights: 70.3125% FFN and 42.1700% total reduction.
This fixed duration recipe is closed. Further training needs a distinct, justified hypothesis.


## Latest result: three-way interactions fail despite representational capacity

H100 tests raw and bounded products of three learned projections on fresh
Gaussian tasks with hidden rotated directions. Both have 351,276 parameters,
70.27% fewer than full GELU. Both are rejected: aggregate MSE is 2.079/4.543 times
narrow GELU and native update time is 1.562/1.881 times its cost.

The 312-run screen uses four tasks, three seeds, two rates, batch 256 and 300 updates,
taking 6.97 minutes. All 30 qualification checks and 177 saved pairs pass. Independent
audit reproduces 624 checkpoint scores and 156 rate selections exactly.

A privileged construction proves raw products can represent the cubic target;
randomly initialized training does not find it adequately. A plain cubic ridge
partially learns cubic (18.02% less error than zero prediction) but misses the
50% reduction gate. Low final cubic feature alignment motivates a distinct
feature-discovery investigation, not further refinement of rejected products.

Post-audit report imports fail separately; preserved traces and a checksum-checker
correction lead to successful NumPy diagnostics under Python 3.12.9. Original
training and the independent Torch audit are unchanged. No active model is added.
The language, convergence, compute, scale and broader-data goal remains unmet.

[Full results and failure record](triadic_interaction_results.md),
[frozen plan](triadic_interaction_plan.md),
[audited result](../results/triadic_interaction_v1/result.json).

## Previous result: nonlinear residual learning, but no activation promotion

H099 completes 264 fresh runs: eleven forms, four Gaussian nonlinear-residual
tasks, three seeds and two rates, each with 300 updates and batch 256. The study
takes 5.76 minutes. All 25 qualification checks and 163 saved tensor comparisons
pass; independent audit reproduces 528 checkpoint scores and 132 selections.

A four-parameter learned rational mixture uses 351,052 parameters, 70.29% fewer
than full GELU. Aggregate MSE improves 12.39% versus narrow GELU and 13.79%
versus narrow SwiGLU, but only 0.060% versus established StarReLU. It loses one
seed to StarReLU, regresses on individual tasks and costs 40.49% more native
update time than narrow GELU. Its fixed recipe is rejected at this budget.

Nonlinear learning beyond affine structure is demonstrated on three tasks.
Cubic remains near zero-predictor error for every ordinary model, although the
privileged known-feature readout learns it. This exposes a feature-discovery or
optimization question, not evidence that a new activation solved cubic learning.
All full and narrow models have biases and start with zero output; this is a
changed, explicitly documented assay, not a direct continuation of H094 scores.

Learned curves and gradients are recorded. No active model is added and no
kernel or longer run is allocated to the rejected mixture. The full parameter,
language NLL, compute, convergence, scale and broader-data goal remains unmet.

[Complete findings](nonlinear_residual_results.md),
[frozen plan](nonlinear_residual_plan.md),
[audited result](../results/nonlinear_residual_v1/result.json).

## Previous result: the affine control closes the older general lead

H096-H098 fits ordinary linear and affine controls to exactly the training
samples seen by each H094 run. The even-feature recipe is only **1.724% better
than affine least squares** in aggregate, missing the frozen 2% material-benefit
gate. It wins on piecewise targets by 12.60% but loses on smooth, oscillatory and
multiplicative targets. Its three small aggregate seed wins remain recorded.

The affine control needs 147,840 coefficients, versus 350,208 used in the neural
training and 282, 624 after folding. The previously measured native speed deficit
also stands. Close the general nonlinear-benefit claim for this fixed recipe;
retain the piecewise result as scoped evidence, without kernel refinement or
automatic longer training. The original 12.92% advantage over narrow GELU is
numerically correct but does not survive as a sufficiently strong general claim.

All 24 least-squares fits are independently verified, including reconstructed
sample counts/statistics, 96 CPU score checks, 168 recomputed parity evaluations
and 12 saved odd-linearity pairs. H096's native access violation is preserved;
H097 did not reproduce it, and H098's explicit CPU-solver recovery passed the
unchanged mathematical/statistical checks. No new neural optimizer update ran.

The next assay must measure nonlinear residual learning beyond affine structure,
include a representable positive control and stronger conventional activations,
and separate representational limits from short-budget optimization. The full
parameter, NLL, compute, duration, scale and broader-data goal remains unmet.

[Complete findings](affine_falsification_results.md),
[frozen diagnostic](affine_falsification_plan.md),
[recovery](affine_falsification_recovery_plan.md),
[audited result](../results/affine_falsification_recovery_v1/result.json).

## Historical H094-H095 interpretation: superseded by the affine check

The three richer rational/Hermite/trigonometric pair recipes are closed at the
fixed budget. They reduce aggregate reporting MSE by7.21-9.31% versus narrow
GELU, but a simpler duplicated-even control is stronger. The additional odd
basis does not earn its parameter allocation in this experiment.

The even control lowers aggregate MSE by12.92% versus narrow GELU and
8.89% versus full GELU at300 updates. This is a post-selection
synthetic lead, not a validated language architecture. A direct-GELU control
already improves7.58% versus narrow GELU, so gains cannot be assigned entirely
to a new nonlinearity. Core-linear is slightly better on the multiplicative
task, and the oscillatory task remains near the zero-predictor error.

The duplicate control folds exactly in real arithmetic:
A*x + V*E(U*x) + W*E(U*x) = A*x + (V+W)*E(U*x),
where E(z)=z^2/(1+abs(z)). Twelve selected checkpoints pass FP32 output/input-gradient
checks and reporting rescoring after folding; maximum absolute MSE change is
below1e-8. One FP64 function/Jacobian check and26 saved tensor-pair rechecks pass.

**Training used350,208 weights (70.31% fewer than full). Folding reduces stored
and inference weights to282, 624 (76.04% fewer).** It is not evidence that training
from scratch with282, 624 weights reproduces these results or AdamW trajectories.

Native FP32 batch 256 inference is0.261 ms for folded even versus
0.297 ms for duplicate,0.129 ms for narrow GELU,
0.177 ms for full GELU and0.226 ms for full SwiGLU.
Folding lowers its own inference time by12.11%,
but it remains2.02 times narrow GELU. The compute
target is therefore unmet. This is a native inference measurement, separate
from the H094 training-time gate.

H095 retained the folded even-feature source as an unvalidated lead. H098 now
closes its general nonlinear-benefit claim; source and scoped results remain archived.
The three richer fixed recipes remain rejected. No test of a language corpus,
convergence, larger scale or broader real data occurred in these rounds.

[Detailed fitting results](input_basis_results.md),
[folding plan](input_basis_fold_plan.md),
[folded summary](../results/input_basis_fold_v1/summary.json),
[compact release receipt](evidence/input_basis_release.json).

## Latest completed geometry study (H088?H090)

**All four learned geometry recipes are rejected at the fixed fitting budget.**
The two pair-twist initializations finish 0.715% and 1.298% above narrow GELU in
aggregate held-out MSE. Bezier1P and four-group Bump2P finish 6.676% and 6.460%
above it. Pairwise coupling is mathematically valid but does not earn practical
promotion: native updates are about 2.25 times slower than narrow GELU.

The fresh grid completes 336 runs: four synthetic task families, three seeds,
two rates, 600 updates and batch 256. It takes 13.29 minutes including data,
checkpoint and plotting work. All 27 recovery qualification checks pass, and an
independent audit exactly reproduces all 336 selection and 336 reporting scores.
The original BF16 failure remains preserved; the explicit precision correction
changes BF16 rounding without changing the planned FP32 fitting arithmetic.

Learned curves move away from initialization, but they do not replace the
removed width successfully. The four-shift GELU control lowers aggregate MSE by
3.227% versus narrow GELU while remaining 11.18% worse than full GELU. Keep it as
a comparator, not a promoted new architecture. No new active model folder is added.
[Results and curves](neuron_geometry_results.md),
[precision failure/recovery](neuron_geometry_precision_results.md),
[independent audit](../results/neuron_geometry_audit_v1/result.json).

## Current direction and retirement (September10)

H087 archives native rational BlockShuffle after repeated memory failures.
Dense full/narrow controls and plain BlockShuffle remain active comparison tools.
[Retirement plan](release_cleanup_plan.md) and [artifact policy](ARTIFACTS.md)
preserve the research history without committing large local tensors.

H086 passes16 native sparse-operator checks, but the first accelerated inference
case stops with CUTLASS unsupported. It has no earned training allocation.
[Complete partial-result report](compact_sparse_operator_results.md).

The coupled feature-pair mechanism was tested in H090 and its fixed recipes
are now closed after failing the promotion gates.
See [direction, equations and prior work](research_direction_2026_09_10.md).
The historical source-preservation and108-test statements below describe their
original rounds; H087 intentionally changes the active tree and records a source
snapshot for reproducing those older rounds.

## Latest controlled offset fitting

H085 completes 240 fresh runs and all independent score checks pass.
Learned offset changes selected aggregate held-out MSE by -13.149%
versus plain; offset with the fixed first-factor LR correction changes it by
-14.529%. Neither offset recipe earns full-model resource qualification. These fixed candidate recipes are closed.
The broad research goal remains unmet.

All240 selection and reporting scores and48 resets reproduce exactly. Both
selected-recipe and matched-rate conditions are evaluated against fresh full,
narrow, gain and optimizer controls. [Completed study](latent_offset_fit_results.md).

## Latest mechanism result

H084 separates learned internal gain and offset without new training. Removing
gain multiplies aggregate checkpoint MSE by 2.972; removing offset multiplies it
by 1.843. Gain alone does not add represented functions, because it can be
absorbed into the first factor. An offset can give an isolated SwiGLU a nonzero
linear response at zero. These are local algebra and checkpoint results.

All 24 independent new reset scores match exactly. Folding the affine operations
preserves FP32 held-out error within a relative 1.7e-8, with explicit bias-buffer
storage. No BF16, speed, training-causality or full-model benefit is established.
No candidate is promoted. [Completed analysis](latent_affine_mechanism_results.md).

## Latest completed nonlinear fitting study

Both internal curves improve synthetic fitting over plain BlockShuffle, but
neither qualifies: tanh changes aggregate held-out MSE by -5.757%
and sine by -7.069%, while the equal-count affine control changes
it by -15.034%. Both curves lose to affine and narrow GELU.
The positive-control assay passes. These fixed recipes are closed; no language
or full-model resource run is earned.

H083 completes all 192 cells: four permuted nonlinear tasks, two rates, three
seeds, 600 updates and batch 256. The curves add 48 weights per FFN. All eight
recovery checks pass; independent analysis exactly reproduces 192 selection
scores, 96 selected held-out scores and 36 resets. Resetting learned curves
worsens error, but does not explain the training advantage over plain.
No active model is added. [Completed result](latent_activation_recovery_results.md),
[original adapter failure](latent_activation_results.md),
[scalar proofs and limits](latent_activation_theory.md).

## Latest completed short screen

H081 completes all 24 fresh 200-update WikiText-2 runs and all final checkpoint
rescores match exactly. BlockShuffle calibrated: fails fixed screen; BLAST SwiGLU: fails fixed screen; BLAST GELU: fails fixed screen.
This is one-seed short-budget evidence; longer, multi-seed, scale and broader
data requirements remain open. [Complete screen](blast_learning_screen_results.md).

| Model | Selected LR | Final NLL | Peak MiB | Mean update ms | Clip % |
|---|---:|---:|---:|---:|---:|
| Full SwiGLU | 0.0012 | 5.907694 | 384.56 | 102.25 | 10.5 |
| Full GELU | 0.0012 | 5.879278 | 393.25 | 87.15 | 12.0 |
| Narrow SwiGLU | 0.0012 | 5.912505 | 279.45 | 93.05 | 8.5 |
| Narrow GELU | 0.0006 | 5.902164 | 279.02 | 91.39 | 17.5 |
| BlockShuffle legacy | 0.0006 | 5.970604 | 279.77 | 126.89 | 16.0 |
| BlockShuffle calibrated | 0.0012 | 6.073010 | 279.77 | 139.54 | 7.5 |
| BLAST SwiGLU | 0.0006 | 6.165411 | 281.20 | 153.90 | 38.5 |
| BLAST GELU | 0.0006 | 5.951733 | 280.95 | 125.34 | 11.0 |

## Latest operator qualification

[H080](blast_operator_recovery_results.md) qualifies a conventional BLAST comparator:
GELU h3200 and SwiGLU h1984 each have 2,801,664 FFN / 9,099,648 total weights by
count. All 34 checks pass; independent analysis verifies 32 tensor files,
114 exact checkpoint pairs, analytic derivatives, four exact same-width embeddings
and four matrix witnesses. H080 itself contains no learning or full-model resource
measurements; H081 is the first language allocation.

[H079](blast_operator_results.md) remains incomplete because of damaged artifacts.
One explicit recovery preserves numerical tests and adds durable readback; all
24 readable original tensor payloads match. A separate scalar-norm audit mismatch
is corrected on the original devices without changing thresholds. There are zero
optimizer updates, corpus targets or Transformer runs. Only a separately frozen
learning comparison is earned; no active model is added.

For data, batches, duration and learned-activation results, see the
[plain-language progress overview](PROGRESS_OVERVIEW.md).

## Latest complete longer-budget language comparison

The [H078 report](ungated_duration_results.md) contains all six fresh seed-17
trials at 3,200 steps. Models retain H077's fixed recipes, with rates originally
selected under H076's equal three-rate short search. Only steps and logging change;
warmup grows to 320. These are fresh schedules, not checkpoint continuations.

| Form | Fixed peak LR | Final NLL | Peak MiB | Mean update ms |
|---|---:|---:|---:|---:|
| Full SwiGLU | 0.0012 | 4.105417 | 384.558 | 92.600 |
| Full GELU | 0.0012 | 4.127881 | 393.245 | 85.931 |
| Narrow SwiGLU | 0.0012 | 4.153048 | 279.454 | 90.440 |
| Narrow GELU | 0.0006 | 4.205838 | 279.017 | 87.550 |
| BlockShuffle SwiGLU | 0.0006 | 4.144306 | 279.767 | 121.954 |
| BlockShuffle GELU h3264 | 0.0012 | 4.242014 | 280.017 | 104.317 |

Matched's relative NLL differences are +2.7649% versus full GELU,
+3.3272% versus full SwiGLU,
+0.8601% versus narrow GELU and
+2.1422% versus narrow SwiGLU. Negative favors matched.
It exceeds both full-control NLL allowances and fails to beat either narrow control. Each full-control allowance is 1%; BOTH narrows must be beaten strictly.
The separate 0.2% narrow margins remain descriptive.

Versus plain, candidate NLL changes by +2.3576% and mean update time
by -14.46%. These are one-seed, sequential laptop-GPU measurements.
No confidence interval or replicated long-duration conclusion follows.
Late-plateau diagnostics pass 0/6 models; the report gives each final-800
loss change. Plateau screening does not establish optimal convergence.

Peak allocation includes the CUDA corpus cache and validation. Time averages
3,190 updates after ten timing-warmup updates; optimizer warmup is separately
320. Full-sequence inference is not autoregressive serving. Heavily reused
WikiText-2 development validation is not an independent holdout.

The [H077 three-seed comparison](ungated_lm_replication_results.md) remains
shorter-budget evidence: all 18 fresh 800-step trials pass each seed's primary
gates for matched GELU, mean NLL 4.834511 +/- 0.013382 (sample SD), but it beats
plain only in 1/3 seeds with a 0.0959% mean gain. None of its 18 plateau screens pass.

## Retained repository scope

After H087 retirement, the active tree has two model folders, five registered variants and eight recipes.
Experimental GELU uses the existing ungated operator with an isolated adapter;
it is not a duplicated model folder or new registered variant.

| Component | Role and limitation |
|---|---|
| [dense_ffn](../src/dense_ffn/README.md) | Four full/narrow GELU/SwiGLU controls for fair quality and resource comparisons |
| [blockshuffle_ffn](../src/blockshuffle_ffn/README.md) | Structured operator and plain SwiGLU reference; supports the isolated GELU experiment |

All 171 original language/profile runs remain, with 21 H076, 18 H077 and six H078 trials
isolated beside their protocols. Discarded architectures and obsolete drivers
are in the [archive](archive/README.md). Historical current-state pages are saved
in the per-round before-documents archives; detailed decisions stay in the
[complete ledger](idea_bank.md) and linked reports.

## Findings that still constrain the next step

| Evidence | Decision that remains in force |
|---|---|
| [H051 longer plain replication](long_duration_replication_results.md) | 3,200-step mean NLL 4.147443 is 1.256% above full SwiGLU; longer-training goal fails |
| [Affine matched-rate correction](affine_rate_replication_results.md) | Only 0.101% mean benefit, wins in 2/3 seeds; earlier larger claim was rate-confounded |
| [Native rational activation](learnable_activation_results.md) | 200-step NLL 5.898003, but 883.17 MiB peak; learned-shape reset barely changes quality |
| [H063 native](native_recompute_results.md) / [H066 staged](staged_resource_results.md) recomputation | Numerical fidelity passes, rational memory gates fail; no automatic partition or language repeat |
| [H061 additive low-rank screen](additive_block_lowrank_screen_results.md) | Quality gates fail; implementation retired in [H062](additive_retirement_results.md) |
| [H068 conditioned activation fitting](token_activation_fit_results.md) | Static/dynamic gains miss the frozen 2% gate; richer routing receives no automatic allocation |
| [H069 factor balance](factor_balance_results.md) | Exact local preservation but insufficient measured imbalance; no optimizer experiment earned |
| [H071 rotated-shuffle fitting](rotated_shuffle_fit_results.md) | Both tested rotations fail learning gates despite a valid local rank witness |
| [H072 local GELU analysis](ungated_blockshuffle_results.md) | Linear odd-component restriction and scoped population error floor; no whole-network or learning guarantee |
| [H073 fitting](ungated_fit_results.md), [H075 resources](ungated_resource_recovery_results.md) | Qualified the language screen; synthetic gains and timings are not corpus or convergence proof |
| [H076 language screen](ungated_lm_screen_results.md) | Same-width GELU fails; matched passes with only 0.1167% gain over selected narrow GELU at 200 steps |

Bezier/quadratic, single-factor grouped, coupled/shared, paired-feature,
parallel/headwise/overcomplete, shifted/affine and failed rational compiler branches
remain archived. Local proofs survive within their stated assumptions; failures
do not justify automatic tuning or relabeling known components as novel.

## What remains before the research goal can be accepted

1. This fixed duration recipe is closed. Further training needs a distinct, justified hypothesis. Keep both full and both calibrated narrow controls, equal tuning
   effort, data policy and actual memory/runtime accounting. Duration-specific
   rate sensitivity and convergence remain unestablished.
2. Establish scale and broader-data results with multiple seeds. Reused development
   validation and fixed short-selected rates cannot prove those requirements.
3. Reproduce strong published structured comparators under the same task scope.
   The [primary-source comparator note](ungated_comparator_note.md) identifies
   BLAST and BTT-MoE distinctions and the original count-compatible BLAST budget.
4. Further learnable activation work needs a distinct justified mechanism, correct
   derivatives/initialization, practical memory and measured gain over the strongest
   relevant control. See the [activation guide](learnable_activation_domain.md).
5. Local representation proofs, measured learning, training speed and full-sequence
   inference remain separate claims. No arbitrary-data or global-gradient guarantee,
   novelty priority or automatic publication is established.

## Verification

Five H078 integration checks pass with zero optimizer updates/model forwards.
All six initial states reconstruct exactly, initial NLL/first training loss match
H077, sampler states and full validation order agree, and every final checkpoint's
BF16 NLL rescores exactly. Independent checks cover finite weights/moments, all
3,200 step records per run, counts, calibration and primary/descriptive decisions.

The [final audit](../results/verification/ungated_duration_final_v1.json) preserves
112 scientific sources, 67 frozen plans, 798 earlier fitting cells, 21 H074/H075
resource checkpoint pairs and all historical language evidence. H078 uses
19,200 updates / 39,321,600 training targets; independent rescoring adds
1,936,128 validation targets and zero updates. All scientific trials finish on
the first attempt. The previously passing 108-test active suite and its source
are unchanged; isolated integration checks do not inflate that count.

The [H080 final audit](../results/verification/blast_operator_recovery_final_v1.json)
checks 118 scientific sources, 69 frozen plans and the unchanged original H079
damaged/intact files alongside earlier language, fitting and resource evidence.
Its 34 isolated checks do not replace the unchanged 108-test active suite.
