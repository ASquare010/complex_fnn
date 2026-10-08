# Selected-model handoff COMPLETE (2026-10-08)

User ended research and selected the saved64:1 context512 model in BOTH repos.
Steps1/2 marked finished for implementation/handoff only,not universal lossless
compression or all original research targets. Latest999/1000short,202/206packed
checkpoint is now inference default;old weights/evidence retained. New exports
main dump/selected-residual64-512/encoder.pt and standalone artifacts/
selected-residual64-512/encoder.pt verified exactly against frozen model outputs.
Codec strips routed_layers only for the selected four-layer depth_route model;
rejects grown layouts rather than silently misloading. Notebooks load saved model.
Optional15000-ADDITIONAL-update continuation recipe in standalone config/train.json
and notebooks/train_residual64.py;44000to59000,lr3e-5,restored optimizer/sampler.
Training notebook opt-in RUN_TRAINING=False. Prepared/inspected,NOT launched.
Older curriculum moved to docs/history. Shared base modules retained as selected
model dependencies;historical sources/data/checkpoints/records preserved.
Tests:7passed,1optimizer-training test excluded;both-package frozen logits/vectors
and encoder export parity passed. Initial export check hit cwd output guard then
corrected;no native failure or optimizer updates. Previous native history remains.
See docs/selected_handoff.md and dump/residual64_handoff_verification.json.
# Plain residual64 LONG RUN COMPLETE (2026-10-08)

All10000 additional updates finished,44000 cumulative. Final=selected best at10000:
short999/1000 exact,99.9989191%tokens,NLL0.0011079569;packed202/206 exact,
99.9957132%tokens,NLL0.0011518618. Parent957/1000 and151/206. Same10456576total/
7327232encoder params,context512/span64/width256. Training426.7817/474MiB allocated/
reserved;50.22min summed update time excludes validation/preparation. No failures.
Controller completion audit passed full final/best validation replays,export parity,
source/checkpoint/record hashes. Independent pinned read-only saved-evidence rehash
verified result,last,best,encoder,persistent record,frozen source archive and all
10000 consecutive training updates. GPU idle0/8188MiB. No additional training.
Strong improvement from training alone;still not100% exact recovery. Single
inherited initialization,reused selected validation,short/packed share text,no
untouched test evaluation. Do not call universal lossless compression or infer
encoder bottleneck proved. Default checkpoint unchanged. See docs/residual64_long.md.
# Plain residual64 LONG RUN TRAINING (2026-10-07)

Launch verified: worker7312 completed100/10000 additional updates; hidden launcher
8612. Complete data identities and exact model/source/parent checks passed.
resume_receipt confirms restored AdamW(step4000),sampler and CPU/CUDA RNG.
Early423.08/470MiB allocated/reserved,GPU629/8188MiB;not final peaks. No launch
failure. Sources frozen. Progress dump/residual64-long-v1/progress.ps1. Budget
one x10000 additional,34000to44000 cumulative. No new quality claim or automation.
# Plain residual64 LONG CONTINUATION AUTHORIZED (2026-10-07)

Latest user explicitly requests only10M model for10000 updates. Interpret as10000
ADDITIONAL from latest957short/151packed final checkpoint,34000to44000 cumulative.
Budget one x10000,max1worker. Exact10456576params,4enc/1dec,width256,context512,
span64,code125;plain residual/BranchSigmoid/RoPE unchanged. Restore full AdamW
state and sampler/CPU/CUDA RNG;lr3e-5,existing mixture continues. No larger models,
no probes,extra training or inference promotion. Preserve past evidence/weights.
Complete decoded identities,source/parent checks,finite gradients,memory guards,
final/best validation/export/hashes retained. Prior separate preflight/initial
full validation waivers remain,not passed. PinnedCPU16,uv run --no-sync. Prior
native failures unresolved;no automatic new native/data retry. Metadata-only
PermissionError retry retained. Sources dump/residual64-long-v1;output standalone
artifacts/runs/residual64-long-10000-v1. See docs/residual64_long.md. No automation.
Prepared; launch observation follows.
# Residual64 growth COMPLETE (2026-10-07)

All three finished4000 additional updates. Final short/long exact: continue_plain
957/1000,151/206; encoder_plain942/1000,153/206; balanced_plain959/1000,169/206.
Packed-selected best: continuation958/155 at3000; encoder963/154 at3000; balanced
959/169 at4000. Short-only selected peak encoder964 at2500 is not its saved best.
Parent904/103 had30000 inherited updates; these endpoints have34000 cumulative.
Same-size model previously progressed781 at26000 to904 at30000 to957 at34000.
Allocated/reserved trainingMiB:424.58/474;496.89/560;497.22/558 respectively.
Update-loop totals1230.21/1424.02/1395.01seconds exclude evaluation/preparation,
with shared GPU contention and different launch times, not isolated speed tests.
Pinned read-only saved-evidence check verified all three checkpoint/export/result/
persistent-record/source archive hashes,4000-step sequences, matching target hashes
and true final/best replay,encoder parity,data identity receipts. No new model replay.
V1 controller failure is preserved; per-arm continuation audit completed independently.
GPU idle0/8188MiB. No new continuation launched or additional budget consumed.
One inherited seed,reused validation,no test evaluation,no100%exact recovery.
User asks about4000 versus8000; latest batch was4000 each, not8000 each. See
 docs/residual64_growth.md. No default promotion or automation.
# Residual64 growth THREE WORKERS TRAINING (2026-10-07)

Verified all three independently passed complete decoded identities and exact
initial parent-logit/vector transfer checks, then advanced optimizer updates.
Latest observation: v1 continue_plain25076 at500/4000 validating; v2 encoder_plain
7196 at50/4000; balanced_plain25352 at25/4000. GPU2045/8188MiB observed. Larger
training peaks494.42/546 and494.90/538MiB allocated/reserved so far, not final peaks.
No worker failures in repaired v2. V1 file-lock failure remains preserved below.
Training Python sources frozen. Progress: dump/residual64-growth-v2/progress.ps1.
No quality result yet; no extra budget, no default checkpoint changes, no automation.
# Residual64 growth controller file-lock repair (2026-10-07)

V1 controller3984 failed status.json rename with Python PermissionError/WinError5.
Larger workers2568/10480 passed identities and transfer but exited0updates after
seeing controller failure. Same-size continue_plain25076 is live; DO NOT restart.
No new native or decoded-data failure. One diagnosed v2 repair authorized within
requested training: launch only encoder_plain/balanced_plain after dead-process
verification; bounded metadata-rename PermissionError retry, no optimizer retries.
V2 sources dump/residual64-growth-v2; output standalone artifacts/runs/
residual64-growth-4000-v2. V1 continuation stays in residual64-growth-4000-v1.
Same parent,seed43,target schedules and4000 each; total12000,max3workers unchanged.
V1 frozen sources/failed controller/zero-update worker evidence preserved. Final
report must combine v1 continuation and v2 larger arms with independent aggregate
schedule/hash/replay verification. Do not claim v1 controller passed. No automation.
See docs/residual64_growth.md. Prior native failures still unresolved, no blind retry.
# Residual64 modest growth AUTHORIZED (2026-10-07)

Latest user authorized plain-only training, including same-size longer continuation.
Three x4000 additional updates,max3workers: continue_plain4enc/1dec10456576params;
encoder_plain6/1 and balanced_plain5/2 each12554752(+20.07%). All span64,width256,
context512,code125. Same904short/103packed plain parent,30000 inherited updates;
fresh AdamW3e-5,sampler43,same existing25/25/25/25 short/packed/uniform/pattern data.
Preserve original four-layer depth router; added encoder blocks are ordinary
residual refinements after its mixture. New block output projections zero-init;
in-worker exact initial FP32 output/vector parity required,0 probe optimizer steps.
Same-size continuation explicitly authorized, not redundant old control retraining.
Full decoded identities,source/parent hashes,finite gradients,memory guards and
final/best audits retained. Previous separate probes/initial full replay waived,
not passed. CPU pin16/17/18; uv run --no-sync; no environment changes. Earlier native
failures remain unresolved, including unpinned CLR audit crash. No automatic new
native/data/worker retries. Preserve all prior data/weights/failed evidence.
Output standalone artifacts/runs/residual64-growth-4000-v1; sources
 dump/residual64-growth-v1. See docs/residual64_growth.md. No default promotion.
Budget12000 new updates only. No automation. Prepared; launch observation follows.
# Residual512 four-model batch COMPLETE (2026-10-07)

All four completed4000 additional updates;16000 total,zero worker failures.
Controller completion_verification passed: full final/best replays,encoder parity,
source/checkpoint/export/persistent-record hashes,complete update/validation
sequences,matched target schedules and initializations within each ratio.
All best checkpoints are4000 endpoints. No more training authorized.

Final packed206 exact/token%/NLL; original short1000 exact/token%/NLL:
s64_plain:103/99.8521/0.011889;904/99.8876/0.010493.
s64_margin:107/99.8650/0.011393;904/99.8887/0.010094.
s128_plain:0/22.8261/4.854762;93/50.0465/3.208889.
s128_margin:0/23.1240/4.826250;122/50.7458/3.158122.
Margin64 narrowly wins packed metric,plain64 ties short exact. Both64 retain
10456576total/7327232encoder parameters,423.61/472MiB allocated/reserved training;
parent423.39/474,781short exact. Improvement123short paragraphs also includes4000
extra updates,changed context/data mixture/optimizer; not isolated architecture
or context benefit. Both128 have10457600total/7327744encoder,422.88allocated,
472plain/474margin reserved.128 fails this recipe,not a theoretical limit.
One inherited initialization/reused selected validation; long and short views
share content; no fresh test evaluation. No100% recovery. Default plain residual
parent remains selected; no automatic model promotion by status request.

Additional independent PowerShell saved-evidence verification encountered a NEW
native CLR crash0x80131506/process0xc0000005 while serializing the final JSON
report; no independent completion receipt was saved. Do not claim that extra
audit certified, and do not automatically retry this native failure. No model
evaluation/optimizer work in this attempt; trained checkpoints untouched.
Controller's successful completion receipts remain available. New post-training
crash does not mean a training worker failed; runtime root cause still unproven.
See dump/residual512-affinity-v5/independent_audit_failure.md and
docs/residual512_screen.md. All prior failures/diagnostics retained. No automation.

# Residual512 runtime mitigation verified; FOUR MODELS TRAINING (historical,2026-10-07)

Latest user explicitly requested fixing the issues and running training. Bounded
zero-optimizer diagnostics found intermittent parsing/decoded-identity failures
after tokenizer use; import-only and some tokenizer controls passed. Pinning to
logical CPUs16/17/18/19 passed four concurrent real-import processes, each three
complete52000-row hash/decoded-identity checks (624000 row checks), clean exits.
This is a TESTED MITIGATION, not a proven tokenizer/CPU/hardware root cause.
No Python, tokenizers, torch, CUDA, parser or dataset replacements/updates.
One diagnostic output incorrectly had passed:true plus a later error; that
tokenizer_decode_only run is FAILED. Reporting bug corrected; raw evidence kept.
One debug-allocator diagnostic first had an argument syntax error (no work),
then completed after syntax correction. No model/probe/optimizer updates in any
diagnostic. All earlier native, parsing and policy failures remain preserved.

Applied process-local CPU affinity before heavy imports in new frozen sources
dump/residual512-affinity-v5. Controller/preparation CPU16; s64_plain16,
s64_margin17,s128_plain18,s128_margin19. Successful packed512 preparation:
10303train/206valid/199test, same original split-disjoint cached paragraphs.
Every worker passed independent complete packed and original train/valid decoded
identities, source/parent hashes, strict weight transfer and matched initialization
within each ratio. All four now TRAINING, observed175-200 updates each; no new
worker failure. Hidden uv launcher3912, controller6908; real workers22548,
14388,2988,25712. Early peaks421.94/468MiB at64 and421.78/466 at128; total GPU
about2503/8188MiB,99% utilization. Final peaks/quality remain unknown.

Budget unchanged: four x4000 additional updates, max4workers,context512,
width256,4encoder/1decoder,lr3e-5,micro2xaccum4,sampler seed41,26000 inherited
updates; fresh optimizer every arm. Exact10456576total/7327232encoder at64,
10457600/7327744 at128.128 SVD map transfer retains53.41% encoder token-projection
energy and63.52% decoder energy; not a lossless transform. All body weights exact.
Margin0.25 fades final25%. No extra models, control retraining, or budget changes.
Separate model preflight/resource probes/initial full validation remain WAIVED,
not passed. Runtime caps1300allocated/1500reservedMiB per worker; all final audits
and in-worker data/finite-gradient checks retained. Do not change pinned sources
or automatically retry new failures. Preserve trained parent and all history.

Live output standalone artifacts/runs/residual512-screen-4000-v5; progress command:
powershell -NoProfile -File D:\Git\complex_fnn\dump\residual512-affinity-v5\progress.ps1
Launch verification/repair receipts in the source directory. Full packed206 and
short1000 validation every500; full final/best replays, checkpoints, encoder parity,
source/target/record hashes and complete4000-update schedules required at end.
Plain residual256 remains selected inference checkpoint until evidence supports
replacement. No fresh independent test-model evaluation; selected-validation and
single inherited seed limits persist. No monitoring automation created.
See docs/residual512_screen.md. Historical blocked entries below are superseded
only for this authorized repaired launch.

# Residual512 user-requested direct attempt v4 FAILED (2026-10-07)

Latest user explicitly requested "for just run the tranig". One new direct attempt
launched with hidden uv PID6360, same four x4000 budget,max4workers,context512.
No separate preflight/resource probes/initial validation. Python/controller ran,
but preparation exited1: json.loads ValueError, invalid literal for int() with
base10:'376'. No model workers, optimizer/probe/research updates. GPU0/8188MiB;
no Python/uv processes remain. Original split hashes still match. Runtime cause
unresolved; no automatic new failure retry or parser/data-identity bypass.
Environment unchanged. Preserve source snapshots, partial packed512-direct-v4,
logs/status/exit/failure receipt in dump/residual512-direct-v4 and standalone
artifacts/runs/residual512-screen-4000-v4. Sources frozen for this failed attempt.
Prior failures retained; entire four x4000 training budget remains unused.
No automation. Repeated direct launches have not repaired the parsing failure.
# Residual512 attempt after user disabled SAC: data parser FAILED (2026-10-07)

User showed Smart App Control Off and explicitly said "now try". Registry state0
confirmed; no prior Python/uv workers, GPU0/8188MiB. Hidden uv launcher13100
started the same bounded four x4000 direct attempt using uv run --no-sync.
Windows policy block is resolved for this launch; agent changed no security or
Python/CUDA environment. Controller started, but its single preparation process
exited1 at21:36:26PDT with Python json.loads ValueError: invalid literal for int()
with base10:'264'. This is a Python exception, not a new native-crash claim.
All original train/valid/test SHA256 hashes still match; independent .NET
System.Text.Json parsed50000/1000/1000 rows successfully. Runtime cause unresolved.
Do not bypass identities, substitute parsers in workers, or automatically retry
this new failure. No model workers or research/optimizer/probe updates occurred.
Partial paragraph-packed512-direct-v3 contains only tokenizer.json; retained.
Attempt sources frozen in dump/residual512-direct-v3 and standalone artifacts/
runs/residual512-screen-4000-v3/source. Data diagnostic, logs, exit/status and
process evidence preserved. Budget still four x4000,max4workers,context512;
all previous waivers and mandatory in-worker checks remain. No extra training.
No live training workers or automation. See docs/residual512_screen.md.

# Residual512 renewed direct launch BLOCKED by Windows policy (2026-10-07)

User again explicitly requested "strat traning directly", authorizing one new
direct attempt with the same four x4000 budget and previous check waivers.
Old controllers/workers verified absent; GPU idle. Sources copied to
dump/residual512-direct-retry with only fresh output/data paths. Hidden uv
launcher19628 used uv run --no-sync; Windows blocked the venv Python executable
before the controller could start: error4551, CodeIntegrity event3077 at
21:31:29PDT. Blocked file D:/Git/latent-text-compressor/.venv/Scripts/python.exe;
policy ID0283ac0f-fff1-49ae-ada1-8a933130cad6. This is an OS Application Control
block, distinct from previous native crashes. Do not bypass policy with another
launcher/path or weaken security. Requires authorized policy resolution.
No model workers, data preparation, research/probe/optimizer updates. Environment
unchanged. New output residual512-screen-4000-v2 contains blocked-attempt evidence
only. Prior attempts and partial datasets retained. Same parent,4000 each,
max4workers,context512; no additional budget. Sources/evidence preserved under
dump/residual512-direct-retry. No automation. See docs/residual512_screen.md.

# Residual512 direct launch ATTEMPT FAILED before training (2026-10-07)

Latest explicit user: "strat traning directly". This authorized one direct attempt
within the same four x4000 budget, waiving separate preflight/resource probes and
initial full validation (not passed). Complete independent decoded identities,
source/parent hashes, matching pair initialization, finite gradients, memory
guards and final validation/export audits remained required inside the workers.

Controller launched with uv run --no-sync, hidden launcher16920. Before any worker,
its sole preparation subprocess PID22944 crashed in python312.dll0xc0000005 at
21:11:04PDT, exit3221225477. No Python traceback in preparation.log. New partial
data directory paragraph-packed512-direct-v2 contains only tokenizer.json;
previous partial-v1 retained. Zero research/optimizer/probe updates, no model
workers launched, GPU0/8188MiB; only VSCode formatter Python processes remain.
No automatic retry after this new native failure. Environment unchanged.

Frozen attempted sources: dump/residual512-direct and standalone artifacts/runs/
residual512-screen-4000-v1/source. Failure/status/exit/log evidence retained in
that run; Windows events in dump/residual512-direct/failure_events.json. Same
four candidates, parent and budgets as below. Progress script:
dump/residual512-direct/progress.ps1. No monitoring automation. Runtime cause
still unresolved; do not describe this as a model-quality or memory failure.

# Residual512 four-arm screen PREPARED, runtime failure blocks launch (historical,2026-10-07)

User explicitly chose4000 additional updates per variant. Prepared s64_plain,
s64_margin,s128_plain,s128_margin,context512,width256,4encoder/1decoder,max4workers.
All warm-start audited plain residual64 final checkpoint (781/1000),SHA
1a9c57693b14258a2e090f4f27ecff8a2b69129cd4c2e12d3f19b8a882cecb35.
Inherited26000 updates; fresh optimizer for every arm,lr3e-5,sampler seed41.
Expected full params10456576 at64/code125;10457600 at128/code63.128 uses untested
deterministic SVD packing transfer; all other learned weights copied. No trained
result or resource pass claimed. Four x4000 is the entire authorized new budget.

NO new training workers, research updates or optimizer probes. Earlier preparation
dump/compression512-balance/preparation.log rejected valid JSON integer'290';
unchanged source hashes and independent .NET parser passed. Partial packed512-v1
contains only tokenizer.json, no manifest; preserved and never silently replaced.
Current preparation hit the existing-directory guard without parsing/retrying.
New default export command reached both-package encoder parity and saved receipts,
then Python PID18452 exited with native ucrtbase.dll0xc0000409 at21:00:45PDT.
Cause unresolved. Do NOT automatically rerun failed data/native work or launch
training; no bypass/new environment/sync. Native event/process evidence saved in
dump/residual512-screen. Only live Python processes found are VSCode formatter.
All earlier attempts retained. Current draft supersedes older512 draft for budget
and chosen parent, not failure history. Sources NOT frozen; probe has NOT passed.
See docs/residual512_screen.md. No monitoring automation.

Plain residual is now the configured product default in both repos. Active
residual.py and codecs support it; checkpoint retains its tested256 context,
span64,7327232encoder/10456576total params. Newly exported encoders agree exactly
on the sample; completion receipt precedes the native process-exit crash and is
NOT a clean-runtime certification. Old span32 weights remain available and have
the stronger1000/1000 historical validation result. Generic training defaults
select plain residual512; historical curriculum preserved in standalone config/
branch_rope_curriculum.json. This generic fresh recipe is not the authorized
warm-start screen and must not be launched in addition to it.

# Scaling pair20M/context1024/span128 COMPLETE (2026-10-07)

Both10000 fresh updates completed; final and best are the10000 endpoints.
Plain long97:0exact,28.4054% tokens,NLL3.394535;short1000:0exact,10.2761%,7.385770.
Margin long97:0exact,25.7271%,3.550404;short1000:0exact,17.9936%,4.992787.
Plain wins long-token recovery; margin wins short-token recovery; neither useful
exact compressor. Same19998208total/13672704encoder parameters,30714208training
tokens each. Training847.3413/990MiB plain and847.3418/988 margin allocated/reserved.
Full final/best replays,export parity,source/data/checkpoint/record hashes,all10000
updates and matched targets verified. Independent saved-evidence audit passed
against FROZEN sources; active residual integration drift explicitly disclosed.
No fresh model reevaluation used migrated sources. No new worker/native failure
during that completed batch. Two discarded probe updates;v1 UTF8 failure0updates.
One seed,reused selected validation; long/short views share the same held-out text.
Changes to size,depth,context,span,initialization and training history confound
causal attribution. No new training authorized by completion. See
docs/compression128_scale20m.md and dump/compression128-scale-v2/
independent_completion_verification.json. No run locks remain.

# Scaling pair20M/context1024/span128 RUNNING (historical observation,2026-10-07)

V2 both workers passed independent complete packed decoded identities, matching
initial-state hash and initial full97-sequence validation, then advanced through
10 updates each. Controller24492,launcher25420; no new worker failures observed.
GPU total2148/8188MiB. Early training peaks832.97/914-916MiB allocated/reserved;
actual future peaks remain guarded. Budget still10000 fresh updates each. Sources
now frozen. Progress: dump/compression128-scale-v2/progress.ps1.

# Scaling pair20M/context1024/span128: UTF-8 launch repair (2026-10-07)

V1 both workers passed complete packed identities and initial full validation,
then failed before update1 while checkpoint metadata read tokenizer.json using
Windows cp1252. UnicodeDecodeError, not native failure/data mismatch;0 research
updates. Owners verified dead; all v1 logs, locks, sources and initial metrics kept.
V2 is an explicit diagnosed serialization fix within the same two x10000 budget:
UTF-8 tokenizer reads, bounded error string, new output directory only. Computational
sources/data/initialization/schedules unchanged byte-for-byte; successful CUDA
probes reused with repair receipt. UTF-8 torch-save/load metadata roundtrip passed,
zero extra probe/model updates. No environment changes or blind retry loop.
Active sources dump/compression128-scale-v2; output standalone artifacts/runs/
compression128-scale20m-s41-10000-v2. Historical v1 remains failed. No native retries.

# Scaling pair20M/context1024/span128 authorized (2026-10-07)

User authorized two candidates at about20M,total10000 fresh updates each,new seed,
context1024,target128tokens/vector,parallel. Pair plain residual attention and
residual attention+margin0.25 fading final25%. Both seed41,fresh same initialization
and optimizer; no old weight transfer. Exact19998208 total/13672704 encoder params,
8encoder/2decoder,width256,heads4,code_features128,Branch Sigmoid+RoPE+depth routing.
One shared schedule warmup250 to3e-4 then cosine3e-5;micro1xaccum4,max4096tokens/update.
Size/context/span/seed/training history all change; not an isolated capacity test.
New data packs original split-disjoint cached paragraphs with newline,whole rows,
up to1024 tokens:4831train/97valid/93test. Can join source documents within split;
not a new coherent long-document corpus. Complete packed decoded identities passed
and independently repeated by workers. Same tokenizer;50%natural/25%uniform/25%pattern
draws shared by both. Report primary97 long sequences and separate1000 short rows;
test receives no model evaluation. Original validation selection limits persist.
Two max-length synthetic probes passed gradients/padding/shape/encoder parity;
2 discarded optimizer updates.679.14/772MiB plain,684.48/752 margin training peaks.
New per-worker guard3000allocated/3300reservedMiB; old474.6 cap superseded only for
this user-authorized size/context expansion. Initial wrong-env import failed with
missing rapidfuzz before updates; corrected cwd, no dependency changes/no native
probe failure. Earlier native failures remain unresolved; no auto worker retry.
Sources dump/compression128-scale; standalone artifacts/runs/
compression128-scale20m-s41-10000-v1. See docs/compression128_scale20m.md. Freeze sources
at launch. Full packed+short validation every500; final/best replay, encoder parity,
all hashes/target schedules required. No additional runs/budget beyond these two.

# Residual64 memory-paired comparison FINISHED: four complete, coverage rejected (2026-10-07)

Plain/difficulty/margin/combined completed8000 updates each. Coverage stopped after
optimizer update4869 at476MiB reserved,1.4MiB above declared474.6 ceiling; not retried.
Its last logged update4868, best unreplayed763/1000 at4500, no final/best completion
audit; excluded from winner ranking. Controller retains failed status due this arm.
Completed four source/parent/checkpoint/export/record hashes, full8000/16 validation
sequences, equal target/source schedules, final/best replays, encoder parity and
memory phases verified; independent saved-evidence audit passed. No new native
training failures. Controller24932/coverage26264 no longer alive; failure locks
preserved as stale evidence. GPU idle. Total36869 optimizer updates plus5 discarded
probe updates; no automatic continuation or extra training authorized.
Best exact winner margin806/1000 at7500,99.7536% tokens,NLL0.018572; final773/1000.
Other best/final: plain782/781,difficulty799/777,combined787/771. Plain wins fixed
8000 endpoint; margin's selected advantage is not a stable final-checkpoint win.
All four423.39/474MiB allocated/reserved training;238.95/378 validation; same
7327232/10456576 params,context256/span64/width256. Reserved+4.87% versus original452,
within5% cap,100MiB below previous574. All still fail100% recovery. One inherited
seed and reused selected validation; no fresh test evaluation. Same cached-ID
waivers retained, not claimed passed. Selected span32 package untouched.
See docs/compression64_memory_pairs.md and dump/compression64-memory-pairs/
reviewed_results.json; persistent ignored records/compressor64-memory-pairs-summary-v1.json.

# Residual64 memory-paired comparison launched (2026-10-07)

All five workers launched and passed their first optimizer update. Controller
PID24932; uv launcher24856. No launch failures; total GPU3106/8188MiB observed.
Current status is live; source files and budgets are frozen for this run.

Latest user requested residual attention + difficulty, margin and explicitly
coverage at near-original VRAM. Five arms plain,difficulty,margin,combined,coverage;
8000 additional updates each,max5workers,from same729/1000 saved parent+optimizer.
Same7327232/10456576 params,context256/span64/width256,lr3e-5. First four use exact
original parent sampler; coverage shuffles within source with same source schedule.
Difficulty/margin0.25 fade in final25%. Combined here excludes coverage.
New plain continuation is authorized to isolate extra-training benefit. No old
control re-run, no source/package changes, no extra experiment budget.
Five one-update synthetic probes passed;5 discarded updates,0 research updates.
419.65-419.85/450-462MiB training;238.83-238.95/338 validation. Evaluator tensor
release preserved16-row outputs exactly; same microbatch8. Clear unused cache
at phase boundaries for all arms. Runtime cap448.2772/474.6MiB allocated/reserved
(5% above saved426.9307/452); stop any crossing, no automatic retries. Full training
and validation peaks now counted; report phase details. Cached data hashes retained;
repeated decoded/initial full replay waiver carried forward, not claimed passed.
Earlier native failures unresolved; separate read-only PowerShell inspection
exited0xc0000374 in preparation, no model work; no native failure in CUDA probes.
Sources dump/compression64-memory-pairs; status standalone artifacts/runs/
compression64-memory-pairs-8000-v1/status.json. See docs/compression64_memory_pairs.md.
Completion requires full final/best replay, export parity, all hashes/schedules,
records, memory ceiling and all8000 updates. No extra training beyond these five.

# Combined/linear five-model batch COMPLETE (2026-10-07)

All five x8000 additional updates completed, no new worker failures. Saved source,
parent/checkpoint/export/record hashes, complete update/validation sequences, same
target schedules, full final/best replays and encoder parity verified by controller;
independent standard-library saved-evidence rehash also passed. No locks remain.
Best combined at7500:788/1000 exact,99.7406% tokens,NLL0.018383 (final786/1000,
99.7427%,0.017736). Other best/final exact: hard_fade786/769,linear_full784/773,
linear_tail776/759,margin_fade769/767. Parent729/1000,99.6606%,0.041006 had8000
fewer training updates; no plain continuation arm in this batch, so improvement
cannot be attributed to add-ons alone. Combined only2 paragraphs ahead of hard.
All7327232 encoder/10456576 full parameters,context256/span64/width256 unchanged.
Winner logged training peak424.87/574MiB allocated/reserved versus parent's426.93/452;
reserved is122MiB higher, so same-memory success is NOT established. Allocated peak
logging excludes validation phases; reserved includes retained allocator cache.
Preflight, repeated decoded identities and initial validation were explicitly waived,
not passed. Old native failures remain unresolved. One inherited seed/reused selected
validation; still not100% recovery, span32 remains selected reliable compressor.
No additional training authorized. See docs/compression64_variations.md and
dump/compression64-variations/independent_completion_verification.json.

# Combined/linear five-model batch launched; preflight waived (2026-10-07)

All five workers launched and independently advanced beyond300 updates. Controller
PID6296; worker PIDs22036/7704/9904/13924/10852 at this observation. No new worker
failures so far. GPU total3080MiB/8188MiB,99% utilization. Training-only peaks so far
422.04-422.11MiB allocated,456-458MiB reserved; these exclude later validation peaks
and do not establish final memory compliance. Same7327232/10456576 parameters and
context256/span64/width256. Progress: dump/compression64-variations/progress.ps1.
No completion or quality improvement is claimed yet.

Latest explicit user: "forget checks start training". Proceed with five x8000
additional updates,max5workers,from729/1000 saved winner. This supersedes the
previous prelaunch-check and repeated decoded-identity requirements for this
batch. Skip separate preflight, repeated re-tokenization and initial full replay;
do not claim those gates passed. Retain immutable data/source/parent hashes,
parameter/target-budget checks, finite gradients and full final/best validation
replays. Use cached IDs and ordinary decode for actual validation every500.
Sources frozen at launch; no further changes or automatic native/worker retries.
Frozen runner dump/compression64-variations/run.py; live status standalone
artifacts/runs/compression64-variations-8000-v1/status.json. Linear layouts must
remain explicit in new checkpoints/encoder exports. Keep prior crash evidence.

# Combined/linear variants authorized; launch BLOCKED (2026-10-07)

Latest user explicitly requested variations and training now, including compatible
linear attention. Declared budget5 x8000 additional updates,max5workers,from729/1000
continue/last.pt. Arms combined,hard_fade,margin_fade,linear_tail,linear_full.
All exact7327232/10456576 parameters,context256/span64/width256; no control retrain.
User-authorized data/runtime retry again crashed tokenizers.pyd0xc0000005 PID4064
at15:11:13PDT. Separate unaffected tensor/operator and synthetic gradient checks
passed; uncheckpointed linear variants exceeded allocated memory. A memory-saving
activation-recompute revision is prepared, but its synthetic-only check crashed
ucrtbase.dll0xc0000409 PID9076 at15:16:52PDT before output. No tokenizer retry in that
check, no optimizer steps, no research updates, no training workers. No automatic
retry/bypass of either new failure. Runtime cause remains unresolved; file hashes
match saved data. Frozen old sources/weights/environment unchanged. Final revision
correctness/memory, identities, checkpoint/export and full-validation gates remain
pending. New training budget is authorized, not launched. See docs/compression64_variations.md.

# Combined residual64 recipe PREPARED; native diagnostic failure (2026-10-07)

User requested one combination of residual attention, coverage, difficult tokens
and/or margin. Prepared one recipe from continue/last.pt (729/1000), with coverage,
milder difficulty0.25 and margin0.25, fading both over final25% of a future budget.
Same exact7327232/10456576 parameters, context256/span64/width256. Not trained;
no new update budget authorized. See docs/compression64_combined.md.
Read-only error-overlap diagnostic crashed during full decoded identity check:
tokenizers.pyd0xc0000005,PID11492,2026-10-07 14:55:42PDT. Zero model evaluations
or updates; no automatic retry/bypass. Logs/events under dump/compression64-combined.
Existing winner SHA still matches audited record. New recipe/tests are prepared
but NOT executed; no correctness or memory pass claim. Selected sources/weights
unchanged. Prior native failures remain unresolved.

# Compression64 continuation COMPLETE (2026-10-07)

All five x8000 additional updates finished. Source/parent/checkpoint/export/record
hashes, full update/validation sequences, initial/final/best full replays and
encoder parity verified; independent completion rehash passed. Winner ordinary
continue at8000:729/1000 exact,99.6606% tokens,NLL0.041006 (parent557/1000,
99.2769%,0.166150). Best exact: coverage723,hard715,coverage_hard729,margin708.
Final exact respectively707,690,728,708. Coverage_hard final has three fewer wrong
tokens than continue; margin has lowest NLL0.038694. No clear add-on exact-recovery
win. All7327232 encoder/10456576 full parameters,context256/span64/width256.
Winner426.93/452MiB allocated/reserved: allocated1.39MiB above parent, reserved
34MiB lower. All still fail100% recovery; span32 remains perfect on this validation.
One inherited seed/reused validation/checkpoint-selection limits. No new worker
failures; old native and rejected-probe history preserved. Workers/locks gone.
No additional training authorized. See docs/compression64_continuation.md and
standalone artifacts/runs/compression64-continuation-8000-v1/completion_verification.json.

# Compression64 continuation authorized (2026-10-07)

Latest user authorized 8000 additional updates per arm from saved depth_route,
with parallel improvements. Five arms: continue, coverage, hard_tokens,
coverage_hard, margin; max5workers. All exact same 7327232 encoder/10456576 total
parameters, context256/span64/width256. Same parent model AND optimizer; learning
rate3e-5. No old control retraining. Only sampling/loss varies as documented.
Sources frozen under dump/compression64-continuation. Preflight passed; all five
workers launched. Status D:/Git/latent-text-compressor/artifacts/runs/compression64-continuation-8000-v1/status.json.
Complete initial saved-parent validation replay is required before each first update.
All five passed independent decoded checks, replayed557/1000 and NLL0.166150,
and advanced beyond100 updates. No launch failures; GPU total about3044MiB.
Full validation every500; retain/replay both final and best checkpoints. No automatic
worker retry. Two preflights discarded ten smoke steps total: first hinge-margin
objective exceeded memory; additive-logit-margin replacement passed419.85/462MiB.
First probe failure/sources are preserved. No native failure was observed in probes.
Every worker must verify full decoded identities; no automatic new native/data
failure retries. Preserve previous evidence/weights. See docs/compression64_continuation.md.
This supersedes the prior no-extra-training status only for this bounded batch.

# Fixed-size compression64 COMPLETE (2026-10-07)

All five x3000 seed17 updates completed. Source/parent/checkpoint/export/record
hashes, target schedules, complete update sequences, parameter caps and final
full validation replays verified. Best depth_route557/1000 exact,99.2769% tokens,
NLL0.166150; compact533, iterative503, depth_nonlinear493, nonlinear472 exact.
None meets perfect recovery; selected32 remains1000/1000. Winner encoder7327232,
full10456576;425.54/486MiB allocated/reserved, higher memory than saved32.
One seed and selected validation, not universal compression. No new worker failures;
prior native failures preserved. No extra training authorized. See docs/compression64_fixed_budget.md
and standalone artifacts/runs/compression64-fixed-budget-v1/completion_verification.json.

# Compression64 five-model batch LAUNCHED (2026-10-07)

Latest explicit user request authorized retry. All five correctness/checkpoint/parent/padding/GPU/cap checks passed after correcting a documented FP32 roundoff tolerance; no model changes. Five workers launched, 3000 updates each, seed17, fixed span64. Each worker verifies complete decoded train/valid identities before updates. Sources frozen, no control retraining or budget growth. Status D:/Git/latent-text-compressor/artifacts/runs/compression64-fixed-budget-v1/status.json. Earlier native failures remain preserved/unexplained; no automatic new native/data retries. See docs/compression64_fixed_budget.md for measured counts, probe memory and completion requirements.

# New fixed-size compression64 batch BLOCKED (2026-10-07)

User authorized new constant-parameter compression variations. Five declared arms x3000 updates, same span64, seed17, max5workers; no encoder/full size growth beyond7342336/10488832. Implementation and guarded controller prepared under dump/compression64-study. First preflight crashed natively in python312.dll (0xc0000005), with thread limits enabled. No candidate training launched, zero research updates. No automatic retry or bypass; preflight passed:false. Existing model/package sources and weights unchanged. See docs/compression64_fixed_budget.md for hypotheses, sources and budget.

# Span screen COMPLETE (2026-10-07)

All six spans48/64/96/128/192/256 finished2000 updates each. Final replays,
checkpoint/encoder/source hashes, matched target schedule and records verified.
Exact paragraphs/1000:972,139,131,38,6,12. Span32 remains highest tested perfect
recovery (1000/1000). Span48 token accuracy99.9697%, about29.8% fewer vectors.
This is a bounded training result, not a theoretical capacity limit. No extra
training authorized. No new worker failure; previous failures preserved.
See docs/rope_span_limit.md and standalone artifacts/runs/rope-span-limit-v2/completion.json.

# Span retry with tested thread limits (2026-10-07)

User requested diagnosis/fix. Five full tokenizer checks and five concurrent span256 CUDA smoke tests passed with native/torch threads limited to1 and tokenizer parallelism disabled. V2 same six x2000 budget launched, five workers, independent identities retained. Status D:/Git/latent-text-compressor/artifacts/runs/rope-span-limit-v2/status.json. This is a tested mitigation, not a proven root cause. No blind retries on new native/data failures. Preserve old v1 failures; no models/data/package sources changed. See docs/rope_span_limit.md.

# Span screen blocked before training (2026-10-07)

Span48 decoded-identity failure; span64 tokenizers.pyd native access violation (PID23632). Zero updates, other four spans not launched, no workers remain. Files match original data hashes. No automatic retry or bypass; source/failure evidence preserved. See docs/rope_span_limit.md. Higher-span limit not measured.

# Authorized span-limit screen (2026-10-07)

User requested larger tokens/vector and a measured limit. Six independent span32 transfers: 48,64,96,128,192,256 x2000 updates, seed17, two workers. No baseline retraining. See docs/rope_span_limit.md. This supersedes previous no-new-training status for this bounded screen only. Preserve span32 weights and all new results.

# RoPE compressor complete (2026-10-07)

All 7000 seed17 updates finished. Full final validation replay: 1000/1000 exact,
NLL0.00227359, token accuracy100%. Encoder7342336 / full10488832 parameters;
379.44/470MiB peak allocated/reserved. Checkpoint/encoder/source hashes and record
verified. Completion receipt dump/rope-compressor-launch/completion.json.
No new training authorized. One seed, previously used validation; no matched
RoPE ablation or superiority claim. Sequence shortening is not byte compression.
Old compressor weights were deleted as requested; Step 1 weights remain.

# Authorized RoPE compressor run (2026-10-07)

Latest user explicitly authorized new encoder training and deleting old compressor
weights. This supersedes prior no-training and compressor-weight retention rules.
One fresh seed17 Branch Sigmoid + RoPE run: 7,000 total updates in five stages,
using the existing 50k paragraph data; no extra experiments. Sources frozen after
launch; do not retry native failures automatically. See docs/branch_rope_training.md.
Step 1 Branch Sigmoid weights, datasets, records and source archives are retained.

# Branch compressor migration (2026-10-07)

User explicitly replaced Step 2 CurveFFN with Branch Sigmoid in encoder AND decoder.
This supersedes historical internal-curve retention and compatible encoder loading.
Both selected packages remain active. Step 1 weights and outputs remain unchanged.
Step 2 is now an UNTRAINED architecture revision: old CurveFFN weights are preserved
but deliberately rejected, and historical reconstruction scores do not apply.
No training is authorized by this code migration. See docs/branch_compressor_migration.md.

Active packages: src/models/branch_sigmoid and src/models/position_compressor.
Keep only their configs plus branch_data_identity.json in src/config. No old
experiment registry, sweeps, alternative LMs, Step 3 integrations or custom kernels.
No training is authorized by cleanup; future training needs an explicit budget.
Use ordinary PyTorch and uv run --no-sync. Do not replace/sync the CUDA environment.
Do not blindly retry native crashes. Earlier DLL/runtime failures remain unexplained.

Preserve selected checkpoints, datasets, historical records/audits and source
snapshots. Do not restore retired source into the active tree to repair old paths.
Original sources/recipes are required for historical optimizer resumption. The
current selected Branch loader verifies the exact winner hash for inference.
Step 2 exports require current architecture/FFN source hashes and Branch v2 format.
Do not modify docs/research_goal.md. Research targets remain unachieved; cleanup
is not a new quality result. Step 2 reconstruction is not next-token LM evidence.

Branch Sigmoid: web3.820069/chat2.411506,8,654,208 total/4,718,592 FFN parameters,
10,000 updates seed461,454.54/480MiB allocated/reserved. All latest branch-next
variants lose to it on both corpora. One-seed/selection limitations still apply.
Historical CurveFFN span32 encoder:4,798,720 parameters; full model7,292,928; bounded
reconstruction success, no universal losslessness or downstream reasoning claim.

No training workers or monitoring automations remain. Archived active-tree snapshot
and deletion receipt: dump/selected-cleanup-20261007/. See README.md and
 docs/selected_models_cleanup.md. Historical evidence remains under dump/records;
keep records ignored/untracked, never force-add. Historical leaderboards are frozen
references; do not silently recompute old runs using migrated sources. Export any
new authorized runs and document their distinct source provenance/results.

Latest user concurrency preference: up to five models concurrently for the same six-span budget. Prepared five-worker v2 runner, NOT launched; do not interpret this as resolution of native/data failures.
