# Growing FOUR-readout model READY; user-run full training only (2026-10-16)

Latest user rejects continuing the one-position query-conditioned experiment,
asks for simpler FOUR learned readouts/four attention positions per vector and
a full training command. Implemented separate src/models/growing_token_memory_readout4
to preserve existing resume guards. Query-reader run already PAUSED6834 before
work; original growing stillpaused35472. No trainer interrupted/resumed/launched.
New output dump/growing-token-readout4-v1 remains absent. No cleanup/subagents.

Read pasted attachment7c2b7100.../Pasted text.txt. Queryreader vsold growing at
same6000dev: CE5.289425543 vs5.275466852, accuracy16.012903 vs16.138536%.
No early gain; do not compare this partialrun tofull2epoch finalquality.

New readouts: RMSNorm256 -> learned256x2048 ->SiLU ->4x512+sharedmemorytype.
Applies tocompletedcodesANDactive0..63prefix. Empty uses4BOS controlstates plus
originalparameterfree lengthmetadata. Standardcausalattention,4thprefixstate
predictsnexttoken. No codecD, rawtokeninputs, query-conditionedvaluereader,
distillation ornewobjectives. Fourreadoutsarelearnedfeatures,notfixed16tokenchunks.
Sharedprojectionadds393216weights; alleightFFNs1534->1486 freeexactly393216.
Total23209216; unchangednonFFNcorefresh47identical, resized/newweightsfresh47.

One64tokencode=>4positions:16xpositionreductionvsraw,4xmemoryKVvsoldoneposition.
39codes=>156slots=>5111808KVbytesacross8layers. Notunchangedcache/compute!
Productionqueriescompute4Qx(4M+4) pluscausalmemory attention, no(4Q)^2querymask.
Independentprefixgroupscannotseeoneanother/futurecodes. Cached/fullpathsagree;
append_memory updatesonlynew4slots/code. generation.generate predictsordinary
tokenEOS2, actualprefixreencode/replace,commit64+clear,appendsKVwithoutrecomputingold.

Newtrainingcopiessamerecipe/data/cache/labels/order/effective32/FP32/AdamW.01/clip1/
warm2003e-4cos3e-5,2epochs48762updates361357104labels,fresh47. ExistingI:cache
compatible; codevectorsonlyprecomputed, learnedreadoutsrecomputedperupdate.
Format growing-token-memory-readout4-v1 rejectsbotholdgrowingformats; mainsource/
data/encoder/checkpoint/workerlockguardsremain. FullcommandinnewmodelREADME:
.venv-cuda64 Python,newtraining.py --name growing-token-readout4-v1 --data
dump/data/unified-chat-v4 --cache-dir I:\growing-token-cache-v4 --epochs2
--batch-size32 --microbatch4 --encoder-batch256 (proper spaced flags in README),
NOstop-after. Fullrun leftforuser; noquality/factrecoverygainclaimed.

24CPUchecks passed(new11+existing13): exactbudget/sharedinit,independentdense
logits/allparam/inputgradients,fourdistinctreadoutsallgradnonzero,empty/future/
padding/queryisolation,full/cachedgrads,4xcache,appendoldentriesunchanged,live
actualtokenrollover/EOS, tinyactualtwoepochAdam/model/RNG/cursorresume. Initial
23pass1fail was3.47e-18 separateprojectionfloatingdifference; correctedonlytest
tolerance andthatcheckpassed. SyntheticCPUupdatesonly,zero corpusoptimizersteps.

GPUpreflight65293 finished0 underworkerlock. Fullsizeoriginalfirst32seed47samples/
7035labels andrealheavycache,34prefixshardSHA+completedSHA/identitiesverified
read-only.16freshEprefixrecodesmax1.1921e-5. Independentdense logits1.1474e-6,
allparametergrads1.6391e-7,cached9.0897e-7; all4readoutgradspositive. Micro4heavyreal
andmax39codes/512queriespassed with2Adam+1scratchbuffers. Peak4.19458GiBallocated/
5.49805reserved.3alternatingidleGPUrounds F/Bmedianoriginal.44572s vsfour.87791s
(~1.97xtime), excludesdisk/E/Adam/eval/save. Notfulltrainingthroughput orquality.
No models/source/cache mutations bychecks. No GPUworkerfromsetupremains.

Evidence dump/growing-readout4-preflight-1792138331838445500.json,
growing-readout4-before-20261016.json, growing-readout4-setup-verification-20261016.json,
growing-readout4-implementation-sources-20261016.zip. All29protectedfilesincl5
existingcheckpoints unchanged; all5runstatusesunchanged. Archivebytesverified,
finalsourcesmatchGPUreport. NewmodelREADME hasfullandresumecommands; rootREADME
andoldgrowingREADME updatedonlyforstatus. NoexistingmodelPythonfilesedited.

# Raw64 cross-attention COMPLETE/evaluated; full growing-reader command READY (2026-10-16)

Latest user reports completed raw64 cross-attention, requests comparison and FULL
two-epoch command for the query reader (remove stop-after4000). No training launch
authorized by this request. New growing-token-query-reader-v1 directory absent.
No active corpus trainer found; original growing remains paused35472. Do not
resume old growing with new architecture or restart rejected full unpacking.

Read-only tools/compare_token_memory_crossattn.py finished25779 exit0. All three
original512/plain64/crossattn64 final=best48762, Adam48762,23209216params. Every
sample/label/LR/cursor across48762updates matches;361357104label exposures each.
Dev CE1.804710358/2.214283787/2.247859514; accuracy61.01325/53.50459/53.03218%.
Fresh full7646test/1757421labels CE1.806953619/2.216947987/2.249793294; accuracy
60.99904/53.50875/53.01268%. Hybrid worse CE on17/17sources vsoriginal,16/17vsplain
bothsplits; memory-source CEbettervsplain but accuracyworse. Paired-document5k
bootstrap hybrid-minusplain testaccuracy[-.5410,-.4510]pp, one seed/priorusedtest.

Same312cases: all3notebookexact0/103, changed0/32, pairedcorrect0/32. Missing
13/25,17/25,18/25; falseexactmissing49/103,65/103,78/103. Hybrid remaining25
answerable responses manuallycheckedwrong;0fullvaluesubstrings. Colors recent
8/8,0/8,0/8;compressed0/8,1/8,0/8;blocked1/8,1/8,2/8 (blockedhits not retrieval).
Hybridchangedoutputs5/32 withoutcorrectpairs; notebookEOS128/128correctmemory,
8/128blocked. Bothcrossgatesnonzero tanhnorm1.0083/1.0772. No usefulfactgain.

Newevaluator usespublichybridforward/packed; independent unfoldedraw+explicitcross
reference maxlogit2.2888e-5; blockedcontentinvariance and4serialreplays/modelpass.
All312oldoutputs andalloldtestmetrics exacthistoricalreplay. Allcheckpoint/source/
encoderhashes unchanged;0optimizerupdates. Separatelyrederivedsavedsegment/source
metrics andnotebook exact/EOSscores. Report/artifacts:
dump/token-memory-crossattn-comparison-20261016 (report,verification,abstention,
cases,predictions,stats,sources.zip). docs/token_memory_64_crossattn_results.md.
HybridcheckpointSHA9f96ae069dc5d7afd71d39f3024e57888e82b960a3f2230aaa35a07a6f94a1e6.

IdleGPU growingbenchmark53215 finished0 underworkerlock, micro4effective32,
first32seed47samples7035labels,3alternatingrounds,FP32TF32off. OriginalF/Bmedian
.50567s vsnew .72671s (~44%moretime); disabledreader .62716s. Cachedquery18.36ms
vs28.62ms. Notfulltraining/generationrate: excludesdisk,Adam,eval/save,liveencoder.
Allselectedcachehashes/weights/sourcesunchanged; noencoder/optimizerupdates.
Report dump/growing-query-reader-idle-micro4-20261016.json. BothKV1277952bytes/39codes.
Separate tools/check_growing_query_reader_capacity.py micro4 max39codes/512queries
with2Adam+1scratchbufferspassed:1.340GiBallocated/1.547reserved,allgradsfinite,
unchangedweights/sources,0updates. dump/growing-query-reader-micro4-capacity-20261016.json.
Fullcommand in modelREADME: fresh name growing-token-query-reader-v1, existing
I:\growing-token-cache-v4, epochs2 batch32 micro4 encoder256, NOstop-after. Same
48762stepLRschedule/fresh47/FP32, notpromiseofsamequality. No modelPythonedits,
checkpoint/cachemutations, deletions,subagents ornewtrainer fromthiscomparison.

# Query-conditioned growing reader IMPLEMENTED, benchmarked; NOT TRAINING (2026-10-16)

Latest user authorizes the first query-conditioned value reader inside
growing_token_memory at2 layers, same budget/one position/no extra KV, benchmark
before training and do not touch active run. Implemented query_reader.py and wired
full_model.py/training.py. Historical transformer.py remains byte-identical.
Do not resume old growing35472 with this architecture or restart rejected unpacker.
No new corpus training launched; active hybrid PID12572 continued beyond45900.
All26 active-run guarded sources unchanged; no process/settings/cache changes.

Readers layers4/8: per-head v_i+B*tanh(v_i[:16]+A*q), q unrotated, ordinary rotary
attention weights. Existing V channels only; includes active-prefix self-value.
Pool16-wide features before B; one code/position, original per-layer K/V shapes.
Reader32768params funded by FFN1534->1518 at exactly those2layers; total23209216.
Other shared core tensors fresh47 identical; new/resized tensors fresh47.
No new gates/objectives/decoder/raw-input IDs. Genuine generated-prefix replace/
commit64 unchanged. Frozen precomputed cache compatible, no cache source changes.

Final trainer uses batched memory/query QKV+FFN; reader layers compactQx(M+1), six
other layers original fused masked attention. Initial all-layer separated path
was slower and replaced; retain diagnostic reports. Generation uses original
memory_cache and new query reader, no extra persistent tensors. Temporary training
activations/compute increase. Newformat growing-token-memory-query-reader-v1
rejects old resume. Same original2epochs48762/effective32/labels/order/FP32/Adam/LR.
Model README has fresh4k stop command retaining full2epoch schedule, micro1.

22CPUtests pass incl independent dense reference/allparams+inputgrads, cached/full
grads, code-specific query conditioning, future/pad/empty isolation, actual-model
live64rollover, exactbudget/sharedinit, tiny actual serializedAdam/model/RNG/cursor
resume and unchangedcache guards. Full-width realdev GPU check16queries: logits
5.96e-7, gradients6.71e-8, cached8.57e-7; allreadergradnormspositive, modelunchanged.
First GPU reference-helper attempt had CPU/CUDA allocation mismatch; fixed only
testhelper, rerunpassed and9readerCPUtests rerunpassed. Zero corpusAdamupdates.

BoundedsharedGPUbench (12%allocator) matched first32samples7035labels, micro1,
3alternatingrounds: originalF/Bmedian3.4667s vsnew3.4075s; cachedquery37.43ms vs46.41.
Largecontentionvariance, NOTspeedwin; disabledreaderarm4.3654s shows timingnoise.
F/B excludesdisk/encoder/Adam/validation/save; query excludesliveencoder. Newheavy
peak746.86MiBallocated/864MiBreserved including2Adam+1scratch-sizedbuffers.
39codesKV1277952bytesexactboth. Heavyreal andsynthetic39codes/512queries pass.
Noquality/factrecoverygainclaimed. Benchmark worker finished; no newtrainer.

Evidence dump/growing-query-reader-benchmark-optimized-20261016.json,
growing-query-reader-gpu-parity-20261016.json, growing-query-reader-before-20261016.json.
Report docs/growing_query_reader_setup.md; originalsourcearchive
dump/growing-query-reader-original-sources-20261016.zip. Current sources/receipt
dump/growing-query-reader-implementation-sources-20261016.zip and
dump/growing-query-reader-setup-verification-20261016.json. No deletion/subagents.

# Full decoder-expansion pilot REJECTED AND REMOVED (2026-10-16)

Latest user rejects decompressing every code back to64 positions before the LLM;
this defeats the attention-length compression they wanted. User explicitly asked
for the new model to be deleted, and FIRST an explanation of how codec decoder
and original growing_token_memory handle vectors, before updating the original.
Do not resurrect/restart the rejected pilot or repeat full per-token expansion.
No compact architecture change has been made yet; respect the requested sequence.

Verified pilot already exited after2 completed updates: during update3 it exceeded
its OWN12% CUDA allocator cap. Last durable checkpoint wasstep1. No live unpacked
trainer/launcher remained. Original hybrid PID12572 stilltraining; no interruption.
Original growing remains paused35472 and has not been resumed or edited.

Removed only rejected experiment paths: src/models/growing_token_unpacked, its
specific test and three helper tools, new pilot output/checkpoints, launch/logs,
sourcearchive, setup/preflight/benchmark files and own lock.45files/502233916bytes.
Native PowerShell exact resolved targets checked within D:\Git\complex_fnn, no
reparse points;62 pre-existing source hashes still match. Original models/codec/
checkpoints/data/I:cache untouched. Audit dump/rejected-unpacking-removal-20261016.json
retains removal manifest, prior source hashes, failure reason and2training rows.
No quality result; earlier passing preflight did not cover the failing batch.

Confirmed code distinction: actual residual.Unpack maps z256 ->64x125 via learned
span_map, then shared125->256 ->64x256 position features. BaseModel.decode applies
ordinary noncausal reconstruction attention/FFN, decoder_norm and vocabulary head.
Depth attention-residual routing is in ENCODER only, not reconstruction decoder.
Growing reader maps RMSNorm(z256) through one learned256->512 projection + type,
plus length metadata for active prefix; standard causal Q/K/V attention, one slot
per completed code and one active-code query per prediction. No pretrained
position readouts; reader is trained for next-token CE, codec for reconstruction.
The added unpacked model did not argmax to token IDs, but did restore token-length
feature sequences, so the user's objection is valid. Future compact readers may
use learned/query-conditioned readouts; do not claim a proven fix or unchanged
compute from merely copying the full decoder. Explain before implementing.

# Raw64 gated memory cross-attention READY, user-run only (2026-10-16)

User requested a separate hybrid, then selected raw64 first (not256), same exact
parameter budget/fresh initialization/two epochs; do not touch the current run.
New src/models/token_memory_64_crossattn has folded causal raw-only self-attention,
pointwise nonlinear memory reader and separate gated cross-attention at layers4/8.
Two512-channel tanh gates startzero. FFNs1534->1214(first4)/1213(last4) freeexactly
2625536params; total23209216. All tensors used; same-shaped core fresh47 matches
baseline, resized/new tensors fresh. Not a pure attention-only ablation. No depth
attention residuals. Every completed64 block appended and raw emptied as plain64.
Generation caches independent memory reader/KV and appends only new code states.

Same original780186segments/order/effective32/labels/FP32/AdamW/warm200schedule,
48762updates/361357104labels across twoepochs. Micro16 starting setting; CUDA75%
allocator. Reuses dump/token-memory-64-cache-v4, not I: growing-prefix cache.
Format token-memory-64-crossattn-v1 rejects old resumes. Full training not launched;
no output directory encoder-memory-64-crossattn-unified-v4 created by setup.
Command in model README. User runs it only after current worker is done/paused;
shared worker lock retained. No existing model sources/weights/cache modified.

14CPUtests passed: full parameter/shared-core identity, fullsize logits, learned
gates packed/reference allparam+input gradients, future/window/pad/empty memory,
gate opening+reader gradients, cached append/live actual-token rollover, actual
tiny synthetic two-epoch Adam/model/RNG/cursor resume plus original7data/cachetests.
No corpus optimizer updates. CPU real-data fullsize preflight92927 exited0:
dump/token-memory-64-crossattn-preflight-1792120763764542600.json; logits1.34e-6,
allparametergradients7.08e-8, exact initial weights restored. CUDA devices hidden
for all setup checks. GPU capacity/throughput/quality unmeasured; no fact gainclaim.
Receipt dump/token-memory-64-crossattn-setup-verification-20261016.json, sourcezip
dump/token-memory-64-crossattn-implementation-sources-20261016.zip. No subagents,
deletion, production restart or worker from setup remains. Evaluate using this
model's forward/packed or generation.generate, never unbound Original.forward.

# Growing prefix cache upload optimized, user-run only (2026-10-15)

User stopped cache preparation (~616448/780186 segments), asked to speed up GPU.
Confirmed no growing worker and no dump/growing-token-unified-v4/last.pt.
Larger encoder batches512/1024/2048 slower than256 (3016 prefixes/sec original).
Changed cache.encode to CPU-pack token/mask arrays then two bulk GPU transfers.
4096 real prefixes,3 repeats:256 now4689/sec (1.55x encoding, not end-to-end),
bit-identical outputs at256; larger batches within5e-5/2e-5 tolerances. FP32 and
frozen encoder/reader/data/targets/model size unchanged. Keep encoder-batch256.
Benchmarks dump/growing-prefix-{batch,upload}-benchmark-20261015.json.

Existing I: train cache source matches audited old cache.py SHA
748957b4dee2df212bbdd0968457e97bd07bae52547ba192067c195e1c0265c7.
CLI --upgrade-prefix-cache explicitly permits ONLY that source transition with
other identities exact. Hash-check every existing shard BEFORE identity migration;
record upload-upgrade-*.json; retain completed vectors; redo incomplete shard.
Original cache identity is still unchanged on I:; no external cache writes made
by this task. User reruns fresh --name growing-token-unified-v4 command with
--cache-dir I:\growing-token-cache-v4 --epochs 2 --batch-size 32 --microbatch 4
--encoder-batch 256 --upgrade-prefix-cache. Not --resume until checkpoint exists.

13 CPU tests pass incl explicit migration opt-in, unrelated identity rejection,
corrupt shard rejection before migration, no rewrite of completed shards, resume.
Fullsize zero-update GPU preflight3747 exited0:
dump/growing-token-full-preflight-1792055498223887200.json, final sources verified.
Receipt dump/growing-token-upload-optimization-verification-20261015.json;
source archive dump/growing-token-upload-optimization-sources-20261015.zip.
Original setup archive retained. No full cache/training launch, deletions,
subagents or model/encoder weight edits; no worker from these checks remains.

# Full-v4 growing TOKEN-prefix trainer READY, user-run only (2026-10-14)

User confirmed two full epochs of the actual per-token prefix re-encoding idea,
NOT direct growing_vector latent prediction or raw64/attention-residual rollover.
User chose I:\growing-token-cache-v4. New growing_token_memory/{data,cache,
full_model,training,preflight}.py; original pilot transformer.py byte-identical
with pilot sources.zip. Full reader forward uses growing codes, no raw IDs.
Fresh baseline seed47 exactly 23,209,216 trainable params; original E frozen.
780186 original train segments; same labels/shuffle/effective32/warm200 LR3e-4
cos3e-5/AdamW.01/clip1. Two epochs48762 updates/361357104 assistant labels.
Micro4; no attention residuals. This is a fresh full-corpus recipe, not pilot
fine-tuning and no new quality result. Original models/checkpoints untouched.

Full supervised nonempty prefix + completed-code FP32 cache189123028992 bytes
(176.13GiB), train/dev. Only strictly past current64-prefix tokens, empty BOS;
completed codes causal, independent query snapshots. 2048-segment disk shards,
source/data/encoder/layout identities + SHA hashes; incomplete shard recomputed,
completed shards reused. Full cache prepared BEFORE optimizer and reused epochs.
Checkpoint format growing-token-memory-full-v1; Adam/model/RNG/cursor resume.
12 CPU tests passed incl initial identity, exact labels/causal prefix boundaries,
interrupted cache recovery/hash corruption, actual tiny two-epoch training resume.
Zero corpus optimizer updates. Final fullsize real-encoder GPU preflight2987 exit0:
dump/growing-token-full-preflight-1791986066240847100.json. All numerical/cache/
gradient/capacity checks passed; initial weights unchanged. No GPU worker from
this setup remains. Earlier lock refusals respected independent attnres comparison;
no interruption or subagents/deletion. Production position-memory remains paused.

Command from repo: .\.venv-cuda64\Scripts\python.exe
.\src\models\growing_token_memory\training.py --name growing-token-unified-v4
--data .\dump\data\unified-chat-v4 --cache-dir I:\growing-token-cache-v4
--epochs 2 --batch-size 32 --microbatch 4. First prepares cache then trains.
Neither I: cache nor full-run output created by this setup. Full training left
for user. README in model folder has preparation/resume commands. Verification
receipt dump/growing-token-full-setup-verification-20261014.json and sourcearchive
dump/growing-token-full-implementation-sources.zip retained. Evaluate with growing
forward/cached_growing, never unbound original raw-token forward.

# Attention residual64 EVALUATED at user-stopped37045; no further training (2026-10-14)

Latest user asks3model comparison, then explicitly says "no i dont wnat to trian
further" after localcheckpoint audit. Respect intentionalend at37045/48762;
doNOTresumeattnres training. Status remains paused; last37045,best37000. No
fulltwoepoch completionclaim. Olderoriginal512/plain64 arefinal/best48762.
New tool tools/compare_token_memory_attnres.py finished38788 exit0,0optimizer.
No worker from evaluation remains. Allprefix37045batches/samples/targets/lr/
cursor identical; new274371917labels vsold361357104. Parameters23217408 vs23209216.
Sources/checkpointhashes/Adamcounters verified. NewcheckpointSHA
70b666f9e601fc33235d16f930a328a24e79062204acd61e20e60300cb7d8a1d.
Exactmatched37000dev(original/plain/attnres):CE1.84651594/2.25939087/2.23951270;
acc60.31226/52.71971/52.96447%. Small+.24476ppvsplain,14/17CEsourcesbetter.
Latestfulltest7646segments1757421labels:CE1.80695362/2.21694799/2.24058791;
acc60.99904/53.50875/53.04409%. Unequaltrainingbudget; no37koldweights retained.
All3notebookexact0/103;changed0/32;pairedoriginal+changed0/32. Missing13/25,
17/25,22/25, butfalseexactmissingtemplate49/103,65/103,87/103! Newother16manually
reviewedwrong,0fullvaluesubstring. Plainsolevaluehitwrongperson. Recentcolors
8/8,0/8,0/8;oldcompressed0/8,1/8,0/8;blocked1/8,1/8,0/8. NewblockedEOS1/128vs
128/128correctnotebookmemory. Changedoutputs5/32,13/32,10/32withoutcorrectpairs.
No reliablefactualreadinggain; original512bestretainedlanguagefoundation.
NEW evaluatorusespublicModel.forward/packed; blocked64preservesorigvisiblecounts
whilemaskingfactkeys. IndependentRouted.reference retainsallqueries. Checked
logitsidentical,blockedcontentinvariant,4serialreplays/model; all312oldoutputs
andtestmetricsexactreplay. Allquerynormsnonzero2.17..5.92. Sourcezip/checkpoints/
encoder unchanged; separateartifactverification passes. Transientworkerlock
refusal resolved on retry after no active trainer found; no lockbypass.
Evidence dump/token-memory-attnres-comparison-20261014-step37045: report.json,
abstention_analysis.json,verification.json,allcases/outputs/stats/sources.zip.
Report docs/token_memory_64_attnres_results.md; README/Step3/modelREADMEupdated.
No deletion/subagents/architectureedits/productionposition-memoryrestart.

# Historical attention residual variant setup (2026-10-14)

Latest user asks to add codec-style attention residuals to64-token model, then
they will retrain. Implemented separate src/models/token_memory_64_attnres;
completed plain64/original512 sources and weights remain unchanged/reproducible.
Codec has learned depth mixing in ENCODER only; reconstruction decoder isplain.
NewModel reuses exact ResidualModel.mix_depth before attention, beforeFFN and
at finalnorm, retaining all preceding sublayer outputs perposition. Samecausal
64windowmask. Adds16x512zeroqueries=8192params,total23217408. Corefresh47 tensors
identical; zeroquery logits agree plain within2.21e-6FP32. No FFN resizing.
Onlydepthmixtures activation-checkpointed forVRAM; sameformula and CPUgrads.
Separateformat token-memory-64-attnres-v1 rejectsold64resume. Originalcache/data/
labels/order/effective32/48762updates/twoepochs unchanged. Newtrainer uses75%
CUDAallocator matchingpreflight, micro16. No fulltraining launched/outputcreated.
Suppliedcommand newname encoder-memory-64-attnres-unified-v4, oldcache path
dump/token-memory-64-cache-v4. Freshinitialization; notfinetune oldcheckpoint.

13CPUtests pass (6new+7old): coreidentity,zeroparity,learnedquerypacked/reference
allparam+inputgrads,future/paddingisolation,checkpointgradidentity,formatguards,
tinyCPUserializedAdam/model/RNGresume and learnedqueriesupdate. Tiny synthetic
CPUoptimizerchecks only;zeroCORPUSupdates. FullGPUpreflight final49871finished0:
dump/token-memory-64-attnres-preflight-1791941803571167900.json. Learnedlogits
max4.29e-6,allgrads9.50e-8,everyquerygradpositive; initialweightsrestoredexact.
Heavymicro16 inclAdam-sizedbuffers5.381GiBallocated8.463reserved. Warmedeffective
forward/backward.743/.752s; excludesencoder/Adam/eval/save; slowerthanold~.294s,
notend-to-end throughput. No claim factrecoveryimproves beforetraining/evaluation.
Newtrainer --prepare-only6597finished0,reusesverifiedfulltrain/devcache hashes,
no newcache/reencoding. Finalsetupverification40021finished0,13passed, original
protectedfilesunchanged. Receipt dump/token-memory-64-attnres-setup-verification-20261014.json;
sourcearchive dump/token-memory-64-attnres-implementation-sources.zip.
README/Step3/newmodelREADME describe commands. No subagents/deletion/production
position-memoryrestart (stillpaused1001). No workerfromthesechecks remains.
IMPORTANT futureevaluation: useNEWModel.forward/packed or generation.generate;
historical tools/compare_token_memory_rollover.py usesUNBOUNDOriginal.forward,
which BYPASSESattentionresiduals. It cannot evaluate thisvariant unchanged.

# 64-token raw rollover COMPLETE, original wins measured comparison (2026-10-14)

User completed dump/encoder-memory-64-unified-v4, asks compare with original.
Read-only tools/compare_token_memory_rollover.py finished86674 exit0; no worker
remains from this comparison. No new training/cleanup/subagents. Original512
and new64 both final/best48762, allAdam48762. Everyloggedbatch/sample/target/lr/
cursor matches:361357104labels. Both23209216parameters, fresh47, micro32vs16.
Fulltest7646segments/1757421labels original CE1.806953619/acc60.99904%; new
CE2.216947987/acc53.50875%. Dev1.804710358/61.01325 vs2.214283787/53.50459.
All17sources worse bothsplits; pairedtestaccCI new-old[-7.6528,-7.3319]pp,
oneseedonly. Testpreviouslyusedresearch, notblindnewbenchmark.
Same128notebookcases: answerableexact0/103both, changedfact0/32both, missing
13/25old17/25new. NewonesubstringhitWRONGPERSON(Vera->Bilal), notcorrect.
Factslotblockedexact1/103old0/103new, missing10/25old0/25new. Blockingpreserves
positions/fullquestion/latermemory; masksfact-containingcodesincludingformatting,
NOToldall-memoryblockedprotocol. NewblockedEOS11/128 vs127/128correctmemory.
8colorfacts recent8/8old0/8new, oldcompressed0/8vs1/8, blocked1/8both. Recent
prompts59tokens initiallyraw; newcrosses64after5outputtokensanddrifts.
No reliablefactualreadinggain. Originalretainedaslanguagefoundation.
Evidence dump/token-memory-rollover-comparison-20261014:report,stats,cases,outputs,
sources.zip,verification.json. Publicforwardparity/blockedcontentinvariance/
serialreplay/unchangedmodelencodercheckpointsource pass; separatestats/source
artifactcheckpasses;128/128originalcorrect-memoryanswersreproducehistorical.
NewSHA047d5bc295e837663fd294ffa968c6286198211cd73c25141063dc0ff88a1cb0;
originalSHAe17b9a8469b5a8c996cc64ff0677eb1c8f52410af52c1c179fd317263b4d9e2d.
Report docs/token_memory_64_results.md; README/Step3/modelREADME updated.
Broadergoalunresolved. Productionposition-memory remains paused1001; no restart.

# Historical 64-token setup: READY, cache VERIFIED (2026-10-13)

User requests a separate src model, fast precomputed training, two epochs matching
the baseline, and a command (NOT a training launch). User explicitly chose FRESH
baseline initialization. New src/models/token_memory_64 is separate from the
growing-prefix experiment: retain 0..63 raw tokens, predict token64, encode one
completed64 block, append, clear raw buffer; then BOS-only plus memory.
Exact baseline parameter names/shapes/seed47 tensors,23,209,216trainable; original
v4 data/masks/order/effective32 original512 segments/48762updates/twoepochs and
361,357,104assistantlabels. Packed independent64 windows share causal memory;
future/otherwindow raw access blocked. New max39 memoryvectors includes seven
earlier-in-current-segment blocks; no added history tokens. Micro16 vsbaseline32.
CE only/FP32/original warm200 3e-4cos3e-5/AdamW.01/clip1, frozenoriginalE.

SevenCPUtests pass incl fullsizeinitialidentity, allgradpacked/separate, future
invariance, exacttargets, diskcachehash, 64rollover, serializedAdam/RNGresume.
Zero-update fullsizeGPUpreflight79734finished0: logitsmax3.34e-6, grad5.96e-8,
cacheE1.91e-5. Micro16 heaviest3.838GiBallocated/3.988reserved withmomentbuffers.
Cached firstbatch forward/backward~.294s new vs.294-.320s old; excludesencoding,
Adamstep/eval/save, notquality/speedwin. Evidence dump/token-memory-64-preflight-1791911684777184500.json.

Old external CUDAvenv disappeared during work; local .venv isCPU-only. Separate
.venv-cuda64 restoredtorch2.14.0+cu132 +deps; preserve it for supplied command.
PersistentFP32cache COMPLETE, preparation36226finished0:
dump/token-memory-64-cache-v4. Train4,937,894codes/dev49,168, total4.756GiB.
Independent verifier55696finished0: all780186/7831segments and180678552/1859388
labels audited; file/layout hashes pass; fresh128randomrecodes eachsplit max
1.955e-5/1.860e-5. Receipt dump/token-memory-64-cache-verification-1791912914940285600.json.
Cache trainSHA8ec4f1237488aafaa7bdf36b48cab16d9873c98108fddc084ae927a2eb7d0191,
devSHA7f211094630e0dacf15958b519941f147f676698c30cc35c9701013b69d40b99.
Sourcearchive dump/token-memory-64-implementation-sources.zip and environment
receipt retained. No worker from these checks remains; zero researchAdamupdates.
Diskspace now ample due independent cleanup, none performed by this task.
README/Step3/modelREADME updated, command uses .venv-cuda64 and persistentcache.
Fulltwoepoch training NOT launched; original models/weights/data not changed here.
Keep cache source/data identity guards; RAM fallback exists but default supplied
command reuses diskcache. No generation-quality improvement claimed or tested.

# Growing token-output prefix pilot VERIFIED, no improvement (2026-10-13)

Latest user explicitly requests implementation of token prediction followed by
frozen-encoder prefix re-encoding, a matched small pilot, and model cleanup FIRST.
This authorizes this cleanup and token-output experiment despite older contrary
notes below. No subagents; no production/full-v4 restart (still paused1001).
Cleanup audited seven duplicate last.pt optimizer states from three completed,
independently verified pilot families; final predictor weights retained. Freed
2,051,418,249 bytes. Audit dump/growing-token-cleanup-20261013.json. Baseline,
codec, paused runs, datasets, reports and archives unchanged. Historical old
verifiers requiring deleted optimizer snapshots cannot rerun unchanged.

New src/models/growing_token_memory/transformer.py preserves EXACT original
token-memory state layout and 23,209,216 trainable parameters. Both arms initialize
from pretrained token-memory v4 step48762; E frozen. New arm predicts token logits,
then E(committed prefix) REPLACES activecode until64, then appends it. No new-vector
regression, no input tokenIDs to growing reader. Fixed length features add no
parameters. Packed independent prefix queries cannot see each other or future
completed codes. Ordinary [END] output, cap32. Live generated tokens re-encoded.

Prepared dump/growing-token-pilot-20261013/data.pt: 2048SQuAD2+2048synthetic train,
128+128dev,256+256test. Old relevant source always512tokens/eightcodes. New synthetic
names across splits, four relations/eightvalues/three facts;20%missing;halfdev/test
heldoutwording. Question lengths exercise64rollover. FP32 cache37823unique codes,
56,118,377bytes; all12530dev+25533test synthetic factrecordtokens recover inclchanges.
Fresh128recodes max1.24e-5. SevenCPUtests pass. Fullsizezero-update GPUpreflight passed,
sameweights/count, cache/full/separate logits4.77e-6, momentbuffers peak907MB/582MB.
Matched1024updates/arm,8epochs,batch32,265296labels each,.9bodyCE+.1ENDCE,freshAdamW
warm30/lr1e-4cos1e-5/wd.01clip1. Baseline micro8,growing32; sameinitial/order/data.
Real oneupdate13376terminalsuccess; checkpoint and allAdam1checked. Resumed trainer
67322terminalsuccess0,2,048total newupdates. Independent verifier34623terminalsuccess0,
zero optimizerupdates. No GPUworker from this pilot remains. Sources are archived.
Final512test: answerable oldfacts baseline16/204,growing6/204; realSQuAD1/209vs0/209;
combined17/413vs6/413. Both0/204correct original+changedfactpairs. Only4/24outputs
respectivelychange whenfactchanges; outputchangealoneNOTfactualcorrectness.
Freshfamiliarname/wording128extraeval: both10/128overall,0/102pairedcorrect; baseline
0,growing8outputchanges. Trainfit64 baseline9/35synthetic+29/29real, growing8/35+0/29.
Newqueryprefix method learns no reliable factual retrieval and is not a replacement.
Training+periodicdev/save244.562/109.422s; final512generation5.844/26.094s including
liveprefixE, preencodedoldmemory both; unequaloutputlengths,notqualityspeedwin.

Verifier checks all39347causalquerylayouts,sourcearchive/protectedhashes,parameter
counts,allAdam1024,final/lastidentity,64savedanswers+128controlsperarmexactreplay,
targetmutation/blockedcontentinvariance,8cached/fullfreeanswersidentical (logits
max1.72e-5). Sampledtrueprefixcodec15950/15968tokens431/446exact;15failedprefixes
alllength<=8 inclonly2/8single-tokenexamples. Length11/12absentfromsamplebank.
Fulloldfactrecordsallrecover; shortprefixDerrorsareadditionaldiagnostic,notproof
ofirreversibleinformationlossorexplanationforallbadanswers. Pairedsynthetic95%CI
growing-minusbase[-8.82,-1.47]pp,oneinitializationonly.
Controls keepquestion/fixedoldslots. Original baseline blockedforward shifts raw
RoPEoffset by8(validcount);8answers match independentlyfixedoffset. Growingblocked
andallchanged/removed/wrong positionspreserved. Reportprotocolerratumincluded.

Aftertraining external D:/Git/latent-text-compressor/.venv disappeared; local
.venv nowtorch2.14.1CPU. Verification used existing D:/Git/WaveMixNN/.venv torch2.9.1
cu130, withlocalrepo site-packages appendedfortokenizers. NOinstall/envmutation.
Verifier5210 completedchecks butfailedfinalarchive relative__file__; fixedresolve
andreran34623success. verification-attempts.json preservesissue,notrainingrepeated.
Report docs/growing_token_memory_pilot.md, README/Step3/researchdirectionupdated.
Broaderusefulmemory/languagegoalremainsunresolved;nofull-v4restartrecommended.

# Grounded real-reading comparison VERIFIED, no winner (2026-10-13)

Latestuserasksifanythingbeatsbaseencodedmemory+tokens. Answer:notdemonstrated;
existingtrainedtoken-memoryv4remainsbestretainedbroadlanguagefoundation,notproof
optimalarchitecture. BroadgoalACTIVE,notachieved. No fullv4restart(paused1001),
nodeletion/subagents. No more experiments launched after this status question.

Newtools grounded_codec_data/prepare_grounded_codec/verify_grounded_codec,
grounded_codec_experiment/train_grounded_codec/verify_grounded_codec_training.
13newCPUtests passed; zero-updateGPUpreflightpassed. OfficialSQuAD2pinnedfiles,
8192train/512dev/512test,article/passage/questiondisjoint,literalv4overlapexcluded
2156reserveQs(notsemanticdecontamination). Relevantfullpassage<=384tokens,
question<=96,answer<=24,80%answerable20%missing. Noanswerspancrop/goldspaninput.
Passage/questionseparateBPEboundarysameforraw. Independent8tokenanswercodes,
ordinary[END],spacepadmasked. Truecodetest512/512answers,2404evidencetokens;
dev510/512answers,2667evidencetokens;twoanswererrorsretained.

All23,209,216trainable,fresh47,batch32,2048updates/eight8192epochs/arm,
629296assistantlabels(367152body),sameorder. Rawmicro8,vectormicro32,FP32.
CE=.9body+.1marker;vectorsMSEfirst200then+.02CE. OriginalE/Dfrozen(extraunique
12,555,776params). Vectorinputs/codesonly,unchangedpredictedfeedback,flow8Euler.
Actualstep1pause2810exit0;resume39200FINISHEDexit0,6144newAdamupdatestotal.
Outputdump/grounded-codec-20261013. FinaltestanswerableEMraw1/410(.244%),
deterministic0/410,flow0/410;F1 .597/.357/.147%. MissingEM35/102,0/102,17/102.
EOS511/0/196of512. Train+periodicdev/save509.875/206.875/237.375s;testgeneration
75.687/77.766/80.531s excludesencoding;differentstoppinglengths,noqualityspeedwin.
Rawalsoweak:cannotisolatecompressionorclaim23Mceiling. Notoldtrainedv4model.

Independentverifier2594FINISHEDexit0,passedtrue,zerooptimizerupdates.64hashselected
answers/codebytesreproducedexact/arm,Adam2048/final-lastidentity/count/frozenD/
source/protectedhashespassed.64controls(53answerable11missing),allarms0/53exact
correct/wrong/blocked. Wrong/blockedretainoriginalrefs:dependencenotnewtruth.
CompletequestionIDs/codesunchanged;passagelength/positionsmaychange. Outputs
changebutdoesnotshowusefulreading.2extraflowdraws0/53each.10reviewedfictionaldate
changesall0exact,changedtrueevidenceDrecoverypasses;9sametokenlength,one+1.
64traincheck49answerable:raw15/49,det0,flow0. No rawgeneralization/vectorfitproof.

Reportsdocs/grounded_codec_results.md/protocol,README/Step3/researchupdated.
Bestpracticaldirectiontask-alignedevidencereadingonexistingtoken-memoryfoundation,
ordinarylanguagepreservation. Selectivedecode-on-readisanexplicitconstraint-relaxing
option,nottestednewwinner. Allpositionfeatureexpansionalreadytried,notnewfixalone.
Originalv4/paused1001/codec/corpusunchanged. Dfree1.06GiB;nonewlargecacheordelete.

# Separate conditional diffusion head VERIFIED, no winner (2026-10-13)

BroadgoalACTIVE; meaningfulprogress,noblocker. No long/fullv4restart (paused1001),
no cleanup/subagents. Current user asks research using codec for memory/newtext.
Reviewed primary PISCO/xRAG/CALM/MAR/SONAR/SQuAD2, updated research recommendation.

NEW tools/conditional_head_flow.py,conditional_head_evaluation.py,
train_conditional_head_flow.py,verify_conditional_head_flow.py and7CPUtests passed.
Same23,209,216trainable:8layer512widecontext FFN1773/1774->1453/1454 frees2,621,440
for2width512biasfreeAdaLNresidualMLPblocks. Existingprojection/norm/head shared.
Constantcausalquery computesconditionONCE/block; onlyhead repeatsdenoising.
OriginalE/Dfrozen,noinputtokenIDs,new8tokenoutputcodeappendunchanged,noreencoding.
Readergraphonce, detachedconditionleaf accumulatesheadgrads, chainbackonce.

Same4352train/all17sources/literal[END]data,2epochs,batch8,1088updates/arm,
5,200,858assistantlabelexposures. MatchedvelocityMSEvsdirectcleanMSE,+.02CEafter200.
Samebatch/noise/time/Adam/FP32schedule as controls; newarchitecture reallocates
capacity,not one-scalarablation. Preflight36312 zerooptimizer: allgradmaxabs
9.835e-7/2.444e-6, cachedcontext2.027e-6; heaviestchecked20,096labels/2512codes,
2.166GiBallocated/2.461reserved inclmoment-sizedbuffers,notuniversalworstcase.
Batch1cachedprompt16steps283->41ms EXCLUDINGencoding/prefill/outputdecode.

Actualupdate1save/pause12894exit0; resumed92890FINISHEDexit0,2176newAdamupdates.
Run dump/conditional-head-flow-20261013. Devselect4/16Euler. Firsttestcontent
.16519/.79843%;exactblocks0/198of36134;0/34exactfreeanswers,0EOSboth,garbledtext.
Train+periodicdev/save267.079/267.500s; excludesprep/finaltests. 34cachedprompt
answers23.016/32.922s vs prior49.250/163.781s, all128cap;22refs>cap. Samplingsaving
withoutusefulquality; trainingnotfasterthancontrols. Originals unchanged.

Independentverifier33323FINISHEDexit0,0optimizerupdates; all34answerIDs+codefloats
perarmreproducedexact, final/lasttensorssame, Adam1088, frozenDoriginal, countpassed.
Fourdrawmeans.1788695655%/.8133402987%. Same32oldnotecorrect/changed/wrong/blocked
all0botharms. Full laterquestiontokens/codespreserved;prefixlengthpositionsvary.
Correct27answerable5missing;changed32answerable;wrong13missing;blocked32missing.
Narrowfull-location/missingphraseheuristic,notgeneralQA. Allguardedsource/protected
hashesunchanged. No worker from these sessions remains. Dfree~2.32GiB,no cleanup.
VelocitySHA4abcd16197202a049f5245d44478ec9519f6c1c3d78e1b7fb28077104b9fb500;
cleanSHAa2af06bdaab08eb59ee8b9c9bbe77c856423908fe9019cdab82c8973e0a13be7.

Reportdocs/conditional_head_flow_results.md,README/Step3/researchdirectionupdated.
Nextrecommendedgate is evidence-supervised realshortgroundedanswers, oracle
passageconstructionseparatefromlearnedselection, thenmulti-blockfreefeedback.
SQuAD2 proposed NOTprepared; no newdataset or longrun. Mereall-positionexpansion
alreadyfailed; changesmusttask-alignreader/objective,notrepeattoys. Token-output
reader/recoveredtextLM orrobustreplacementcodec are explicitconstraint-relaxing
alternatives,NOTsilentlyselected. Broadgenerationgoalnotachieved,noqualitywinner.

# Clean-code flow comparison VERIFIED, no winner (2026-10-13)

Broad goal ACTIVE; meaningful progress, no blocker. No full-v4 production restart
(paused1001), no deletion/subagents. Read required9fea attachment again. New
read-only diagnosis tools/diagnose_full_answer_flow.py completed session28492:
68train/68dev dialogues,999/970queries. True-code content99.9185/99.9580%; free
Euler4 .2308/.2519%, Euler64 .1086/.1679%; teacher90%-target99.837/99.916% is NOT
generation (no-context99.606/99.706%). More solversteps do not rescue prior flow.

New tools/data_prediction_flow.py,train_data_prediction_flow.py and4CPUtests
passed. Same23,209,216params, frozenoriginalE/D, noinputtokens, directcodefeedback.
Two arms velocity_cleanloss (endpointMSE=(1-t)^2velocityMSE) and data_cleanloss
(directcleanhead, samplingvelocity(d-x)/(1-t), no t1call). Both+.02CEafter200.
Same4352train/twoepochs/batch8/1088updates, init47/batch/noise/time/controldata.
Zero-update preflight60141 passed withAdammoments1.402GiBallocated/1.549reserved.
Actualoneupdate99033 saved1; resumed44088 FINISHEDexit0,2176totalnewupdates.
Run dump/data-prediction-flow-20261013. Testcontent.35368/.78925%, selectedEuler4/16,
exactblocks0/88 of36134; 0/34exactfreeanswers,0EOS both, incoherent output.
Train+periodicdev/save255.297/255.5s; finaltests/prep excluded; answers48.516/163.781s
cachedinputs/128cap (22refs>cap), noqualityspeedwin. Originals protected hashes unchanged.

Independent verifier14979 FINISHEDexit0, zerooptimizerupdates, workerlockreleased.
BOTHarms34answers+codefloats reproducedexactly, final/lasttensorsmatch,Adam1088/
frozenDidentity;4drawmeans.35897454%/.80654555%. Both32old-note correct/changed/wrong/
blocked all0;correct27answerable5missing,changed32answerable. Tightenedmissing-
locationheuristic alsorescoredoriginals all0,removingpriorrawfalsepositive.
Full laterquestion unchanged;prefixlength varies. Dataheadmemory-source20.59%
nextblockagreementincludesfixedphrases,NOTfactualaccuracy. Allprotected/source
hashesunchanged,percaseoutputs/results/sourcearchivesretained. No worker from
these sessions remains. Docs/data_prediction_flow_results.md/README/Step3 updated;
full_answer_flow_protocol.md nowexplicitlyhistoricaloriginalpreparation. No winner.
Research doc updated primaryMAR2406.11838v2 separatecontext+smalldenoisinghead idea,
notimplemented orqualityvalidated. Maincleanlossmotivation2603.06454v1. No more
training launched. Dfree~3GiB; no cleanup. Keep original/control artifacts unchanged.

# Full-answer literal-EOS comparison VERIFIED, no winner (2026-10-13)

Broad goal ACTIVE; this turn made progress, no blocker. User asks research on
using the codec for useful memory and new language. No new optimizer updates
this continuation: finished independent verifier session33654 exit0. Trainer
71894 previously finished exit0, all3arms1088updates each (3264total); no worker
from these sessions remains live. Do NOT restart production/fullv4, paused1001.
All originals/source/protected hashes verified unchanged. No deletion/subagents.

Run dump/full-answer-literal-flow-20261013 complete. Exact23,209,216trainable each;
same4352train dialogues/twoepochs,batch8,5,200,858rendered assistantlabel exposures.
Rawmicro2/vector8; MSE+CEdeterministic andvelocityMSE+.02CEflow (CEafter200).
Frozen originalE/D, vectorarms noinputtokens, sampledcodefeedback unchanged.
FinaldevselectedEuler4 forflow. Test contentagreement raw32.8414%,det3.3660%,
flow.16696%; different withinblockteacherconditioning, NOTcomparableNLL.
Free34answers: all0exact, 22references>128cap; rawEOS10,det0,flow0. Actualtext:
rawgeneric/repetition,detcharactercollapse,flowincoherent. No usefulchatwinner.
Meanrepeated4gram raw.3581/det.7085/flow0; randomgarbage'slowrepetitionNOTquality.
Training+periodicdev/save seconds416.265/224.297/258.125; excludesprep/finaleval.
Answerseconds58.094/20.203/49.250,cachedinputs,unequaloutputlengths,notqualityspeedwin.

Independentverification.json passed, all34savedanswers/arm exactreproduction,
vectorcodefloatsexact,final/lasttensorsidentical,allAdam1088,frozenDoriginalmatches.
Fourflowdrawscontent.16696/.17402/.17084/.16166%,mean.16837%;0exact/36134each.
32oldnotecontrols preserveENTIRElaterquestion/tokens/codes:correct27answerable5
missing;changed32answerable;wrongdonor13missing;blocked32missing. Allarms0/27
correctfactand0/32changedfact;allblockedheuristic0. Rawwrongheuristic1/32isFALSE
POSITIVE onmanualread: "I don't have a notebook in the notebook in the notebook."
Notusefulabstention/retrieval. Originalautomaticresultretainedandcaveatedinreport.
Controlschangeprefixlength/positions;notpureattentionablation. Freshrawalsofails,
so cannotattributeallfailuretocompression. Earlierfullv4raw8/vector0is separate.

Readresearchgoal andStep3,reviewed primaryCALM/PISCO/xRAG/SONARpapers+SONARcode.
Researchdoc now distinguishesstorage,learnedreader,newcodeconstruction,grounded
QA/SQuAD2 proposal,teacheronlyonverifiedanswers,lossseparation. Fullpositioninput
expansionalreadyfailedsmallpilot;don'tpresentitasauntriedsolution. Outputtoken
readerdiagnostic and robustreplacementcodec are ALTERNATIVES thatrelax current
directcode/frozencodecconstraints;not silentlyselected/implemented. Frozen23M
pathstillunresolved;nofullruncommandrecommended. Codec256FP32=1024bytes versus
64uint16IDs128bytes:attentionpositioncompression,notdiskbytecompression. No
quantizationclaim. Docs/full_answer_flow_results.md, frozen_codec_research_direction.md,
README/Step3 updated. Broadergoalnotcomplete;failedboundedrunisnotimpossibility.

# Historical full-answer comparison preparation (2026-10-13)

Active goal continuation made concrete progress. No blocker, no winner. User
authorized bounded flow/other experiments, not full-v4 restart. Production
position-memory remains paused1001. LIVE trainer session71894, raw arm last
observed650/1088 (worker was live at poll). Do NOT launch competing GPU worker,
do not edit guarded sources while running. Poll actual71894 handle to confirm
completion, not stale status. Prior one-update session25408 finishedexit0 at1;
current --resume restored1, no discarded researchsteps. Three sequential arms
raw/deterministic/flow_ce, each1088updates (two epochs4352dialogs,batch8), total3264.

Important newly measured codec mismatch: original genuine targets reconstruct
ZERO of738dev/778test EOScontroltoken2; additional220/204nonEOSerrors. Original
codec trained ordinary text, not thesecontrols. Read-only verification37139
finishedexit0; evidence dump/full-answer-generation-check-20261013. Full23M
cached/fullflowcodeerrors<2e-6, decodedidentical;rawlogits<5e-7. Allparamsunchanged.

Reversible derivedformat replaces2 with literal[END] IDs61,3907,38,63, preserves
masks, rejects originalmarker collisions and checks reversibility ALLrows.
Tailpaddingordinaryspace223 excludedCE. BOTHcomparisonarms sameview. Original
data/weightsunchanged. tools/full_answer_codec_format.py +prepare_literal_end_flow.py;
session34079 finishedexit0 in216.641s, outputdump/full-answer-literal-end-20261013.
All738/738dev,778/778test markerscorrect;contenterrors215/197. Notlosslesscodec.
Train2600429labels328044codes;dev285496/36015;test286419/36134;maxrenderedlen2567.
No general arbitrary-content escapingclaim. Train-only normalization, sourcezip.

New full_answer_flow_generation.py cachedmultiqueries, causalappendpredictedcode,
rawKV, finalassistantspan split, EOSsequence/cap, target-free generation. New
full_answer_flow_evaluation.py separates teacherhistoryblocks/contentmetrics and
self-fed lastanswers2/source cap128. Endmarker cancross8block boundary.22CPUtests
passed inclserializedAdam/model/RNGresume;pytesttmpfixturehostACLworkaround uses
uniqueordinaryworkspacefixtures, no deletion. Trainer train_full_answer_flow.py
exact23.209M each,rawfactorizedembed+FFN1646 vsvector1773/1774,samefamily. Fresh47,
sameepochshuffle69410,noise69411+step,time79411+stepGaussianCDFuniform,50warmcos
3e-4to3e-5,wd.01clip1; MSE/.02CEafter200vector,rawCE. FinaldevselectEuler4/8/16.
Rawmicro2 vectormicro8;4096capacityforrenderedmarkers,sameoriginaltext. Cache prep
and runningarm sources frozen inrun. No update to guardedcodeuntil jobstops.

Run dump/full-answer-literal-flow-20261013. NEW independentverifier
tools/verify_literal_flow_comparison.py notyet executed. Waittrainerthenrun it;
reproducessavedfreeanswers/frozenD/Adam/hashchecks, fourflowdraws,32memorysource
correct/changed/wrong/blocked FIRSTnote prefix withentirelaterquestionuntouched.
Diagnosticlengthpositionschange blocked; value substringscore notgeneralQA.
Verifyparser coversactualmemorycases beforeclaimingresults. Report
docs/full_answer_flow_results.md currentlyrunningprotocol. Do notmarkgoalcomplete
frompreparation or nextblockagreement. Broadgoalgenerationqualitystillunresolved.

# Research review and full-answer zero-update preparation COMPLETE (2026-10-13)

Latest user asks how to use existing codec to train useful memory/generation.
Reviewed actual codec, research goal, grounding audit and primary CALM/xRAG/
PISCO/ICAE/SONAR/LCM papers. Research report docs/frozen_codec_research_direction.md
now separates reconstruction, task-aligned reading and new-code generation.
PISCO reconstruction/keyword continuation did not ensure QA; task supervision
is relevant but their billion-scale adapted LLM is not our frozen parallelD.
Recommend explicit grounded-reader curriculum and separately normalized memory
loss, matched raw control; direct-code flow remains unproven. Token output or
codec retraining are discussed alternatives, not selected constraint changes.
No new research optimizer updates, production restart, deletion or subagents.
Broad goal active; no winner or genuine blocker. Production still paused1001.

Preparation session32138 finished exit0. tools/prepare_full_answer_flow.py saved
dump/full-answer-flow-data-20261013 in187.281s, no optimizer.4352train/489dev/504test;
2578931/283282/284085 assistantlabels;325409/35709/35830codes. Normalization all
selectedtraincodes only. Protected/reference/data/source hashes unchanged.
Truecode oracledev282324/283282tokens34762/35709blocks;test283103/284085tokens
34859/35830blocks. Memorysource952/1049dev941/1038test; error category NOTyet
classified (do not assume allEOS or callcodecperfect). Fullsize preflight session
24484 finishedexit0; evidence dump/full-answer-flow-preflight-20261013. No worker
left from these jobs. Do not restart them on stale status.

New tools/full_answer_flow_common.py packs fullanswer losses and exact23,209,216
raw-token control: tied4096x256embedding, biasedprojection,8FFNs1646wide. Vector
has1773/1774FFNs instead; samefamily notoldGELU architecture. Allparamsactive.
tests/test_full_answer_flow.py +test_parallel_flow_queries.py:9CPUpassed.
Real23MGPUquerypred/allgradient vsseparatelytruncated, rawchunkedCEvsnaivepassed,
gradientmax<5e-8. Zero optimizerupdates; model/codec unchanged. Repeatedlongest
real2509token314querydialogue:compactmicro8 .437s2.434GiBallocated2.598reserved;
rawmicro2 .203s2.697/2.842. Momentbuffersincluded;step/eval/save/initialencoding
excluded, notthroughput/ETA. Expandedpositionprobescapacityonly, notselectedfix.
Rawteacherforcinginside8tokenblockdiffersfromjointvectoroutputs; sampledDCE
isNOTrawLMlikelihood/perplexity. No matchedtraining/inferenceclaimedready.

Docs/full_answer_flow_protocol.md updated afterimmutablepreparationfinished;
oldarchivedprotocol is historical, do not requirecurrentdocmatch oldsnapshot.
Next broadworkstill needs cachedsampling/fullanswerEOS+self-fedcodeevaluation,
explicitboundedmatchedschedule/resumabletrainer beforeanybroaderpilot. Do not
silentlylaunchlongfullv4 or treatpreparationasanachievedlanguagegoal.

# Separate decoder reader COMPLETE; full-answer preparation only (2026-10-13)

Broad research goal remains ACTIVE, no winning general-language model and no
blocker. Latest work made progress. Original production position-memory run is
still PAUSED at step 1001; do NOT restart full-v4. Sessions 98478 (bounded trainer),
18453 (independent verifier), and 42335 (train-fit reload) all finished exit 0.
No GPU worker from this turn remains. Older RUNNING entries below are superseded.
No subagents, cleanup, production source changes or reference-weight changes.

Both expanded-reader arms completed 2,000 retained updates. Exactly 23,209,216
trainable, original encoder/decoder frozen, no input token IDs, direct eight-token
codec-vector output. Separate valid frozen-D position states, same old model,
initialization, targets/objective/data order. Expanded micro8 vs old compact32;
same effective batch64. New test256 has 143chat/113stories (stories absenttrain).
First-draw agreement compact/expanded: deterministic3.56445/1.22070%, flow+CE
1.51367/.43945%; all0/256exact. Four-draw flow1.7700/.4517%, all0/1024exact.
Expanded train checked256: one-pass100%exact (independently reloaded), flow2.34375%
token agreement/0exact. Tiny first8 pilot has16,384distinct targetpositions and
1647unique answers; overfitting/poor trainingfit, not a general capacity ceiling.

Independent tools/run_decoder_reader_verification.py passed: sourcezip+protected
hashes, originalD tensors, exactparamcount, Adam2000 counters, final/last identity,
saved IDs/first codes exactly reproduced, four fresh data recodes, targetmutation,
blocked invariance. Cached/full identical-noise sampler maxabs3.58e-6/1.91e-6,
decoded tokens identical on8cases. Verification fixed an untested assumption by
using identical noise tensors instead of different-shaped CUDA RNG draws; active
trainer/helper sources were untouched. Paired expanded-minus-compact95%CIs:
one-pass[-3.9063,-1.0254]pp, flow[-1.8066,-.4395]pp, one trainingseed only.
Chat compact/expanded6.3811/2.1853% one-pass;2.4476/.6993% flow; stories0/0 and
.3319/.1106%. Whole-history wrong/blocked includesquestion, NOToldfact-only proof.

32prompt sample+decode timing: compact onepass20.8ms/expanded143.2ms; compactflow
135.3ms/expanded2226.7ms. Expanded uses4micro8 incl historyKV rebuild; compact32;
selected Euler8 compact vs32expanded. Inputencoding/featurecreation excluded.
Expanded training+periodicdev/save1298.922/1318.453s; successful launcher totals
~49.1min including finaleval, excluding prep/independentverify/discardedattempt.
Earlier killed firstattempt >=300updates lost, exactdiscardedcountunknown; do not
claim only4000totaloptimizercompute. Successful peaks7.94GiBallocated/8.54reserved.
Full-v4 token QUALITYREFERENCE48762updates gives44.3848%,73/256exact onthisset,
chat38.4615/stories51.8805%; NOTmatchedtrainingbudget. No winner/usablechatclaim.

Evidence dump/decoder-reader-flow-20261013/results.json, verification.json,
train-fit-reload.json. Verification sources/launcher/probe archived in
verification-sources-20261013-171554.zip with successful verification-launch JSON.
Deterministic SHA6813f0ab68ccc1728465e6a484c9ca77045728c257ebe303d3d1072c5089997f;
flow SHA474b0c8ba490115519c51f6a02502c4454823702703a49f64b7689b0f0b56d99.
Report docs/decoder_reader_flow_results.md; README/Step3/researchreview updated.
Research review verifies primary CALM/LCM/SONAR/ICAE/xRAG papers and SONAR code;
no paper validates our exact frozen-codec/direct-feedback23M design. xRAG trains
a task-aligned bridge plus contextual answering/distillation, much larger LMs.

New PREPARATION ONLY (no GPU training/encoding): tools/parallel_flow_queries.py
shares history across independent causal output queries. 3CPU tests pass in
tests/test_parallel_flow_queries.py: predictions/all parameter+input gradients vs
separate truncated forwards, future/otherquery/padding invariance, invalidhistory.
No full-size CUDA parity/capacity/speed verification yet. Not used by abovepilot.

tools/audit_full_answer_flow_data.py creates seeded source-stratified complete
dialogue selection,17sources. Train4352dialogs/2,578,931assistanttokens/325,409codes;
dev489/283,282/35,709;test504/284,085/35,830. All labels/masks/tails/EOS/boundaries/
source hashes checked, train/dev/test IDs disjoint. Excludes1024previous bounded
test IDs. First selection missed first256pilotIDs and had25overlaps; preserved
under dump/full-answer-flow-data-audit-initial-exclusions-20261013, corrected
before any encoding/training. Correct data audit/selection/source archive:
dump/full-answer-flow-data-audit-20261013. No near-duplicate exclusion claim.
docs/full_answer_flow_protocol.md records comparison/evaluation requirements.
Do not silently launch this broader training. Needs full-size zero-update CUDA
gradient/capacity check, efficient immutable-code preparation, a matched token
control, explicit bounded schedule, and full-answer/EOS/self-feedback evaluation.
The broader goal is unresolved; another narrow toy success is insufficient.

# Separate frozen-decoder reader diagnostic RUNNING — historical, superseded (2026-10-13)

Broad goal ACTIVE; this turn has made concrete progress, no blocker. Production
position-memory still PAUSED1001. Do NOT restart full-v4. Current research worker
LIVE unified-exec session98478, PythonPID14652 belonged to preceding501 invocation
(do not assume currentPID). Use actual session/status/process evidence. Command:
tools/run_decoder_reader_limited.py --resume. Last observed deterministic1100/2000;
then flow_ce2000 automatically. Do NOT launch a second trainer/verifier while it
holds worker_lock. Status.json updates only every500 and can temporarily show
the old501pause despite an active resumed worker. No subagents.

tools/decoder_reader_common.py keeps EVERY valid frozenD normalized position as
a separate input; no mean pooling records/positions, no tokenIDs. Original
FlowModel/outputgeometry/objective/initialweights unchanged, exactly23,209,216
trainable, originalE/Dfrozen. Contrast existing position_memory_joint pooledreader
plus softrecordmix. Inference caches historyKV once; exact factoring, tested CPU
and real23M field maxabs4.77e-7. Training uses full field and gradients.

Same train2048/dev256 as old flow; source458everyday+1590chat; first8tokens ONLY
of first eligiblefull64assistantspan.16,384targetpositions/1647distinct8answers,
not proper languagepretraining or fullanswers. NEWtesteligible768:1024,143chat+
113stories (storiesabsenttrain); mustreport per-source. Genuine8testcodec2048/2048.
IDsdisjoint/literalprefixoverlap0;near duplicatesnotexcluded. Wrong/blocked alters
questiontoo, NOToldfact-onlyproof. Independent8codes, notgrowingfeedback/EOS.

tools/prepare_decoder_reader.py completed, dump/decoder-reader-flow-20261013.
Train396998positions(mean193.85/max2036) vs8142originalcodes;~.46GiBFP32featurebank.
This loses attentioncompression; storage remainscompressed. Preflight longest8
7.76GiBallocated inclmomentbuffers;2CPUtests pass tests/test_decoder_reader.py.
Same initial OLD dump/flow-vector-real-20261013/initial.pt,2000updates/arm,batch64,
micro8,lengthsortwithinbatch,noise62911/batch62910,AdamW3e-4,MSE+.02CEafter200.
Old compact controls trainedmicro32, re-evaluated final with0newupdates; group
roundoff allowed. Protected SHA/sourcearchive guards remainunchanged.

First attempt session90244 reached at least300steps, WindowsWDDM2.52GiBshared/
11.19GiBdedicated,0.61updates/s. TerminalCtrl+C KILLED before handler/checkpoint:
NO retainedweights, exactdiscardedupdatecountunknown. Mustdisclose wastedwork,
never claim exactly4000totalcompute. Archivedinterrupted-results.json/sources.zip
and interrupted-attempt.json. No original/reference files affected.

New memory-only launcher run_decoder_reader_limited.py setsallocatorfraction.75,
GCthreshold.75. Zero-update probe_decoder_reader_allocator.py passed12effective
batches .531-.937s,8.40GiBreserved;longest8allocated7.81. Allinitialtensorsunchanged.
Freshsameinitial/seedsattempt session15214 completed--stop-after501: dev500acc
1.171875%,0exact/256,CE32.6811,MSE1.3870; badpreliminaryvalidation. Throughput1.58/s,
peakallocated7.938/reserved8.541GiB; noWDDMshared>100MBsample. OriginalD/source/data
identityandallAdam501verified. Saved501SHA3395fc5570d4c64b883816213db054ab5fba5a5e3326596a7cafdf89634b5bab.
Resumed samecheckpoint successfully in98478;no coldGPUbitwiseclaim. Do NOTuse
terminalCtrl+C again; programmatic--stop-after is safe. Original trainer sources
notmodified; launcher/probe need separatelyarchived with final verification.
At1000trainingCE.734/MSE.02295 vsdevCE43.153/acc1.171875%/0exact/MSE1.32778:
clearpoorheldoutperformance despitegoodtrainfit;notyetfinal2kresult.

When98478FINISHES run tools/verify_decoder_reader.py (prepared NOTexecuted),
prefer same allocator budget for GPU. Verifies sourcezips/protectedhashes/frozenD,
Adam2000/final-lastmatch, savedoutputs,reencoded4records,cachevsfull,targetmutation,
blockedinputinvariance,per-source scores,pairedCIs,timing,full-v4tokenreference.
Verifier uses fixed8cases for fullcacheparity and4freshdatarecodes. Reference has
48762updates and rawrecenttokens, notmatchedbudget. Frozeninputfeature decoding
excluded from timing; expandedmicro8 vscompact32 disclosed. Add verifier/launcher
source provenance. Do NOT proclaimwin basedonlysingle-refaccuracy or thissubset.

docs/decoder_reader_flow_results.md currentlyINTERIM; updateaftercomplete.
README/Step3/researchdirectionupdated. Researchreview nowchecks SONAR-LLM actual
code: frozenDreceives teacherforcedrawtokens, generation RE-ENCODES text before
feedback; notproofourparallelD/directfeedbackworks. SONARsmallcounts11M/34Mexclude
codec; ordinarytokenbaseline winsreportedstoryquality. CALM/LCM/ICAE/500xprimary
sourcesalsoinresearchdoc. Nextbroaderprotocol MUST include sourcebalance,successive
8targets throughoutanswers,shorttails/EOS,consistent8assistantfeedback and equally
trainedrawcontrol. No newdataset/fulltraininglaunched. Remainingbroadergoalopen.

# Preconditioned flow pilot COMPLETE, negative; broader goal ACTIVE (2026-10-13)

Position-memory production still PAUSED1001. No long training resumed or original
weights/data changed. Required 9fea8f41 attachment read again. New helpers:
tools/preconditioned_flow_common.py, audit_flow_geometry.py,
train_preconditioned_flow.py, verify_preconditioned_flow.py;
tests/test_preconditioned_flow.py. Evidence dump/preconditioned-flow-20261013.

Train-only 2048-target covariance condition9.26million; top64 axes99.9985% variance.
Full-rank ZCA eigenfloor1e-5 retains ALL256 dimensions, inverse unchanged decoder
IDs on2560 genuine codes, maxFP32roundtriperror.000733. Conditioning becomes5.31.
NoiseRMS.3 dev tokenaccuracy isotropic95.12% vs covariance-shaped26.42%; cannot
assume all out-of-manifold noise is worse or low-variance axes are uninformative.

Three fresh matched2k-update arms from exact prior flow initialization, same
2048train/256dev, batch64micro32,AdamW3e-4,CE.02after200,23,209,216trainable,
originalE/Dfrozen. Output/active-state coordinates+regressiongeometry only change;
history originalcodes. Newtest256 eligible512:768,allchat,no literal prefix overlap
withtrain,IDsdisjoint; near-duplicatesnotexcluded. Onlyfirst8assistanttokens,
not fullv4/answers/EOS/self-fedrollouts. Old controls re-evaluated with0updates.

Old/new firstdrawtokenaccuracy onepass6.543/5.371%,flow1.318/2.051%,flowCE2.051/
2.246%; exactblocks5/1,0/0,0/0. New4drawflowmean2.319/2.429%,0exacteach1024draws.
Pairednew-minusold95%CIs [-2.54,.05],[-.05,1.61],[-.73,1.17]pp; no significant
improvement, all poor samples. Newtrain256acc41.26/19.43/13.87 vsold99.95/10.11/
15.72. Truecodecnewtest2046/2048tokens,255/256exact. Fullv4tokenmemoryreference
sameprompts40.576%/64exact,48762updates; MUCHmoretraining/rawrecenttokens, not
matchedbudget. Wrong/blockedchangesquestionalso,notoldfact-onlyretrievalproof.

Newflow samplerdev-selectedHeun16=32NFE;newflowCEEuler16. Batch32sample+decode
19.6/541.7/273.4ms vsold19.5/138/138, encodingexcluded. Newtrain+periodicdev/saves
239.84/221.17/246.78s,finaltests excluded.6knewoptimizerupdates total. Cross-draw
tokencollisionold/new3.82/61.52%flow,5.77/45.30%flowCE. Brier token estimates
also worse; metrics not publishedBrierLM/perplexity,4drawsonly.

4CPUcontracttests pass; Brierlimits+Bernoulliexpectation checked. FreshGPUreload
all generatedIDs/firstcodes bitwise, geometry trainfit bitwise, originalDidentity,
paramcount, Adamall2000, final/lastmodelmatch,4freshrecordrecodes,sourcearchive,
protectedoriginals verified. Samplerhasnotargetargs;blockedmemoryinvariant.
Pilotresume exists,fullinterrupt/replaynot separately verified. Training54460
andverification79304finished. No researchworkerleft. Report
docs/preconditioned_flow_results.md; docs/frozen_codec_research_direction.md has
primary CALM/LCM/preconditioning/diffusion-head sources and proposed next gate.

Do not repeat coordinate tuning as if generation solved. Next meaningful reader
diagnostic: original compactcode inputs vs SEPARATE all-valid frozenD position
states, same output model/data/budget; no early mean over record positions or
soft averaging different records together. Deliberate attention expansion/cost
must be disclosed; only then test resampling/selective expansion for efficiency.
Currentposition-memory already mixes64states with a pooled reader, so do not
rename that as a new approach. Full-language/self-fedfeedback/robustfacts/baseline
win remain unresolved. Goal ACTIVE, not complete or blocked.

# Continuous flow pilot COMPLETE, negative; broader goal ACTIVE (2026-10-13)

Actual production position-memory status PAUSED1001, not historical402. No long
training resumed. New tools flow_vector_common/train_flow_vector_pilot/
verify_flow_vector_pilot/audit_flow_vector_data.py; tests/test_flow_vector_pilot.py.
Three matched fresh23,209,216-param arms, originalE/Dfrozen,2kAdamWupdates each,
batch64micro32,3e-4, sameinitialization/order. ContinuousGaussian->target straight
path; velocityMSE, with/without .02CE after200; matchedonepassMSE+.02CE. Eight
existingphasevectors provide continuous time; activeamplitude retained. No raw
tokeninputs; direct8token-codeoutput. No production model edits.

2048train(458everyday/1590chat),256dev,NEW256test(allchat,afterprevious256).
Onlyfirst8assistanttokens; up to32pastcodes. Notfullv4/longretrieval/rollouts.
No exact cross-split prefix overlap, IDsdisjoint;near-duplicatesnotruledout.
Train-onlynormalization. Truecodecdev2048/2048,test2046/2048tokens(254/256blocks).
DevEuler8/16/32,Heun16 comparison selectsEuler8 bothflowarms. Freshtestfirstdraw
accuracy6.3477/2.1484/2.0508%;exact4/0/0 of256 (onepass/flow/flowCE).
Fourdrawmean1.6602/2.2339%,both0/1024exact. Train256acc99.951/10.107/15.723%.
Sample+decodebatch32:19.5/139.3/140.7ms,excludesencoding. Train+periodicdev/save
240.4/220.6/245.7seconds,excludesfinaltests. No winning generative model.

Unchangedfullv4tokenmemory48762 QUALITYREFERENCE same8targets:40.2832%,57/256exact.
Muchmoretrainingandrawrecenttokens;NOTfairtrainingbudgetcomparison. Controls
wrong/blockedalsochangequestion;NOTproofoldfactretrieval. TwoCPUcontracttests;
independentGPUreloadbitwisecodes/IDs,originalDidentity,count,Adam2000,last/final
tensors,shortdecodegrad/logitparity,4testrecordrecodes,protectedallv4rootfile/
reference/codec hashchecks. Firstzero-updateverifierinterruptedafterlongstartup;
CPU->CUDAstagedrerunpassed. Nooptimizerretry. PilotresumeCLIexists,fullinterruption
replaynotseparatelytested. Evidence dump/flow-vector-real-20261013; report
docs/probabilistic_vector_results.md. Allworkerhandles93352/46313/78961finished;
4032explicitlystopped. Generalgeneration/EOS/directfeedback/baselinewinUNRESOLVED.
Do not launch longfullv4flowtraining fromthese results.

# Position-memory step402 speed/loss check (2026-10-13)

User full-v4 run dump/position-memory-joint-v4 is PAUSED at402. Do not restart
long training for user. Checkpoint and original codec hashes unchanged. Read-only
32dev-dialogue check:13377tokens/1690codes,CE15.74091,accuracy1.62966%,MSE1.18977
vsconstant train-mean1.09469;truecode oracleCE.17097/99.6113%. Early poorvector
prediction,not decoder inability on truecodes;no quality/impossibility conclusion.

Production speed update:flat pack scored constructor queries;read/mix once,
accumulate constructor gradients on mixedmemory+reader leaves thenchainbackonce.
FrozenD only computes first8 positions for8outputs (remaining56were masked keys).
No weights/precision/objective/targets/causality/budget changes. All64constructor
positions retained. Progress shows per-update allocated/reserved separately.
13CPUtests passed;real fullbatch32 old/new allgradient maxabs2.98023e-8.
Before sourcezip and exact before/after speed_migration.json allow paused402resume;
unknownchanges refused. Original source.zip retained, newarchive made onmigration.
Optimizer/RNG/cursor/horizon intact, FP32 grouping roundoff allowed.

Same3batch micro8query8 zero-update times5.87/6.79/6.63 ->3.60/2.99/3.54seconds.
Recommendedmicro8query16,8consecutiveprobes2.02-3.12s,allocatedmax5.541GiB,
reserved6.948GiB. RestoredactualAdambuffers,NOoptimizersteps; excludesstep/save/eval,
notwhole-runETA. TF32alone gave no meaningfulspeedgain,notenabled inproduction.
Suggested USER bounded resume--stop-after599 runs1000validation thenpauses1001;
stopping exactly1000 skipsperiodiceval due safe-stoppriority. Reviewvalidation;
effectivebatch32 unchanged. No researchoptimizerupdates launched. Evidence
dump/position-memory-speed; tools/diagnose_position_memory_speed.py and
tools/verify_position_memory_speed.py. Broadgoalremainsactive.

# Full-v4 position-memory joint trainer PREPARED (2026-10-13)

User requested proper full-v4 trainer, fixed23M, frozen E/D, data checks and a
resumable command; explicitly DO NOT launch long training. New model-local
src/models/position_memory_joint/{training,transformer,data,generation}.py and
README. Exactly23,209,216trainable via unchanged growing_vector budget/backbone;
original encoder+decoder frozen. Existing source models/weights/data unchanged.

This is an EXPLICIT full-language adaptation, not unchanged toy routing: v4
does not have general gold source/order labels. Differentiable soft record
attention feeds64position-preserving mixed features plus reader state into the
same predictor again. MSE+0.02CE trains both passes; no toy source/plan CE or
general probabilistic prior. Independent8-token codes, NOT8->64replacement.
Inputprompt spans<=64, priorassistantcodes8. Targetcodes loss-only forcurrent
query, priortruecodes teacherforced later; future memory masked, queriesisolated.
Generation appends predictedcodes directly, stopsdecodedEOS, no re-encoding.
Feedback/chatform quality NOTestablished. Tests checkcontract only.

Samev4tokens/masks/splits:483862train,4852dev,4781test;180678552assistanttargets
and22865005vectortargets/trainepoch. Whole-dialogue batches:2epochs batch32 =>
30242updates,22dialogsfinalbatch/epoch. Not matchedstep/batchconditioning vs
old512segmenttrainer. Shorttails EOSpadforMSE only, CE scoresrealEOS/labels once.
Fresh seed47,train-only256dialogue latentnorms,AdamWwarmup2003e-4cosine3e-5.
Optional--init weights-only/preservednorms, freshoptimizer; toytransferunproven.
Same-folder--resume restoresAdam/RNG/epochcursor/LRhorizon/source/datahashes.
Micro/querymicro can change. Best/last/previous withSHA/sourcezip/logtailrepair.
Defaultdevsubset256dialogues,fullfinaldev4852. --stop-after enablesboundedpause
without changinghorizon. Ctrl+C safeupdateboundary. No researchoptimizerupdates.

FeatureLRU2048(~128MiBvalues)+codeLRU10000, no disk/fullcorpuscache. One reader
forward/backward per microbatch: boundedconstructionloss accumulates gradients
on temporaryreader/selectionleaves thenchainsbackonce. Naive/allgradientparity
tested.11CPUtests inclbitwise same-micro syntheticpause/resume Adam/RNG/cursor,
regroupedmicrotolerance, masks/EOS/counts/causality/blockedmemory/freeze/direct
feedback. Real23MCUDAgradientmaxabs1.714e-7. ColdcacheGPUresume notbitwiseclaim.

tools/audit_position_memory_v4.py read-onlyfullfilehash/mask/EOS/group/count audit
passed,1007actualplanssampledacrossall17sources/splits. tools/preflight_position_memory_v4.py
zerooptimizerprobes:micro16query8firstbatch5.36s/7.20GiBallocated8.63reserved;
heavy7.75s/8.11allocated8.58reserved;syntheticmax2560tokens12.125s/8.34allocated
9.00reserved. IncludesAdam-sizedbuffers,excludesstep/eval/save.Notwhole-runETA.
Originalv4/growing4595/pilot/codec/manifesthashesunchanged. Sampletrue8targetcodec
7865/7920tokens,951/1006exactblocks;55errors48EOS.Donotclaimcodeclossless.
Evidence dump/position-memory-v4-verification;README/Step3linksupdated.
Recommend micro16/query8 for12GB withfallback8. No full-corpusjobstarted.
Broadergenerationgoalactive,notcompletedbyimplementingtrainer.

# Full-position construction and joint recovery COMPLETE; broader goal ACTIVE (2026-10-13)

Lastgoalturnwasprogress,nogenuineblocker. Required9fea8f41attachmentreadagain.
FiveNEWtools:train_position_vector_constructor.py,preflight_position_constructor.py,
verify_position_vector_constructor.py,train_position_memory_joint.py,
verify_position_memory_joint.py. No productionmodel/v4dataedits. Source/predictor
still23,209,216trainable,originalE/Dfrozen,directcodecvectoroutput,norawtokenIDs.

Matchedconstruction8pooledgroupsvsall64Dnormalizedpositionstates:commonparent
dump/slot-vector-constructor-20261013/slots/predictor.pt(14klineage),2kadditional
eachbatch64,cosine1e-4to1e-5,MSE+.02CE. CORRECTSOURCE+ORDERsuppliedtoisolatereadout.
Newcleantextprefixescompletejoinedbeforetokenization,nooldwasFileartifact.
20,480recordbank,16trainlayouts,bothfieldorders,768train/240heldnonidenticalpairs,
heldreversalsdisjoint,40train18heldnames. CachedfullFP32Dstates1,363,656,869bytes,
prep25.141s,bankSHA2cba8eefa227f8cd8f95ce9e6fda174c8ff4057d2794d4644a774d29eaf4e081.
Outputdump/position-vector-constructor-20261013/8,/64. Each16ktotallineage.
Fresh512cases/familyoracleexact8/64:trainbank39.9414/99.2188%;newpairs35.5469/
95.8984%;newnames29.0039/92.0898%;oldheldlayouts.3906/27.3438%;newheldlayouts
.0977/8.0078%. Pairednewpairdelta95%CI[55.86,64.65]pp. Fullstatesmeaningfulgain,
notproofaveragingirreversiblylosesallinfo. Train+dev133.922/155.266seconds.
Construction-onlysharedreaderweakened:full64autonomousnewpairs17.5781%.

Then1000jointupdatesONLYfull64parent,batch64,LR3e-5to3e-6,MSE+.02CE+.25sourceCE+
.25planCE. Outputdump/position-memory-joint-20261013;final17ktotallineage.
Sameweightsusedtwice:readerpooledDfeaturesselectsrecord/samples2-wayorder;
constructorexpandsselectedrecordtoall64positions,predictsNEWanswercode;frozenD
emits8tokens. Goldsource/plantrainingonly;predictioninference. Fixedtwoorderplan
isNOTarbitraryprobabilisticlanguagegeneration. Cachedbankallowsfastsyntheticrun.
Jointtrain+dev154.922s;finaltestsnotincluded. No full-corpus/speedparityclaim.

Independent512freshcases/familyx4draws,parent/finalautonomousexact:newpairs19.2383/
76.1230%;newnames15.6738/53.4180%;newquestions15.7227/50.8789%;oldheldlayouts3.7598/
11.8652%;newheldlayouts1.6602/4.6875%. Newpairgain95%CI[52.29,61.47]pp.
Finalmaincorrect/changed/shuffled/wrong/blocked/noquestion76.123/78.711/76.563/
1.172/0/8.643%. Changedfactfollowed94.612%amongpreviouslycorrectdraws.
Mainselection86.133%,plancompliance91.504%,oracle95.703%. Newlayoutoracle9.570%
andoldheldlayout28.223%,soremainingformatgeneralizationandqueryreadingweak.
ExampleOliver:river+house->rain+caronlyoldfactchanged,generatedanswerfollows.

Checks:matchedAdamsteps2000each,joint1000,sourcehashes,predictor/lasttensors,
originalDstateunchanged,Efreeze,exactparamcount,bankrecoding+actualDhookparity,
target-tablemutationinvariance,blocked-memoryinvariance,heldpair/reversalcoverage.
Truecodec8192/8192source(128records)and8192/8192targettokens(1024answers)exact;
finitechecks,notchatcodeclossless. GPUpreflightzerooptimizerupdates,batch64full
features1.925GiBallocatedEXCLUDINGbank/Adam;gradientsreachpredictor,Dnograds.
Originalv4/growing4595/codec/allparents/v4manifestunchanged. Sourcearchives/audits
savedbothruns. ResumeCLIwithoptimizer/RNG/horizon/sourceguards;fullinterrupt
replayequivalencenotseparatelytested. No new long-v4run. Allsessions79685/52847/
60623/81190finished;processinspectionnofivepilotworkers. Dfree10.51GiB.

8predictorSHAb6d03e9d079f399e6158b54348243fe93e8bbc19733a5b3638c055308018ad8c;
64SHA64e03eca4e387139fefae2546c04bf6eb4c43fcc3efae809088f5f23b1e8f74a;
jointSHAea66d5dcf2f7f41165364e78124d717beb8e667475630221c80319463592c2ee.
ConstructiontrainerSHA887188eedb4a50d60d5f8bddaa8492cf665b537ec44023d7d2e579d345cf8841;
jointtrainerSHAd4cf58ac05b5f16640fcc38c364b046d7a43ffc9540c19ef9ae6eac8ca58f75c.
Reportdocs/position_vector_constructor_results.md,README/Step3updated.

ScopeNOTcomplete:fixed8tokenanswer/32-wordtoy,notgeneralLM;heldlayoutsweak,
variableoutputlength/EOS,growing8->16->...64replacement,self-fedpredictedcodes,
arbitraryprobabilisticlanguage,DBandmatchedrawquality/speedunverified. MustNOT
redefinesuccessaroundmoretoytemplates. Nextworkshouldtesttransfertolessconstrained
text/variablelengthanduser'sactualfeedbackcontract,oraddressremainingquery/format
representationwithaclearbroadertransfergate. Keepdirectoutput/frozencodec/budget.

# Varied-record constructor pilot COMPLETE, failed; broader goal ACTIVE (2026-10-13)

Lastgoalturnwasprogress,nogenuineblocker. Required9fea8f41attachmentreadagain.
Currentuserkeepsdirectvectoroutputandrequeststestprobabilisticgeneration.
tools/train_slot_vector_constructor.py matchedraw/eight-slotconstruction3kupdates
eachfromfeature-vector-planner/features(parent11klineage=>each14k). Samebatch64,
dataorder,plans,LR1e-4to1e-5,MSE+.02CE+.25sourceCE+.25planCE,23,209,216trainable,
originalE/Dfrozen. BothreaderusespooledDfeatures;rawconstructsfromselectedcodec
vector,slotsuses8orderedmeansof8Dhiddenpositions. NoinputtokenIDs;direct8token
codecvectoroutput. LearnedprobabilisticplanstillONLYtwoanswerorders,nogeneralprior.

16trainrecordlayouts,bothfieldorders,prefixoffsets0/4/8/12,40trainnames,768trainpairs,
240heldnonidenticalpairs,heldreversalsdisjoint. Bank20,480records,189251941bytes,
prep7.734s;SHA8ee624ad7420df184450071640acd0eef6f06116e2763625e3c1434b68ad70c3.
Training+dev462.765sraw/457.078sslots(singletrial,finaltestsexcluded).
Outputdump/slot-vector-constructor-20261013. Independentverify512cases/familyx4draws:
exactraw/slotsnewpairs.48828/.58594%;newnames.19531/.29297;newquestions.19531/.43945;
all3heldrecordlayoutfamilies0/0. Paired95%CInewpairsdelta[-.6836,.8789]pp,nobenefit.
Newpairselection91.0156/92.96875%;goldsource+planexact.3906/.78125%,heldlayouts0.
Variable-word-onlyaccuracyequalsfullanswer;notpunctuationissue. Truecodec384records
24576/24576tokensexact(botharmschecksamerows,doNOTdoublecount).

tools/audit_slot_vector_constructor.py:trainingbank256casesexact0/.4883%,oracle
source+plan0/1.171875%;notonlyheldpairgeneralizationfailure. Foundtokenprefixjoin
artifactwasFile/forFile. Preservedlaunchedsource/data;read-only256pairedcases:
knownlayoutsoriginalexact.1953/.3906%,spacejoin0/.3906%,no_prefix1.3672/1.1719%.
Heldlayoutsall0. Codec4096/4096tokensperboundarycondition. Inferencerepairdoesnot
rescue;doesNOTprovethatcleanTRAININGwouldhavenoeffect. NextdataMUSTcleanjoins.
Donotsilentlypatcharchivedtrainerandbreakresume/sourceidentity.

AllAdamsteps3000,model/predictorfiletensorsmatch,source/protectedparents/v4/growing
4595/encoder/codec/v4manifestunchanged. FrozenDoriginaltensors,Efreeze,parambudget,
featurehookparity,targettablemutationinvariance,blockedmemoryinvariancechecked.
Preflightgradientchecks/0optimizerupdates,sourcebankprecomputeavoidrepeatencodes.
RawpredictorSHA84990dcd0d588e278e7b0ecbbf6b6b17d6a0a197aa87f3f41b054248fc75e409;
slotsSHA114267a9cfbcb123582238aa081fffbefb8b387c175da928ce0cb480d15d6935;
trainerSHAb6c6098e1689c9de6b50de3e151b8cacb449ec1375e33e998186f4acc3c997ce.
tools/verify_slot_vector_constructor.py savesverification+pairedstats. Report
docs/slot_vector_constructor_results.md. Sessions2498/32478/71393ALLfinished.
No active researchworkerfromthisturn.13.3GiBDfreebeforefinalevidence,no cleanup.
PilotresumeCLIrestoreslastperiodicoptimizer/RNG/seed/horizon;partialdiscarded,
fullinterrupt/replaynotseparatelytested. No long/fullcorpusrestartrecommended.
Nextboundedhypothesis:all64frozenDpositionstatesforselectedrecordconstructor,
cleanjoins,firstgold-sourcefit/generalization. Eight-groupaveragingnotestablished
useful;notproofalllatentgenerationimpossible. GeneralLM,growingfeedback,EOS,DBopen.

# Matched frozen-feature reader pilot COMPLETE; broader goal ACTIVE (2026-10-13)

Lastgoalturnwasprogress,nogenuineblocker. Required9fea8f41attachmentreadagain.
tools/train_feature_vector_planner.py runsraw/featuresmatched3kcontinuations from
shared-vector-plannerfinal4k(parent8ktotallineage=>each11k). Samebatch64,dataorder,
MSE+.02CE+.25sourceCE+.25planCE,cosine1e-4to1e-5,23,209,216params,frozenoriginalE/D.
Featuresarmreadsmean-pooledfrozenDnormalizedstates(one256Dvectorperrecord/query),
butconstructsnewanswerfromORIGINALselectedrecordcode. No predictorinputtokenIDs.
No additionaltrainableparams. Extra24trainquestions=>40total,extra4orders/kind,
same768train/240heldpairs;new8heldquestions/3heldorderphrases. Questionpreflight
144768combinationsmax48tokens;frozenfeaturemanualpathmatchesdecoderforwardhook.

Outputdump/feature-vector-planner-20261013/raw,features. Both3kcomplete;train+dev
455.765/499.906seconds(raw/feature). Independenttools/verify_feature_vector_planner.py:
512cases/familyx4draws exactraw/feature:knownwording67.7246/83.1055%;newnames55.5176/
57.2754;oldheldquestions25.1465/38.0859;new8questions18.75/47.7051;oldheldorders32.7148/
48.1445;new3orders35.4980/34.7656. Pairedbootstrapfeature-minus-raw95%CIknownwording
[10.74,20.17]pp,newquestions[23.88,33.84];newnames/newordersCIincludeszero.
Featureknownwordingcorrect/changed/wrong/blocked/noquestion83.11/81.98/.34/0/8.06%.
Featureselection90.625%;goldsource+requestedplan100%onknownrecordformats.

Crucialnewfailure:twoUNSEENrecordlayouts(256caseseach),includingsecondfieldbefore
first, BOTHarms0%exact evenwithgoldsourceANDplan. Featureselection39.45/29.30%
vsraw6.25/12.11%butconstructorfails. tools/audit_feature_vector_planner.py verifies
frozenDexact32768/32768tokens,512/512recordsonsamelayouts. Infoispresent;constructor
isformat-dependent. NextshouldtestfrozenDfeaturesinCONSTRUCTIONplusdiverserecord
formats/fieldorder,possiblyposition-preservingfeatures. No generalLM,growingfeedback,
EOS,DBorchatbotcompletion. No newlong/fullcorpusrunlaunched,productionunchanged.

Independentchecks:frozenoriginalD/E,paramcount,targettablemutationinvariance,
blockedmemoryinvariance,decoderfeaturesviaactualforwardhook,Adamsteps3000each,
sourceSHA/protectedparents/originalv4/growing4595/codec/v4manifestunchanged.
RawpredictorSHA4f588a7699da6de79c0c489161353bb69e649af91ee09d9e0f79df84b88a3731;
featureSHAac45fefc0adb72f4f5355792bafcf686385801317671aa057be68310a990360f.
Reportdocs/feature_vector_planner_results.md,allsourcearchives/results/auditretained.
PilotCLIhasperiodicweights/optimizercheckpointsbutNOresumeCLI;notproductiontrainer.
Sessions52751/33681bothfinished;no researchworkerleftfromthisturn.

# Shared probabilistic integration COMPLETE; broader goal ACTIVE (2026-10-13)

Lastgoalturnwasprogress,noblocker. Requiredattachment9fea8f41...readagain.
tools/train_shared_vector_planner.py combinesselection/learnedplanprior/construction
inSAME23,209,216params,two passes,frozenoriginalE/D,noinputtokens. Readphase3uses
lastattentionQKforsourceCEandfirst2existingheadoutputsforplanCE. Constructphase1/2
getsselectedoriginalrecord+fixedactivevector,notquery. Inferenceargmaxsourceand
sampledorder;goldsource/ordertrainingonly. Parent sampled-plan-vectorcheckpoint
(4ktotallineage);4knewupdates,batch64,cosine1e-4to1e-5,MSE+.02CE+.25sourceCE+
.25planCE. 8records,768train/240heldpairs,40train18testnames,16train4testquestions,
8prefixes/3suffixes/4orderphrasesperkind;3orderphrasesheldout. No v4training.

Outputdump/shared-vector-planner-20261013;4kcompletein597s incldev. Independent
tools/verify_shared_vector_planner.py reload512cases/familyx4draws:exactnewpairs
correct44.7266/changed45.5566/shuffled46.7285/wrong.3906/blocked0/noquery7.5684%.
Newnames37.5488%,newbasewording14.3555%,neworderwording17.5293%. Familiarselection
45.703%,sampledplancompliance97.900%;oracle source+predictedargmaxplan98.242%,
oracle source+requestedplan100%all4families. All1024trueanswercodesexact;512questions
16238/16238tokensreconstruct. Informationrecoverabletestedinputs,butpoorqueryreading.
Bothvalidorderssampledwhenallowed;learnedcanonicalprior.514. Pairedcasebootstrap
correct-minus-wrong95%CI[40.19,48.63]percentagepoints. Changedfactfollowed86.24%
amongpreviouslycorrectdraws. NOTgenerallanguage/chat/EOS/growingfeedback/DBproof.
Nextneedsrobustqueryreading;noautomaticfullcorpusresumeornewunboundedpilot.

Preflightdecodernew_onlylength8vsfull8logitsmax4.58e-5,argmaxidentical,gradmax2.43e-8.
VerifierchecksfrozenDoriginaltensors,Efreeze,paramcount,targettablemutationinvariance,
blockedmemoryinvariance,heldpair/reversaldisjoint,protectedhashes. AllAdamsteps4000,
sourcehashesmatch. LastSHA d369defaffb02956c9f123912e7eb04be93cd08b2ee88f68aaa66a7e5b18b11c;
predictorSHA823bc5cb8125c0bce189a673018e945757ecda18f99039a007c933b0f67d43e8.
Step0/every500atomiccheckpoint,fulloptimizer/RNG/seeds;source/inputscheckedresume.
No fullinterruption/replaylifecycleverificationclaim. Originalv4/growing4595/codec/
parentunchanged;no productionmodelcodechange. Sessions50917/60443bothfinished.
Reportdocs/shared_vector_planner_results.md;alloutputs/auditsretained.

# Composition curriculum/two-pass progress (2026-10-13)

Previousgoalturnwasprogress. Requiredattachment9fea8f41...readagain. No live user
trainingfound; sharedworkerlockused. No productioncode/corpus/originalweightchanges.
Newtools/test_vector_composition_curriculum.py:32words,1024two-word answers,
heldout256pairs,train96sparseor768expanded,bothcoverallwordsinbothslots.2layouts,
40train18testnames,16train4testquestions. FrozenE/D,exact23,209,216trainableparams.
Matchedfresh2kupdatesAdam3e-4,batch64,200MSEwarmupthenMSE+.02CE;eacharmtraining-only
normalization. Oraclecorrectrecordgiven: independent512newpairsexact19.73%sparse/
100%expanded. Expandedchanged100%,wrong.78%,blocked0,newnames100%,newwording99.61%.
ThisiscompositionNOTautonomousretrieval. All1024trueanswercodesexact;independent
256source64-tokenrecords16384/16384tokensincluding512/512facttokenscorrect.
Expandedcontinued2kwith1,2,4,8records=>33.20%fresh8recordnewpairsexact.
tools/test_vector_attention_curriculum.py adds.25source-indexCEtosamelastattention,
sameoracleparent/data/schedule;actualanswers18.36%butselection90.43% (freshpaired).
Read-onlytools/diagnose_vector_selection.pyseparatesselectorvsconstructor.

tools/test_two_pass_vector_memory.py reusesSAMEweights:phase2selectstoredcode,
phase1constructnew8tokenanswervectorfromselectedoriginalcode+query, frozenDdecode.
No tokensornewparameters;goldindextrainingonly;inferargmax. 2kupdates1e-4fromaux
checkpoint (6ktotallineagevs4kplain;notmatchedtotalbudget). Freshreload512each:
newpairsexactcorrect91.60/changed91.21/shuffled92.58/wrong.39/blocked0/noquery9.77%.
Newnames72.07%;newwording4.30% (majorfailure). Sameweightscomposegold97.07%familiar,
7.81%newwording:queryOODhurtsbothstages. Localbatch32onepass21ms/twopass41ms,
encodingexcluded. tools/verify_two_pass_vector_memory.py verifiesfrozenDmatches,
count,targettablemutationinvariance,blockedmemoryinvariance,originalhashes.
Evidence dump/vector-composition-curriculum-20261013,vector-attention-curriculum-
20261013,two-pass-vector-memory-20261013 inclsourcearchives/weights/percaseoutputs.
Reportdocs/vector_curriculum_results.md. GoalremainsACTIVE;notgeneralchat,EOS,
growingfeedback,multirecordreasoningorDBproof. Nextneedsquerygeneralization.

Extra controlledprobabilisticpilots COMPLETE. tools/test_ambiguous_vector_generation.py
andtest_sampled_plan_vector.py; outputs dump/ambiguous-vector-generation-20261013
anddump/sampled-plan-vector-20261013. Correctsingle64-tokenrecordprovided; either
(a,b)or(b,a)8tokenanswervalid. Same768trainpairs/heldout240nonidenticalpairs,
expanderoracleinit,matched2kupdatesMSEthen+.02CEafter200. Threevariants:onepass
ambiguousregression,7passdenoising,onepassconditionedonindependentlysampledorderbit
viaexistingphase1/2embeddings. PlanbitcontainsNOvalues. Hand-definedmode/prior,
notlearnedarbitrarylanguageplanning. Allsame23.209M/frozenE/D/vectoroutput.
Freshreload256records*8draws:valid0/10.84/100%;bothorders0/3.91/99.61%respectively.
Sampledplanwrongmemory.39%,blocked0%,planfollow100%. Separate512test100%valid,
newnames256100%. Samplingbatch32time21.5/135.8/20.5ms,encodingexcluded.
tools/verify_stochastic_vector_controls.pychecksnewseeds,targettablemutation,
blockedinvariance,frozenD/protectedhashes. Alljobsfinishedsessions32290/74722/6209.
Two-passretrieval andsampled-planmodelareSEPARATEcheckpoints,NOTcombined23Msystem.
Nextintegrationmustshareparamsandlearna contextualplanprior;alsofixwording. Broader
goalstillACTIVE,notfull-language/growingfeedback/DB/chatbotcompletion.

# Direct output and probabilistic pilot COMPLETE; broader goal unresolved (2026-10-13)

User explicitly chose direct vector output and probabilistic generation, not token
output. No production code/model/corpus changes or long-run resume this turn.
tools/test_frozen_memory_generation.py: two matched1200update fresh23,209,216param
pilots rawcodes/pooledDfeatures; frozenE/D, eight64token records -> new8token answer.
Train24 color/noun pairs,heldout8; all32 trueanswercodes decode100%. Raw newhistory
58.40%,heldoutcombos1.17%,newnames42.77%,newwording9.96%; pooled0%exactallfamilies.
Oracle-record variant removes retrieval usinggoldrecord: raw familiar/newnames/
newwording100%,butheldoutcombos1.56%; pooledheldout0%. NOTautonomousretrievalproof.
Evidence dump/frozen-memory-generation-20261013 and -oracle-20261013;512cases/family.

tools/test_probabilistic_vector_generation.py: matchedfreshonepassvs7leveldenoising,
same23.209M,2kupdates,batch64,Adam3e-4,MSEthen+.02CEafter200updates.2048train/
256devdialoguesfirst8assistanttokensonly;prior<=32vectors. Realtestfresh256:token
accuracycorrect7.71/8.40%,exact6/3out256,wrong.83/1.51%,blocked0/.15%. Samplesgarbled.
Onepassmemorizes256checkedtrain100%exact;denoisetrain5.13%tokens/0exact.7passsample
batch32time129.9msvs22.5ms(~5.8x),excludesencoding. Noquality/impossibilityclaim.
Denoising fourdraws3.87unique/testexample;diversitynotquality. Truecodedev2048/2048,
fresh test2047/2048(99.95%);do notclaimcodecperfect.
Teacherlow-noise.05~49.8%tokenaccvsnoise1~5.66%,notgenerationresult.
Evidence dump/probabilistic-vector-20261013 inclpreparedtensors,weights,sourcearchives,
results.json,verification.json. CPU samplerchecks+GPUreloadtargetindependence,
blockedinvariance,reencodeddataandlabels,freshsplit checks pass. Protectedoriginalv4,
paused4595,encoder/fullcodec hashes unchanged. Sixsamplerchecks in
tools/verify_probabilistic_sampler.py;freshreload tools/verify_probabilistic_vector_generation.py.
Docs/probabilistic_vector_results.md. No longtrainingcommandrecommended. Broader
goalstillACTIVE; do notmarkcompletebasedonmemorization. Nextcontrolledworkneeds
unseencombinationgenerationbeforefullcorpus; userconstraintdirectvectoroutputpersists.

# Active goal: frozen codec learning investigation (2026-10-13)

User authorized alternatives after stopping growing23M at4595. Training process
confirmed gone and worker lock acquired. PausedlastSHA80822ed7c4156c2f36c94807704783a778d4bf77fd01afbad0d0a74b850930f3 unchanged.
Read required attachment9fea8f41.../pasted-text-1.txt. Three matched300update
continuations on512train/128dev full64assistantblocks: MSE-only devCE47.14,
CE-only16.65,MSE+.01CE21.88; initial14.94. All<1%newtokenacc,0exact. Truecode
oracle99.902% on2048tokens;30%interpolationtowardprediction24.02%. Notuniversal
impossibility. Reports dump/frozen-loss-diagnosis-20261013; tools/diagnose_frozen_vector_losses.py.

Three fresh vector-only pointer pilots reuse23,209,216paramModel, frozenE/D.
CE chooses one of8 storedcodes, copies selectedoriginalcode, decoderrecoversrecord.
Notfuture-vector generation. Short600updatepilot98.44%newhistory/96.09%newnames/
38.67%newwording. Full64records+20questiontemplates1000updates94.92/71.68/16.60.
Matched full64pilot using meanpooled frozenD hiddenfeatures97.27/76.17/47.66.
Controls wrong/blocked~chance12.5%; changedfactsfollow; exactrecordmatchesselection.
Independent fresh256reload verifies full64feature reader97.27/72.66/46.88;
rawfeaturecosine47.66/53.13/38.67,queryremoved/wrong~chance. FrozenDhookverification
andblockedmemoryinvariancepassed. All outputs/evidence/sourcezips in
frozen-memory-router-20261013,frozen-memory-router-expanded-20261013,
frozen-memory-reader-20261013. Tools test_frozen_memory_router.py (--expanded,
--read-features) and verify_frozen_memory_router.py (--run).

Originalv4/codec/encoder/datasetmanifest hashes unchanged. No productionchanges or
longrun resumed. Scope still open: reader wording/layout generalization and new
text/vector generation conditioned on memories. Do NOT mark activegoalcomplete
based on narrow retrieval. Report docs/frozen_codec_memory_investigation.md.

# Growing-vector packing speed update (2026-10-13)

Live research worker left untouched; user must pause/resume for new code. CPU
metadata bulk transfers5 instead of thousands of GPU writes; decoder only executes
queries containing scored targets. Same predictor, masks, MSE/CE, data/schedule.
First32dialogs1790real/8576paddedqueries. Before sources/data/model archived under
dump/growing-vector-packing-speed. Exact old->new source hashes allowed through
packing_migration.json; original23M checkpoint sources verified. CPU input parity
bitwise; scored decoder logits/loss/allgradient tolerance test added. No new GPU
benchmark or research updates launched, so no measured end-to-end speedup claim.
Floating-point batching may differ; optimizer/state resume preserved.

# Growing-vector exact reference budget (2026-10-13)

User requested same23M trainable capacity. Current predictor EXACT23,209,216:
8layers,width512,heads8; four BranchFFNs1774hidden,four1773hidden, all active.
Original encoder/decoder frozen, same8→64feedback,data,objectives and schedule.
7CPU tests pass incl count,gradients,causality,resume. Zero-update GPUmicro16
first32dialogs2.375seconds4.412GiBreserved incl moment-sizedbuffers; excludes
optimizersteps/eval/save and notworstcase. Evidence growing-vector-training-verification-16.json.
No longtraining launched. Fresh command unchanged; old18.62Mcheckpoints cannot
resume into thisarchitecture. Earlier18.62Mcounts/benchmarks below are historical.

# Growing-vector production trainer READY (2026-10-13)

Current src/models/growing_vector: 18,624,768 trainable predictor parameters;
original frozen encoder/decoder. Single active code replaced 8→16→…→64; direct
predicted feedback in generation, append at64. Train teacher-forces true prefix
codes. CE scores only new8 tokens once; normalized MSE full growing prefix.
Short final targets EOS-filled to next8 only for codec, fill excluded from CE.
75 MSE-only updates then75 CE ramp. Phase-specific normalization from256train.
Whole dialogues from unchanged v4:483862train/4852dev,2epochs batch32=30242updates.
Same-folder Adam/RNG/cursor/source/data resume; microbatch may change. LRU10000.
7 CPU tests pass incl causality,frozen gradients,label coverage,resume and partial
last batches. GPU zero-update first32dialogs: micro16 2.078sec,4.35GiBreserved;
micro32 2.141sec,5.89GiB. Moment buffers included, optimizer/eval/save excluded.
Not whole-run guarantee. No long research training launched. User gets command.
Original v4 checkpoint, encoder/fullcodec and data manifest hashes unchanged.
Deleted superseded block_vector sources (byte-verified archive),22 experimental
checkpoint/sidecar files,2135019768bytes; disposable test fixtures552792562bytes.
Manifests dump/growing-vector-cleanup-20261013.json and growing-vector-fixture-cleanup.json.
Old vector/block-vector runs cannot resume. Token/raw references retained. Previous
entries claiming those experimental weights remain are superseded. README has commands.

# Frozen decoder / growing-vector bounded diagnostics COMPLETE (2026-10-13)

User authorized originalfrozenD MSE+CE realtextpilot,then clarified ONE active
vector replaced8->16->24->...->64,not appendedpartials. Three300update pilots
completed; no productioncode/checkpoint changes orlongrun. Existingblockvector
paused7096beforetests andstillpaused;SHAfc7f73b7894cc01caaac791ee7cb8c3c4b53203fb67f00758633527372be8e18
unchanged,asoriginalencoder/fullcodec/v4weights. Sharedworkerlockused.
Tools test_frozen_vector_prediction.py/test_growing_vector.py; protocols/results/
sourcearchives dump/frozen-vector-real-300 anddump/growing-vector-real-300.

512train/64devdistinctdocs,onefull64assistantspan/doc,last256knowncontexttokens,
seed47,batch32micro8,AdamW3e-4,75MSEwarmup+75CEramp+150jointupdatesperarm.
Originaldecoder frozen but CEgradflow; predictor18,620,672params; growing adds
4096phaseparams. Codec genuine8prefix511/512tokens63/64exact;16..64all64exact
onthissampleONLY. Independent64devCE18.2339accuracy.46%first8accuracy2.15%,
MSE1.3352vsconstantmean1.0143;independent8CE14.1253accuracy3.32%,MSE1.5763vs
mean1.0327. Neither0/64exact improves toanyexact. DifferenthorizonsnotmatchedLMCE.
Independent8 exploratoryrollout exceeds trainedhistorylength;notconclusive.

Growinghistory64chunks+ONEactivevector,phase8..64; targetfullprefixcode,CEallprefix
positions. Trainteacherforced priorcode; freerollout ownpredictedcodereplacesit.
DevCE13.5819MSE.7628,allprefixaccuracy7.19%butNEW8accuracy1.95%.16contextfixed64
rolloutnewaccuracy.59%,previouslyemittedpositionrevisionrate80.44%overrepeated
phasechecks. EOSstoppingnotused fordiagnosticfixedhorizon. No knownfuturetokens
infreefeedback. Decoderstateidentity andprotectedhashesunchanged. Growing syntax
errorfixedbeforeexecution;onlyoneoptimizer run. No encoderretrainingneedshown;
predictor/feedbackquality NOT solved. No impossibility/sizelimit/longrunrecommend.
Nextproposedonly:committedprefixconsistency+predictedfeedbacktraining,notjust
duplicatingcurrentoldtokenCE. Report docs/frozen_growing_vector_diagnostic.md.

# Block-vector CE + MSE experiment READY (2026-10-13)

Latest user requested no token inputs in predictor/decoder,64-token vector targets,
CE+MSE,efficient encoding,tests then training command. Implemented separate
src/models/block_vector:23,192,065trainable (predictor18,620,672/decoder4,571,393),
frozenencoder9,426,432. Predictor fresh; decoder independent pretrained codec copy
plus past-state cross-attention. Decoder outputs64ordered slots,EOS2eligible;
host stops firstEOS. No within-block token AR/raw input. Actual emitted text gets
encoded at block boundaries for feedback; predicted vectors not blindly appended.
No DB. Full corpus run NOT launched; user triggers model-local Python CLI.

Same v4binaries/targets/seed47/batch32/twoepochs48762steps. Assistant-aligned <=64
target spans preserve every label and prevent hiding the question inside targets.
MSE on full64assistant blocks only; shorter tails CEonly. Norms first256train
segments. Causal completed nodes, queries see only prior completed nodes+self;
target codes loss-only. CE+1*normalizedMSE separately normalized across effective
batch. 10000-entry CPU LRU ~9.77MiBvalues+keys; batch256missencoding,no fullcache,
no per-token prefix encoding. Unique data has fewhits;speed mainlyfewerencodes.
Same-folder resume restores Adam/RNG/cursor/schedule; microbatch maychange.
Defaultmicro8; measuredmicro16 recommended. Skipinitialvalidation/subset256;
finalfull7831. Commands and caveats src/models/block_vector/README.md.

7CPUtests pass including causal/target/gradients, EOS, count, exact same-micro
resume,epochcoverage and objective guards. Regrouped decoder Q/K Adam values
can differ5.64e-5on syntheticdegenerateencoder;otherparams<1.2e-7;allgradient
parity separately passes. Real GPU3effectivebatches(nooptimizerupdates),Adam-sized
buffers:micro8 1.88updates/s3.53GiB;micro16 3.09updates/s4.81GiBreserved.
Excludesoptimizersteps/eval/save;notwhole-run guarantee. Futuretargetmutation
logitsmax0;realencoderparallel/sequentialwithin5.35e-5tolerance. Initial codec
oracle8084/8192tokens21/128exact:doNOTclaimchatcodec lossless.

Two matched300update syntheticpilots were run, then corrected/rerun after finding
a color-dependent filler-alignment shortcut. First evidence retained under
dump/block-vector-pilot-300; not retrievalproof. Correctedv2 aligns factblock64
and asserts remaining704tokensidenticalacrosscolors. Corrected evidence in
dump/block-vector-pilot-300-v2. CE andCE+MSE both fit32/32colors;fresh16cases
correct/wrong/blocked16/0/4;changedONLYfirstoldfactvector16/16replacementfollowed.
Freshcolor score NOT fullanswer:only8/16exactblocks;unseennamecopyfails.
Fresh MSE1.29996CE-only vs.00189556joint. Does not establish MSElanguagewin.
Broad32devCE34.42/37.27aftertoy-onlyfit:diagnosticweights NOT chatbot/longruninit.
No data/modelsize/benchmark superiority claim. Existingencoder/fullcodec/originalv4
andoldvectorcheckpointSHAunchanged. No optimizer touches those models.

User-requested src/models/vector_memory andits2tests/2helpersremoved after
byte-verified archive dump/retired-vector-memory-source-20261013.zip; manifest
dump/vector-memory-retirement-20261013.json. All oldweights preserved; restoring
archive needed for oldvectorresume. Disposablesyntheticlifecycle artifacts from
this task removed (1,947,978,312bytes), manifest block-vector-lifecycle-cleanup.json.
Originaltokenmemory/rawmodules unchanged. Step3/README updated. Historical below.

# Vector-only input experiment PREPARED (2026-10-13)

On-demand speed follow-up: user chose faster on-demand after cache-space question
(Dfree19GiB vs full FP32prefixcache~316GiB). No fullcache/preprocessing/training
launched. Vector data nowbulk packs/transfers/scatters, lengthbuckets8tokens,
encoderbatch256. Trainer omits onlyunscored prefixencodings (neverattendedbyothers);
all completed/userhistoryblocks preserved. Causalmodel/data/CE/paramsunchanged.
8CPUtests passed inclscoredlogit/loss/allgradientparity whenomittingunscoredprefixes.
GPUzerooptimizerbench: firstshuffledmicro16 2.6445->.6417s, longhistory3.6095->1.3998s,
~4.34GiBreservedwithoutAdammoments. Actualscoredlogitmax2.9e-6; allparamgradsmatch
tolerance. Singlepassnotwholetrainingguarantee. Evidence dump/vector-encoding-speed
containsold_data.py, beforesources, benchmark.json. Exactmigration frombothknown
oldsource versions; latestdurablecheckpoint114hash/sourceverified. Userresumes
samefolder micro16 --valid-segments256 --skip-initial-validation. No automaticrun.

Follow-up validation-speed fix: user'smicro32run failed10.5GiBguard after6updates;
verifieddurablelast.pt isstep0. Trainer adds --valid-segments (fresh256,0full;
oldresume inheritsfull unlessexplicit), --skip-initial-validation, progress and
training-onlyETA excludingvalidation. Finalalwaysfull; subsetmetrics labeled,
changingselectionarchivesoldbest/resetsscore. Exactbefore/aftermigrationJSON
allows onlythisknowntrainerchange; resumedsourcearchivepreserved. SevenCPUtests
pass; no researchtraining launched. Recommendresumeexistingrun micro16,
--valid-segments256 --skip-initial-validation. Originaldata/model/weightsuntouched.

User approved new vector-only consumer and asked for training command; no research
training launched. Separate src/models/vector_memory uses unchanged v4 corpus,
assistant targets, seed47, batch32, 2epochs48762updates and LR schedule. Exactly
23,209,216trainableparams, same frozen encoder. Completed64token blocks + one
re-encoded current1..63token prefix; empty-block learned query. Outputs ordinary
tokens with CE, no decoder/DB/raw-token consumer inputs. Prefix training graph
sees prior completed blocks+self only, never other prefix versions/future blocks.
Parallel/sequential logits agree; representation/compression timing differs from
original (up to39completed vectors), source text/targets matched. Same-folder
resume restores Adam/RNG/cursor; only microbatch maychange. Defaultmicro16.
6new CPU tests +2original lifecycle regression tests pass. Real GPU zero-update
micro16 probe2048history/433..508targets:4.00GiBreserved, excludesoptimizer moments
andstep temporaries; no speed guarantee. Evidence dump/vector-memory-verification.
8different text slices xlengths1..64:16544/16640rawdecoderargmaxcorrect,416/512exact.
Errors82specialtokens+14ordinary atlengths1..3;length1only1/8exact. Do NOTclaim
shortprefix codec lossless or encoderlosttokens: reconstructiondecoder can fail.
Data labels/masks match original, realrollout maxerror1.8e-6. Protectedv4/encoder
hashesunchanged and originalv4source compatibility passed. Existing memory tests
remain complete as below; no need rerun unchanged model. Commands/limitations in
src/models/vector_memory/README.md, Step3linked. No newtrainedcheckpoint yet.

# V4 data/loss and raw/vector grounding diagnosis COMPLETE (2026-10-12)

User requested samefactrecentvsoldvectors,inspectloss/data,no training/sizeincrease.
Samefinal48762SHAe17b9a8469b5a8c996cc64ff0677eb1c8f52410af52c1c179fd317263b4d9e2d.
8singlecolorfacts(Amina/Rina x4colors):unambiguouscorrect8/8shortrecent,8/8long
filler-before-factrecent,0/8compressedfiller-after-fact,1/8blocked. Greedy64;
correctrequiresonlyexpectedcoloramong8recognizedcolors,notfirstcolormatch.
32freshthree-recordnotebooktasks crossedfamiliar/heldoutnamesandwording:
fullvaluesubstring1/32short,1/32longrecent,0/32compressed/blocked;familiar/familiar
also0/8allarms. Bindingdifficultybeyondcompression;notjustnovelnames/templates.
Codec oracle64chunks:4064/4096tokens,32/64exact;all32errorsEOS2OUTSIDEfacts.
All2202fact-bodytokenscorrect,encoderexport/fullvectorsbitwiseequal. DoNOTclaim
entirechatcodeclossless;EOSboundaryrepresentationisalimitation,notlostfactproof.
Lossinspectionshift/attentionvisibility/globalbatchnormalizationcorrect;no bugfound.
128trainmemoryallCE.1582/93.22%,locationBPE.8464/63.35%,other.0189/99.27%.
128testallCE.2244/91.38%,location.8175/63.94%,other.1109/96.64%.
Memory0.6997%scoredcorpustokens;variablelocations693/4116sampletokens =>~.12%
alltargetsestimated,notgradientshare. Narrow12trainingnames,notebookrelation,
4questiontemplates;~25.4%reasoningtargets;TinyStories4genericprompts.
Proposedonly:shortverifiablegroundedchat,pairedrecent/compressedcurriculum1to3facts,
diversenames/wording/values/distances,separatenormalizedretrievalCEe.g10%initial
experiment,value-focusedtargets,languagepreservation,generated-answerselection.
NoDB/codearchitecture/dataset/weights/trainingchanges.No65percentceilingclaim.
Evidence dump/unified-v4-grounding-diagnosis, docs/v4_grounding_diagnosis.md.
Scripts diagnose_v4_grounding.py,diagnose_v4_single_fact.py(--long-control),
inspect_v4_codec_errors.py in tools. Allweights/encoderhashesunchanged.

# V4 final checkpoint read-only evaluation COMPLETE (2026-10-12)

User requested latestcheckpoint correct/wrong/blocked memory and chat examples.
Final48762 loaded with source/hash checks;lastSHA
e17b9a8469b5a8c996cc64ff0677eb1c8f52410af52c1c179fd317263b4d9e2d.
SaveddevCE1.80471036,accuracy61.01325%;notfull-devreplayedthisturn.
tools/evaluate_unified_memory.py:seed47014,200non-memorytestdialogueswithhistory,
432segments,121217targets/71562history. HistoryCEcorrect1.99158975/wrong2.11925120/
blocked2.13766051,accuracy58.18/56.37/56.09%. Paired5000docbootstrapblocked-minus-
correctCI[.13547,.15722];first64CE2.39149vsblocked2.80566. Genuine context benefit.
128testmemoryepisodesgreedy:exactcorrect/wrong/blocked13/13/11;103presentfacts
ALL0exact,25missing13/13/11. Correctlocationsubstringsalso0/103allconditions.
32changedoldvalues0exact/0newvaluesubstrings. Fresh model has not reproduced old
specialist retrieval; do not claim universal compression failure or lost oldweights.
9fixedchatoutputs:recentpurplefavoritefollowupworks;greetingirrelevant,ice loops,
three tipsrepeatedirrelevant,bicyclebindingwrong,3+2wrongbothmodes,storyweak/truncated.
No usablechatbotclaim. Reasoningonloops. Greedy qualitative,not samplingbenchmark.
tools/verify_unified_evaluation.py checked32prefixlogits exact/argmax32of32 and
formatinsensitivelocations. Initialverificationhelperdouble-addedmode,corrected
onlyharness;mainretrievalusedexactstoredprefixes,chatrawpromptsunchanged.
Weights/encoderhashesunchanged,nogradients,blockedmemoryinvariance/emptycontrols
pass,testfileshashverified. Nooptimizer/training/source-modelchanges.
Evidence dump/unified-v4-final-evaluation; report docs/unified_v4_final_evaluation.md.
No65percentceilingevidence. Nextfocusgeneratedfactualbinding/quality,notblindbudget
extensionorDB. Olderdataremovedasbelow;thisevaluationusedretainedv4only.

# Old dataset payload cleanup COMPLETE (2026-10-12)

User selected v4 as final dataset and requested deleting old data/freeing space.
Deleted98explicit old dataset/source-cache payload files,14.60GiB; manifest
dump/old-dataset-cleanup-20261012.json. Retained allv4files,allcheckpoints,frozen
encoder,tokenizers,metadata/cards/scripts and small historical validation sets.
Retired folders have PAYLOADS_REMOVED.json; old manifests remain historical.
Older training runs now require rebuilding their original datasets to resume;
never substitutev4onresume. V4training was already active (micro32); left running,
no source/model/optimizer changes. V4allfilehashes verified before and after.
Trainerdefaultstillv3forcheckpointsourcecompatibility; explicit --data v4 is
required for fresh training. V4reproduction still works by redownloading/rebuilding
v3 first; extension prefix audit requires rebuiltv3. Do not claimolddataavailable.

# Tool/reasoning expansion v4 READY (2026-10-12)

User requested more xLAM/Hermes and public distilled reasoning examples. Added
tools/download_chat_extensions.py,extend_unified_chat.py,audit_chat_extensions.py.
No trainer/model Python changes, parameter changes, environment installs or
research training. Same-folder old resumes untouched; use explicit --data forv4.
Two pinned SmolTalk2 shards downloaded (259925841bytes): Qwen3 multi-turn IF and
DeepSeek-V3-0324 smolagents traces, revisionfc6cc2103c066455aade5d7fbb346039ae36ca5e.
Existing xLAM/Hermes/Qwen shards scanned fully, dedup against unchangedv3prefix.
New adapter accepts mixed schema wrappers/JSONlines, preserves instructions,
associates plain observations only with one pending call, rejects invalid calls.

Prepared dump/data/unified-chat-v4:483862train/4852dev/4781test;42210additions.
331750037visible train tokens,180678552assistant targets,780186segments;
2epochs batch32=48762updates,partial26lastbatch. Reasoningon40667/off452828.
Additions:xLAM19222,Hermes14,QwenSystemChats12134,tables4281,multiturn2956,
DeepSeek3603. FinalxLAM59222/Hermes1602. Hermes remaining rows often malformed
or have mismatched results (4407), not silently repaired. DeepSeek3591are only
final_answer function calls; only12dialogues have25observed web-search outputs.
Do NOT call all3603multi-step tool workflows. Corpus573Toolturns (559/5/9).
All wholeexamples<=2560tokens; reasoning and answers never truncated.

Fullhash/mask/EOS/split/coverage/memory auditpassed1007retokenized; extensionaudit
confirms12basefileprefixesbyteidentical,counts/shares match.20CPUtests passed.
ManifestSHA f769f6421a88b416f1c9a9bc13558826c9e17ed10021a7b7d4bd5a9c57cae0c2.
Protectedoriginal20k/encoder/pausedv2hashesunchanged, receipt
dump/unified-chat-v4-protected-weights.json. Reports in dataset audit.json and
extension_audit.json. Docs docs/chat_dataset_v4.md has reproduction/fresh/resume
commands. Defaulttrainerstillv3; explicit --data dump/data/unified-chat-v4.
Micro16,batch32 recommended; no training launched. No universal quality or tool
execution claim; benchmarkdecontamination and future DB variant still pending.

# Unified chat corpus v3 READY; one memory-model entry (2026-10-12)

User requested free public distilled/persona/tool/reasoning data, one reproducible
corpus and consolidation; stopped v2 training. Paused checkpoint2168 retained,
SHA dc85b87c45f2d5201afdec409d9a6dc9267e544a2a75f90e2f882579da2c43f7.
Canonical src/models/token_memory/training.py now dispatches chat_data.training;
redundant token_memory_chat folder archived then removed. Same23,209,216params,
frozenencoder,512active+32oldvectors. Exact source migrations preserve pausedv2
and raw17k resumes; unknown model/source changes still rejected. Original20k
and encoder hashes unchanged. Legacy functions retained for historical tooling.

Prepared dump/data/unified-chat-v3 directly from pinned public source shards:
442401train/4496dev/4388test,287065313visible training tokens,153589862assistant
targets,691938segments. Twoepochs batch32=43248updates,final2segmentbatch/epoch.
451285totaldialogues:17693reasoningon,433592off. Sources:Smol-SmolTalk,everyday,
TinyStories,SystemChats,Tulu personas,OpenHermes,science,TableGPT,publicQwen3
reasoning,Hermes/xLAMtools,FineWebEdu,and40kmemoryepisodes. No paidteacher.
Memory0.8percentofscoredtargets; monitor retrieval,not a preservationclaim.
40kxLAM+1588Hermesexamples;533observedtoolturns,notallhaveexecutionresults.
6795unsupportedmixedschemawrappers excluded and disclosed; no executor/DB yet.

Public download scripts tools/download_chat_sources.py/download_unified_sources.py;
prepare_unified_chat.py;audit_chat_corpus.py --data dump/data/unified-chat-v3.
Frozen tokenizer fetched by downloader. Deterministic split/group/dedup,complete
examples<=2560,assistant-only CE,tool observations masked,Reasoning:on/off.
Fullauditpassed incl922retokenizations;two1500examplebuildsbyteidentical;17CPU
tests passed. ManifestSHA562bdb936f7c7f11a266416d40c3711c5824a1805d49b3266b095442915c5531.
Dependencyversions tokenizers0.23.2,pyarrow25.0.1,numpy2.5.3. No researchtraining.
Fresh default datasetv3; resumev2 preservesv2data. Recommendmicro16: realmicro32
v2peak10.48GiB,notprevioussynthetic8.09. tools/chat_unified.py --thinking on/off
prepared for futurev3checkpoint,displayscallswithout executing. Nochatquality,
reasoning,DBor1B-superiorityclaim. Need independent namedbenchmarkdecontamination
before claims; exactdedup is not semanticcleanliness. Docs/unified_chat_dataset.md
and src/models/token_memory/CHAT_TRAINING.md contain commands and limitations.

# Resumable chat baseline trainer READY (2026-10-12)

User asks command, will launch long run. New model-local entry
src/models/token_memory_chat/training.py reuses unchanged token_memory Model;
trainer src/chat_data/training.py. Fresh seed47,23,209,216params,frozenencoder,
FP32,assistant-only token-normalizedCE,noDB. Data chat-memory-v2,2epochs,batch32,
22264updates,partial19lastbatchperepoch,warmup200to3e-4cosineto3e-5.
Same-folder --resume restores AdamW,torchCPU/CUDARNG,exactepoch/batchcursor and
preserves schedule/data/source identities. Onlymicrobatch maychange. Ctrl+C saves
safeupdateboundary; validation finishes if inprogress. Save250/eval1000,full dev
withsourceCE,not test. last/previous/best only; no checkpoint proliferation.
Crash log tail archived before replay; mismatched hashes refused. Older recovery
archives newerbest. Frozen sourcezip andrecipe/progress logs retained.

Micro32 GPU synthetic worst-case full32vectors+512tokens FP32 probe passed:
7.466GiBallocated,8.090GiBreserved,Adam momentsallocated,0optimizerupdates.
Adam-step temporariesexcluded,notconcurrency/speedguarantee. Guard10.5GiB.
User command docs src/models/token_memory_chat/README.md; recommendmicro32,
effectivebatch32; reduce16onmemoryfailure. SixCPUtests pass, includingbitwise
weights/Adam/RNG pause/resume,sampler2epochcoverage andpartialbatch, regrouped
microbatchagreement, logtailpreservation,CE/mask/serializerchecks. Probe
report dump/chat-microbatch-32-probe.json. CLIhelpverified. No researchtraining,
parentweight changes,environment installs,DBvariant orautomaticworkers launched.

# Chat-first corpus PREPARED; comparison package relocated (2026-10-12)

User requests dataset/cleanup first, then matched DB/no-DB encoder models; hard
consumer cap23,209,216 including heads. No new research training or DB implemented.
Moved comparison trainer/layers/runtime into raw_transformer (trainer.py internal);
old folder removed, source zip dump/comparison-relocation-20261012.zip retained.
Known-source migration validates raw17k checkpoint; old unknown changes rejected.
Historical reporting tool tools/compare_saved_runs.py. Original memory sources and
all protected checkpoint hashes unchanged. TwelveCPU tests pass, including tiny
raw lifecycle/resume and chat masks/shift/epoch coverage; fresh prep smoke passes.

Prepared dump/data/chat-memory-v2:245974train/2490dev/2369test dialogues;
134649410train visible tokens,68254786assistant CE targets,356211segments.
2epochs,batch32=>22264updates,136509572scored targets. Sources pinned in manifest:
100kTinyStories total,108829filteredSmol-SmolTalk,2004everyday,40000syntheticmemory.
Full multi-turn dialogues retained, public assistant replies<=384tokens,total<=2560;
512rollover,EOS2,unchangedtokenizer,assistant-onlyCE,user tokens stay attention-valid.
Memory has3records,20%missing; same name pool for present/absent, heldout names and
questiontemplates. Source token/chunk/write/read/no-match labels separate fromtext.
Session-reset metadata only; no DB execution. Data not compatible with legacy
trainers: next needs shared chat trainer/serializer plus budgetedcontroller.

Preprocessing tools download_chat_sources.py,prepare_chat_corpus.py,audit_chat_corpus.py;
shared src/chat_data. Hash/mask/EOS/split/coverage/chunk causality audit passes,
532retokenized dialogues. Exact not semantic dedup; no universal chatbot guarantee.
Frozen prep sources/cards in dataset. One-off missing-name shortcut correction
archived in dump/finalize-chat-corpus-20261012.py, not active tools. Dataset docs
and2epochcontract docs/chat_dataset.md. No code/facts/labels should be treated as
proof of learned DB behavior. Keep saved original/reference weights safe.

# Mixed language/retrieval/chat COMPLETE, chat quality FAILED (2026-10-12)

All3000updates complete; selectedbest at3000, SHA
e84721d648f3b2b7b61205df65f6580543b5046344641e2f9a28a89286c56b68.
Original/retrieval parents and encoder unchanged; no extra training launched.
100dev CE3.01312original->3.00493mixed. Additional900documents (not selection):
CE3.026854->3.020117,paired95%deltaCI[-.007401,-.006128],712/900improve.
Retrieval512each correct/shuffled/blocked:newhist100/100/0%;newwording98.83/
99.02/0%;newbinding+wording86.13/85.16/0%. Changedfact followed100/98.83/86.13%.
Chat heldoutCE original2.74096->2.12515, but greedy loops and sampled replies
remain irrelevant/incorrect. Do NOT call this a usable general chatbot.
tools/chat_memory.py is a tested experimental CLI, /reset,/quit,EOS/rollover;
sampling options do not fix knowledge. Fixedsamples all retained. Full report
docs/mixed_memory_chat_results.md; result/fresh_language_comparison and completion
audits in run.3000-step/source/hash/protected-weight/prediction audits pass.
Previous recipe below records the completed run, not authorization to relaunch.

User authorized mixed training, better language, and a usable chatbot while
preserving original weights. One3000update continuation from retrieval2000
checkpoint (not overwriting original20k). New AdamW3e-5to3e-6,warmup100;
separately normalized losses LM.75/chat.15/retrieval.10; per-update batches
32/16/32,micro8. Frozen encoder,unchanged23.209M architecture. Tools source
train_mixed_memory.py/mixed_memory_support.py frozen at launch and copied to
dump/token-memory-mixed-chat-3000-20261012. No automatic failure retries.
Language sampler continues original seed47 shuffle from draw640000. Chat32k
train/256dev/256test, user-question SHA-disjoint from public smol-smoltalk pinned
revisionf73fe857d519ff6ac5af2ea67c4d3834da7b8bcc; one230MBshard downloaded.
16retrieval question templates train,4dev,4freshtest; heldout bindings retained.
New chat CLI tools/chat_memory.py prepared; final quality failed as above.
8CPU tests passed, target shift/EOS/causality. InitialLM CE3.2460 from specialist.
Completed plan: checkpoint selected using prespecified dev score only; paired900
additional language docs vsoriginal, retrieval ablatives,heldout chat CE and
greedy samples. Mixed experiment is complete; general chatbot quality remains unmet.

# Memory-dependent training COMPLETE (2026-10-12)

User authorized building memory-required data, training and final tests. One
2000update specialized continuation of token-memory20k, freshAdamW3e-5to3e-6,
batch64micro16,128000unique histories, encoder frozen,no architecture changes.
8record names,16single-token access words,8x64token history, answer-only CE.
Output dump/memory-retrieval-2000-20261012; source tools/train_memory_retrieval.py.
Reloaded final test correct/shuffled/blocked accuracy:newhistories100/100/8.40%,
held-out bindings90.63/89.84/0%,unseen question wording20.51/21.29/7.42%.
Changing only queried oldfact makes answers follow replacement100%familiar,
90.63%heldout. Genuine controlled retrieval, not universal fact recall.
Natural devCE worsened3.01312to3.24600,accuracy41.30to38.53%; keep original as
general model, experimentalcheckpoint not promoted. Template generalization weak.
No further training launched. Checks2000steps,savedmetrics/hash,no dataoverlap,
parent/encoder unchanged allpass. Script/data/audits/logs/results retained.
FinalSHA adb95a7745d8f0c40d32ebc95fc3de995953a139b0d8e39321f39e4b40b02a3d.

# Memory attention diagnostic COMPLETE (2026-10-12)

User authorized read-only attention/encoder checks; no training. Script
tools/inspect_memory_attention.py, evidence dump/memory-attention-20261012.
Same20k checkpoint100dev documents,400historysegments. Mean memoryattention
81.95%firstquery,20.38%queries2-16,10.00%17-64,5.34%65-128,3.84%129-256,
2.78%257-512. Not causal importance. Read-only logitbias log2/log4 worsens
historyCE3.0306624to3.0321724/3.0385272; paired bootstrap excludeszero.
Weights/encoder unchanged; zero-bias attention parity exact. Short encoder
1/8/32/64token inputs return finite1x256vectors; no quality claim. Single-token
codes differ from64-token history codes and are not a drop-in for trained LM
embeddings. No architecture edits, optimizer, training or selected bias deployed.

# Memory ablation COMPLETE (2026-10-12)

User authorized read-only correct/wrong/shuffled/blocked memory evaluation, no
training. Evaluated saved token-memory20k on same100dev documents,211200targets,
160000 with history. tools/evaluate_memory_ablation.py; output
dump/memory-ablation-20261012/report.json plus per_document.npz and source copy.
History CE correct3.0306624,wrong3.0403938,shuffled3.0312001,blocked3.0386164.
First64 CE correct3.3718677,blocked3.4129506. Paired document bootstrap intervals
exclude zero for blocked/wrong-minus-correct; small useful memory contribution,
strongest at rollover. No exact-retrieval or raw-parity claim. One corruption
draw, reuseddev,onecheckpoint. Blocked keys in every layer preserve RoPEpositions;
verified blocked predictions invariant to memory, empty-history control identical,
weights/encoder hashes unchanged,no gradients,savedCE replay within7.16e-6.
No optimizer instantiated, no research training or generated-text test.

# Hierarchical experiment RETIRED (2026-10-12)

User requested retiring hierarchy and cleaning legacy code/weights. Final20k
CE4.200900109,accuracy24.643939%,211200dev tokens,reload replay passed; original
token-memory20k CE3.013115671,accuracy41.297348%. No impossibility claim.
Removed hierarchy source directory and29 model checkpoints plus29sidecars from
three positively identified hierarchy runs:7,906,120,097bytes including code.
Manifest dump/hierarchy-cleanup-20261012.json; verified source/test archive
dump/retired-hierarchy-source-20261012.zip. Logs/data/recipes retained.
Removed hierarchy-only decoder layer, trainer branch and lock support; retained
raw and token-memory architectures/weights, required codec/Branch dependencies.
Narrow old-source hash migration supports raw resumes; optimizer/data/schedule
unchanged.14CPU tests passed including synthetic resume. No research training.
Historical preparation entries below no longer authorize hierarchy launches.

# SwiGLU experiment RETIRED (2026-10-12)

User removed src/models/token_memory_swiglu and requested cleanup. Confirmed
folder absent; removed 36 checkpoint/sidecar files from only the two SwiGLU runs,
freeing 4,643,232,892 bytes. Manifest dump/swiglu-cleanup-20261012.json.
Obsolete test archived in dump/test_token_memory_swiglu.retired.py then removed
from active tests. Logs, recipes and source archives retained. Other model code,
weights and data unchanged. At 12k SwiGLU CE3.1410817/accuracy39.6387%, original
memory CE3.1134804/39.7751%; schedules differ (12k vs20k endpoints), so no clean
FFN equivalence claim. No training launched. Historical preparation below.

# SwiGLU memory consumer PREPARED (2026-10-11)

User requested standard FFN instead of Branch Sigmoid in memory model. Active
fresh version src/models/token_memory_swiglu: bias-free down(silu(gate(x))*up(x)),
hidden1024,width512,8layers,exact23,209,216trainable params (same budget). Same
attention/512rollover/data/CE/seed47/batch32/LRschedule; non-FFN initialization
bitwise matches original. Frozen pretrained encoder unchanged. Legacy token_memory
kept for oldweights/evidence/inference; raw/hierarchical sources untouched, so
their resume hashes unaffected. Fresh CLI new model training.py; defaultmicro8,
20ksteps; rejects Branch checkpoints. Own format and source archive. Ten CPU tests
passed including formula,masking,gradient,exactbudget and synthetic2-step resume.
No research training or new GPU benchmark launched. Command package README.

# Raw activation checkpointing disabled (2026-10-11)

User explicitly requested speed: removed activation recomputation from raw model.
No weights/architecture/loss/data/schedule changes; matching forward/gradient
test passes. Shared trainer permits narrow known-source migration for this change
and raises raw memory guard to10GiB; hierarchy stays5GiB. CPU load of immutable
step10000 from raw-transformer-resume-20261011-075243 verified source migration.
Live worker left untouched; user must Ctrl+C, wait PAUSED, resume latest last.pt
from that folder into timestamped new name. Keepmicro8 initially; new GPU memory
peak/speed not benchmarked. Old checkpointed memory probes no longer predict fit.
Tiny synthetic lifecycle tests only; no research optimization launched. Qualitative
inference examples do not establish compression failure; controlled validation
and memory ablation remain the relevant tests. No hierarchical redesign made.

# Obsolete latent checkpoint cleanup and raw resume (2026-10-11)

User requested freeing space by removing unneeded models. Removed236 checkpoint
files (20.02GiB) from9 explicitly enumerated abandoned encoder-latent CE/MSE and
old latent CE/pilot runs. Only initial/last/best/step-N.pt removed; no recursive
directory deletion. All prepared.pt,data,logs,source archives,diagnostic runs,
three current comparison arms and pretrained codec retained. Old deleted runs
cannot be resumed/replayed from those weights. Exact manifest:
dump/obsolete-latent-checkpoint-cleanup-20261011.json.
raw-transformer-2560-resume already exists: last.pt hash/load verified step8500,
source hashes match. Stale status8625 and empty training.jsonl are not a completed
checkpoint. No active Python training worker observed. Recommended user resume
from that folder into new raw-transformer-2560-resume-v2, still20k total. No
training launched or original/current checkpoints deleted.

# Matched comparison arms PREPARED (2026-10-10)

User requests two fresh models, then will launch them in parallel themselves.
token-memory-512-micro16 completed20k: CE3.01311567,accuracy41.29735%,saved replay
passed. New raw_transformer23,077,376 trainable (GELU,width512,8layers,context2560)
and hierarchical_memory23,244,032 (global7x512BranchSigmoid,small3x256causal GELU
decoder,64token blocks,conditioning vector plus masked global-state cross-attention).
User selected source-equivalent raw context2048+512. Same data/tokenizer,seed47,
batch32,20kupdates,LRschedule,validation and target-token loss as existing23.209M.
Frozen encoder9,426,432 extra params in hierarchical only; raw never loads it.
Existing token_memory sources/checkpoints untouched. New internal comparison
trainer supports microbatch-only resume,per-arm locks allowing both jobs,CPU14/15,
progress/checkpoints/hashes/source archives. CLI per-model Python paths, no root
or PS1 launchers. User commands docs/three_model_comparison.md. Defaultmicro8.
16CPU tests passed including synthetic2-update resume for each arm,causal/no
future-code leakage,gradient flow,target alignment,activation checkpoint gradients,
generation global-state reuse. Zero-optimizer GPU forward/backward micro8 probes:
raw1.73GiB,hierarchy1.47GiB reserved; moments estimate0.17GiB each extra. Not a
concurrent-fit or training-speed guarantee. No research training launched.
Architecture differences (GELU/Branch,64/512 compression boundaries) and equalsteps
not equalcompute disclosed. compare.py reports common-step validation CE. Memory
ablation still pending. No additional models,databases or background training.

# Token-memory microbatch resume (2026-10-10)

User requests faster continuation of token-memory-512-fresh. Validation4000:
CE3.41827,accuracy35.88%; memory contribution not yet ablated. Running worker
left alone; no new research training or GPU benchmarks launched. Trainer now
allows microbatch-only regrouping on resume, preserving effective batch32,
optimizer, shuffled draw position and original20k LR schedule. Recommend micro8
(4accumulation passes); speed/fit not benchmarked. Ctrl+C once, wait PAUSED and
shell prompt, then resume into new name. Original trainer hash explicitly
migrated; all computational/data/encoder/runtime hashes still enforced.
Seven CPU tests and tiny synthetic resume/regrouping optimizer test passed;
regrouped floating-point results need not be bitwise identical. See training CLI
docs. Memory removed/shuffled evaluation remains pending, after speed priority.

# Step3 token-memory simplification PREPARED (2026-10-10)

User abandoned vector-output/MSE/parallel decoders for causal token CE with frozen
encoder memory. Active src/models/token_memory:23,209,216trainable parameters,
width512,8layers,8heads.512token active blocks; completed blocks become8independent
64token vectors. Older vectors and shifted raw tokens share causal attention.
No separate decoder. All prior memory retained; current data covers32older vectors.
Reuse downloaded2112token documents via prepare_data.py binary repacking:5segments
per document,512/512/512/512/64targets. No startup corpus encoding/retokenization.
Source hashes checked when repacking; training checks tokenizer/manifest/binary size.
Old source/test/docs byte-verified archive dump/superseded-vector-output-source.zip;
old active latent_predictor files removed, all runs/data/weights retained. Step1/2
shared CE trainer tests pass.5masking/boundary/gradient/data tests and synthetic
2-update resume equality passed. No research training or full conversion launched.
User triggers training; commands docs/training_cli.md. Old recipes below historical.

# MSE-only latent diagnostic COMPLETE (2026-10-09)

User requested encoder-target MSE only,decoder removed from training. One1000
update run dump/latent-mse-only-v1,same fresh initial weights/config/LR/16pairs
as CE predictor diagnostic. Train-normalized MSE2.481e-9;post-training decoder
evaluation100%tokens,16/16exact on fitted examples. Separate16dev examplesMSE1.5153,
0.0977%tokens,0/16exact:memorization success,not generalization. Encoder-only export
verified;decoder loaded only after training,no CE loss. Reload vector metrics exact,
1000sequence/hash verified. No new larger-data run,no default trainer loss change.
See docs/latent_prediction_diagnostic.md. Earlier CE-only restriction superseded
only for this explicitly requested bounded MSE test.

# Direct latent CE diagnostic COMPLETE (2026-10-09)

User authorized optimizing16free vectors without transformer. One1000-update
run dump/latent-direct-ce-v1,4096trainable scalars,initialized from prior fixed16
predictor outputs. z=initial+training_std*delta;Adam0.03to0.003,warmup25,no decay,
CE only. Accuracy23.1445to25.0977%,CE8.71375to8.48699,0/16exact. Finite gradients,
codec tensors unchanged,reload metrics exact,1000-step/hash audit passed.
Failure despite independent vectors supports decoder-objective optimization
difficulty from this initialization,not proof of an impossible objective or
unique root cause. No MSE/larger model/additional run. Evidence preserved;
see docs/latent_prediction_diagnostic.md. Per-example fitting,not generation.

# CE-only latent diagnostic COMPLETE (2026-10-09)

User authorized proposed16-example fitting diagnostic. One1000-update fresh
4.33M predictor from batch32 initial weights, same CE/frozen codec/LR,16fixed
first-to-second independent64-token chunk pairs. Output dump/latent-diagnostic-v2.
Training token accuracy23.1445%,CE8.71375,0/16exact;oracle16/16exact. Shuffling
inputsCE19.1993/0.3906%accuracy. Gradients finite/nonzero,codec tensors unchanged.
CE along predicted-to-target interpolation rises8.71to25.94at50%,then nearzero at
target:measured path barrier,not proof of local minimum or capacity shortage.
Reload/hash/optimizer1000 checks passed. V1 stopped0updates on changed-batch
bitwise assert;7.629e-6difference,original batching16/16bitwise matches verified.
All evidence preserved. No MSE,larger model,unfreezing or further updates launched.
See docs/latent_prediction_diagnostic.md. No generalization claim from fitting test.

# Direct per-model Python entry points (2026-10-09)

User removed train.ps1 and explicitly prefers python path/to/model/training.py.
Do not recreate PowerShell/root launchers. Root train.py removed; all three model
training.py files dispatch before heavy imports, resolving src themselves.
Shared CLI remains internal. Use python src/models/latent_predictor/training.py
with --name/--resume/--steps/etc. --status --run reads saved progress. Existing
CUDA Python must be selected by the user; no environment auto-switch/installation.

# Reusable training CLI added (2026-10-09)

User requested proper per-model arguments/progress/resume after their second
latent CE run completed1000updates in dump/latent-ce-20261009-170025. No live
training worker at implementation start. New root train.ps1/train.py dispatch to
model-local cli.py/training.py entry points for latent,branch,compressor; common
lifecycle models/training_cli.py and objective adapters models/training_tasks.py.
CE only remains selected; no MSE or new research runs launched. Resume restores
optimizer/sampler/CPU/CUDA RNG,checks prepared/checkpoint/architecture hashes and
writes separate continuation output. --steps means total; --additional-steps
extends explicitly,defaulting extension LR to parent's terminal LR. Shape/batch/
seed changes on resume refused. Ctrl+C saves at safe update boundary. Old pilot
checkpoints and frozen source archives untouched. Synthetic lifecycle tests
verify resume equality and accumulation; not research training. See docs/training_cli.md.

# Step3 latent-vector pilot COMPLETE, quality failed (2026-10-09)

User selected vector-first autoregressive output, fresh Branch Sigmoid consumer,
frozen downloaded codec, public baselines inference-only. Explicit budget1000
updates completed once, seed47,4,327,680 trainable parameters. Output
dump/latent-pilot-v1; implementation src/models/latent_predictor. Sources frozen
in run archive. CPU16 unavailable on this PC; single available logical CPU pinned
before imports. RTX5070 12GB. No environment changes or retries.
Independent64-token chunks prevent bidirectional teacher-forcing leakage from
joint512-token encoding. Frozen codec526/526dev chunks exact. Predictor final
decodedCE15.279390,token accuracy0.4793%,0/326exact; repetitive incoherent rollouts.
Not useful generation,not standard token-LM NLL,not a baseline win. Context32vectors
capacity but actual paragraphs max15vectors.3000train/200dev existing FineWeb
paragraphs,full original decoded/hash checks and split disjointness; codec exposure
possible,not fresh test.1000sequential updates,source/checkpoint hashes verified;
final reload replay exact;codec unchanged.271.87/288MiB postinitial allocated/reserved.
No extra budget. Preserve negative result. Step3B later persistent vector CRUD;
no database work yet. Historical compressor entries below are not current workers.

# Fresh60k second PC-crash recovery TRAINING (2026-10-08)

User again reported PC crash and explicitly requested continuing. No prior workers
survived. Recovery-v2 durable logs reached46500 but last.pt and best.pt have missing
payload/zero tails (not merely missing ZIP directory); unrecoverable as complete
optimizer checkpoints. Status,best_validation,memory JSON zero-filled. All retained.
Original best.pt at10000 still matches SHA
ad6bdbfbf413c135afd66b5dc7d684630c3b856ea6049a965798e65e5f609370 and loads correctly.
Resume that checkpoint:50000 remaining to60000.36500 logged v2 updates discarded;
unlogged work unknown,plus previously documented crash/first-attempt discards.
This is not resumption from46500 and must not be reported as such.
New sources dump/residual64-fresh10m60k-recovery-v3 and standalone output
artifacts/runs/residual64-fresh10m-60000-s47-recovery-v3. Restored AdamW10000,
sampler/CPU/CUDA RNG and reconstructed/verified target prefix. Full independent
decoded data identities retained. Same model/data/LR schedule and final target.
Launcher4492;worker17216 verified at10050. Checkpoint saving now flushes/fsyncs,
verifies ZIP CRC,then Windows write-through replacement; retains previous and
5000-step milestone backups. This reduces risk,not proof against disk/power loss.
Original v2 sources/evidence untouched. Prior validation memory peaks unavailable.
All final/best replay/export/hash checks retained;oneCPU16worker,no env changes,
extra experiments,automatic retry or automation. Sources frozen at launch.

# Fresh60k reboot recovery TRAINING (2026-10-08)

User reported PC reboot and requested continuation. No surviving Python/uv
workers. Original last.pt is incomplete (missing ZIP central directory); original
training log has a null tail,with250 consecutive durable rows after10000. Saved
status says10300,so at least300 updates were executed after the recovery point;
exact additional unlogged work unknown. Preserve original files and stale locks.
Verified best.pt at10000 is readable,finite,and matches exact model/source/data;
SHA ad6bdbfbf413c135afd66b5dc7d684630c3b856ea6049a965798e65e5f609370.
Resume only50000 remaining to60000,not a new60000 budget. Restore AdamW10000,
sampler,CPU/CUDA RNG; reconstruct and verify original target-schedule hash.
Original model/data/LR schedule unchanged. Frozen recovery sources:
dump/residual64-fresh10m60k-recovery-v2. Output standalone artifacts/runs/
residual64-fresh10m-60000-s47-recovery-v2. Original10k log/validation prefix copied
for full cumulative audits; previous26 discarded first-attempt updates retained.
Worker25864 verified update10025; independent complete data
identities and resume checks passed. V1 recovery stopped before updates on damaged memory.json metadata. V2 reconstructs
training memory from retained logs; prior validation memory peaks unavailable.
No decoded-data/native failure. Checkpoint
and null-tail damage diagnosed after reboot; no parser/data-identity bypass.
OneCPU16worker,uv run --no-sync,no environment changes,extra experiments,automation
or deletion. Do not alter frozen sources or blindly retry new failures. Full
final/best validation/export/hash audits retained. No inference promotion.

# Fresh60k user-confirmed restart TRAINING (2026-10-08)

User explicitly confirmed they stopped the first attempt and requested restart.
This explains missing controller/worker; do not label it an unexplained native
training crash. First attempt preserved in artifacts/runs/
residual64-fresh10m-60000-s47-v1-user-stopped-26:26 logged discarded updates,
possible unlogged in-flight work,no trained checkpoint (initial.pt retained).
Same frozen sources/recipe restarted fresh60000 updates,launcher28032,
worker22900 verified at update1. All in-worker identities and both-repo
fresh output parity passed again. No further weights deleted, no environment
changes or additional experiments. Previous launch PIDs below are historical.
Restart receipt:dump/residual64-fresh10m60k-v1/restart_receipt.json.

# Fresh selected 10.46M / 60,000 updates TRAINING (2026-10-08)

Latest user explicitly requested fresh 10M,60000 updates,delete current11M,
same64:1/context512,selected plain residual attention,and higher then decaying LR.
One fresh seed47 run:10456576 total/7327232 encoder parameters,code125,width256,
4encoder/1decoder,Branch Sigmoid/RoPE/depth_route. Zero inherited weights or
optimizer updates; fresh AdamW. Same25/25/25/25 short/packed/uniform/pattern mix,
micro2 x accumulation4. Warmup1000 to6e-4 then cosine to1e-5 at60000.
Launcher24860/controller22044/worker12192 verified past update1. Complete data
identities,source checks,and exact fresh logits/vector parity across both repos
passed inside worker. Initial336.04/376MiB allocated/reserved,not final peaks.
Only four positively verified completed11M checkpoints were deleted:initial,last,
best,encoder. Result/data/source history and Step1 retained; deletion manifest in
dump/residual64-fresh10m60k-v1/deleted_11m_weights.json. Earlier113 deletions remain.
Two conservative process-guard refusals preceded launch; no updates/deletions
in those attempts. Elevated inspection identified VSCode formatter/Jupyter
processes and GPU was idle. PowerShell startup failed before commands with
BadImageFormatException/security configuration read error; unchanged environment,
existing pinned Python via CMD used instead. Earlier runtime failures unresolved.
Sources and recipe now frozen. Do not modify or duplicate workers or automatically
retry new native/data failures. One worker CPU16,uv run --no-sync,no extra probes,
no additional training or automation. Full final/best replays and hashes retained.
Output standalone artifacts/runs/residual64-fresh10m-60000-s47-v1. Config points
to this fresh run; no trained-quality claim or100% guarantee. See
docs/residual64_fresh10m60k.md. Old999/1000 weights were deleted previously.

# Fresh11.02M residual64 COMPLETE; recovery target FAILED (2026-10-08)

All10000 seed47 fresh updates finished;final=selected best at10000. Short0/1000
exact,72.2112932%tokens,NLL1.3240414422;packed0/206exact,73.2611724%tokens,
NLL1.2930058041. Same11022336total/7610112encoder,context512/span64/code142.
Training434.5425/492MiB allocated/reserved;55.66min summed update-loop time excludes
validation/preparation. No worker failures;GPUidle. Controller full final/best
replays,encoder parity and all hashes passed. Independent pinned read-only rehash
verified result/final/best/encoder/record/frozen-source hashes and10000-step sequence.
Loss still improving:last shortNLL1.5014at9000,1.4151at9500,1.3240at10000. Cannot
infer sufficient future budget or guarantee100%. This random-start recipe failed
the reconstruction target;not comparable to old44k inherited/curriculum result.
Old999/1000 checkpoint was deleted in user-authorized cleanup;only historical
records remain. Fresh checkpoint saved;no additional training authorized. Config
already points to fresh last.pt;do not describe it as high-accuracy or ready
lossless inference. Product-format encoder export remains pending (research
encoder.pt exists with audited parity). See docs/residual64_fresh11m.md.
# Fresh11.02M residual64 TRAINING; old compressor weights deleted (2026-10-08)

User-authorized fresh10K run launched:worker11280,launcher30540,verified50/10000
updates. Exact11022336total/7610112encoder parameters;seed47,code142,64:1,512ctx,
zero inherited weights/optimizer/updates. Initialization receipt confirms fresh.
Full in-worker data/source/parameter checks passed;initial.pt saved before updates.
Early432.62/476MiB allocated/reserved,GPU635/8188MiB. No launch failure.
Old continuation stopped at7105 logged recovery updates. After broad-deletion
rejection,read-only metadata classification positively verified113 compressor
checkpoints (encoder+compress tensors,positive span,no Step1 blocks tensors).
Narrow deletion then approved and completed:11635507119bytes,including999/1000
winner and old encoder exports. Three non-checkpoint files retained. Step1 weights,
data,logs,records,source snapshots preserved. Old optimizer resumption no longer
possible from deleted weights;historical scores retained as historical only.
Manifest dump/residual64-fresh11m-v1/verified_compressor_weights.json and deletion
log retained. Both repo defaults now target fresh model;no encoder export yet.
Fresh training sources/recipe frozen;no automatic new failure retry,no extra budget.
Progress dump/residual64-fresh11m-v1/progress.ps1. See docs/residual64_fresh11m.md.
# Fresh residual64 approximately11M AUTHORIZED (2026-10-08)

Latest user:from scratch,delete old models,11M with10Kupdates. Supersedes old
compressor weight-retention rule only for verified compressor weights,including
999/1000winner;Step1 weights,data,logs,records retained. Old continuation process
tree stopped at7105 logged recovery updates. No active continuation authorized.
New one x10000 TOTAL fresh updates,seed47,max1worker,11022336total/7610112encoder
params. code_features142 vs125;stored vector256,span64,context512,width256,
4enc/1dec plain residual/BranchSigmoid/RoPE unchanged. No parentweight/optimizer
transfer. AdamWwarmup250to3e-4,cosine3e-5;usual dataset mixture and memory guards.
Initial save0,every250 thereafter;validation500,full final/best audits retained.
No extra probes,environment changes or automatic new native/data failure retries.
Source dump/residual64-fresh11m-v1;output standalone artifacts/runs/
residual64-fresh11m-10000-s47-v1. Prepared,launch observation follows.
Broad cleanup auto-review rejected:scope not proven Step1-safe. Read-only checkpoint
classification underway;delete only positively verified compressor weights after
narrow approval. No deletion has occurred yet. See docs/residual64_fresh11m.md.
# Residual64 recovery TRAINING (2026-10-08)

Worker31944 verified at50/14500 recovery updates (5550/20000 original schedule),
launcher20524. In-worker complete identities,archived computational sources,parent
and finite-state checks passed. Resume receipt confirms AdamW19500 and all sampler/
CPU/CUDA RNG restored. Early425.20/458MiB allocated/reserved;not final peaks.
No launch failure. Frozen recovery sources and recipe;no automatic new failure retry.
Previous999/1000 inference model unchanged. Progress dump/residual64-recovery-v1/progress.ps1.
# Residual64 interrupted-run recovery AUTHORIZED (2026-10-08)

Latest user says run and train longer for100%. Resume remaining14500 updates from
verified step5500 of interrupted20000 run;49500to64000 cumulative. No new20000
budget. Original log5545 means45 discarded logged updates will be re-executed;
possible unlogged work unknown. No original process remains;cause of interruption
unknown,no native-crash claim. Original checkpoints/logs/locks preserved.
New sources dump/residual64-recovery-v1;separate standalone output artifacts/runs/
residual64-recovery-14500-v1. Parent SHA1c2749d12e33854a6ab2b2e6ff55e0507c7d0228a4495d69d63c6ec31448f604.
Exact model/config/archived sources/finite optimizer checked;restore AdamW19500,
sampler/CPU/CUDA RNG. Same10456576params,64:1,context512,lr3e-5,data mixture.
SingleCPU16 worker;uv run --no-sync,no environment changes. All in-worker complete
identities and final audits retained;no automatic new native/data failure retries.
Sources freeze at launch. Previous999/1000 selected checkpoint unchanged. No extra
models,probes or automation. Previous pinnedPowerShell documentation write crashed
CLR0x80131506 after successful read-only checkpoint check;failure still preserved.
# Residual64 20,000-step continuation TRAINING (2026-10-08)

Launch verified: worker16392 completed first optimizer update of20000;launcher29372.
All in-worker data/source/parent checks passed. Resume receipt confirms full AdamW
state(step14000),sampler and CPU/CUDA RNG restored. Initial GPU627/8188MiB;
first-update training404.30/448MiB allocated/reserved,not final peaks. No launch
failure. Notebook training sources and recipe frozen. Progress
 dump/residual64-20000-v1/progress.ps1. Budget one x20000 additional,44000to64000
cumulative. No other training or inference promotion;no automation.
# Residual64 20,000-step continuation AUTHORIZED (2026-10-08)

Latest user explicitly requested onlysame10.46M model,64:1,context512,20000updates.
One20000 ADDITIONAL run from44000 to64000 cumulative. Previous15000 recipe was
prepared only and is NOT launched in addition. Run-specific recipe
 dump/residual64-20000-v1/recipe.json;standalone notebook runner unchanged.
Restore optimizer/sampler/CPU/CUDA RNG;constant3e-5,same data mixture/model size.
Full in-worker identities/source/parent/finite-gradient/memory checks and final
replays/hashes retained. Prior separate probes/initial validation waivers remain.
Max1worker,pinnedCPU16,uv run --no-sync,no environment changes,no blind new native/
data failure retries. Output standalone artifacts/runs/residual64-continue-20000-v1.
Sources/recipe frozen at launch. No model promotion or extra experiments. See
 docs/residual64_20000.md. No automation. Prepared;launch observation follows.
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
