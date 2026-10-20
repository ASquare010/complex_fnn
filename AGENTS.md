# Latest: canonical native256 low-bit memory cleanup (2026-10-20)

User explicitly requested removal of memory variants, native256 without256-to512
projection, exact23,209,216 reader budget, fresh47/twooriginalepochs and commit.
Active model src/models/token_memory:256-wide/8blocks/4heads/FFNtotal38062,
readlayers2/5/raw64/1globalposper64. Low-bit payload retained. FrozenE separate.
No production training started. Full command in src/models/token_memory/README.md.
Trainer readable logs/fullDEV every1000/save250; strict resume format/source/input/
recipe/sampler/cursor checks. New run name token-memory-native256-v1 (not created).
Old512 weights cannot be resumed into new256 architecture. No perfect recovery
or retained-quality/speed claim. Earlier freeze directives are superseded ONLY
for this user-requested canonical source migration; original data/E/cache/weights
remain untouched. Native2568750 and wide512 checkpoints preserved, never resumed.

Verified576-file local source archive +SHA manifest BEFORE cleanup under
 dump/token-memory-cleanup-20261020. Removed16 memory variant directories and278
associated obsolete helpers/tests, preserving experiment .md and completed run
sources.zip/receipts/checkpoints. Old chat CLI redirects to canonical trainer;
raw baseline/codec/corpus utilities retained. Broader cleanup rejected by automatic
review, narrowed to verified memory-specific targets and completed safely.
CPU model/objective and real tiny-CLI resume checks PASS; max-shape GPU micro4
zero-update capacity PASS, peak2.684GiB/cold1.858s; original corpus/cache/label
hash verification PASS. Final results doc records exact validation. No agents.

Step3A: compact storage and some reading established, perfect factual recovery
UNMET. Completed wide512: comp67/285 vsraw27/285 familiar exact facts, language
62.322818vs63.709511%, totalKV23.27xsmaller at2k, requests0.10–3.68%slower, transfer
weak/reserved0 includingknown. Its old results do not describe newnative256.

# Earlier completed study (historical source paths now archived)

# Latest: wide512 FULL COMPARISON VERIFIED; all workers finished; mixed result (2026-10-20)

Both48762/361357104LMlabels/fresh47/23209216reader+separatefrozenE; allfulltraining
auditsPASS(65203/17081), both identicalinitials/orders/LR/exposures confirmedagain
by pairedcomparison84261 TERMINAL0/PASS. MainGPU77837comp +pipeline76742 both
TERMINAL0; mainCPU48597comp/20630raw, notebook+transfer33564comp/49837raw,
rawreserved19280 allTERMINAL0/PASS. Compreserved89843handlepruned but itsbound
independent-verification receiptPASS and canonicalcomparison validatedit. Eacharm
has5 independentreceipts/132 deterministic publicCPUreplays. Do NOT rerun/restart
these completed studies. No study worker remains. No model/E/checkpoint/cache/
corpus/frozen182sources changed; docs/AGENTS are outside guardedclosures.

Canonicalfull-dev comp62.322818% CE1.726689593 vsraw63.709511% CE1.652498787.
Delta-1.386693pp/doc95CI[-1.438113,-1.336515]. Originalfactsexact comp67/285 vsraw
27/285(+14.035pp/sourceCI[9.8246,18.2456]); falseunknown195vs258. Missing60/63vs63/63.
V12vs6/Q19vs8of64; BOTHcorrectpairedV11vs6/Q11vs1. Notebookfull66vs39/256,
allfour7vs0/64; completion158vs130/256/allfour18vs6/64. Blocked0both. Transfer
original0/224both;prefix comp0 vsraw56/224; allfour0both. Sevenlegacy512groups0all;
historicalnotebook0/103 andchanged0/32both. Reserved1664enabledALLzeroexactboth,
includingknowncontrols; cannotattributefailureto noveltyalone. Last-number53probe
is referenceagreement NOT certifiedreasoningaccuracy. Single seed/fixedCI caveat.

GPUrequestbenchmark75147 TERMINAL0/PASS; CPUindependentverify8352 TERMINAL0/PASS.
12cases×6measured+warm:old128/512/2048,phase0/63,batch1/4,128forcedadvances.
Compmedians0.10–3.68% SLOWER; no latencywin (~4.3s/request). KV5.56/10.24/23.27x
smaller inclraw64+reserve;at2kB1 comp3.094MiB vsraw72.003MiB. At2kB4 peakallocated
590.62vs744.96MiB(20.7%lower), NOT23xtotalGPUmemory. ShortB1comppeakHIGHERdueE.
IncludesE/pack/prefill/rollover;excludestokenization/disk/modeltransfers/sampling/
textdecode. Raw1 isuncompressedCUSTOMcopyreader/raw64rollovercontrol, NOTvanilla
continuousKVTransformer. DoNOTgeneralizethe speedcomparison toanoptimizedvanillaLM.

Artifacts ownrun/comparison-48762/report.json andrequest-efficiency/{report,
independent-verification}. Fullreport docs/wide512_completed_comparison_20261020.md.
Userpreferences/UNVERIFIEDnextideas docs/wide512_final_architecture_preferences_20261020.md:
future no explicitadapter/nativewidth; currentP runsPREFILL/APPENDnot eachdecode,
131072MAC/block(~2048/oldtoken). Native256shoulduse256embeddingsdirectly; don't
projectevery512rawtoken down just toavoidper64adapter. 500+12encoder notimplemented.
Localreader touchesALLN IDs despiteN/64globalpositions. Fourfloats'low4bits encode
one16bitID;64IDs/256FP32floats,orderedoffsetsimplicit, no tokensdumped. Eunchanged.

Safespeedcandidate: fixedselector8×4096table128KiB avoids16.8MMAC/call; NOTimplemented/
timed/paritytested. Qualityhypothesis: returnedvaluepoolsbytokenID, lacks explicit
source-occurrence offset; add small weightedposition signal inSAMEreadoutwidth,
no extra globalKVpositions; some localwork/training required, UNVERIFIED. Keys/
globalpathalreadyhavecontext/recency/RoPE; don'tclaimallpositioninfoabsent. Current
closedliteraltestnowexposed; newmodelselectionneedsfreshheldoutdata. Keepboth
checkpointsimmutable, native2568750paused/untouched,growingretired,noagents.
ResearchobjectiveUNMET; no100Mvs10Bclaim. Useraskedopinion/comparisonFIRST; do not
automatically launchnewtraining/variants fromthisreport.

# Earlier execution record (status superseded above)

# Latest: BOTH wide512 arms COMPLETE; full training audits PASS; evaluation LIVE (2026-10-20)

Both compressed64/raw1 last checkpoints are48762/cursor(epoch2,batch0); no training
worker remains. Do NOT restart/extend either. User manually completed compressed
then raw1. CPU65203/17081 both TERMINAL0/PASS:361357104LMlabels EACH, all48762
orders/targets/LRs/cursors/actualmicroorders/auxdenominators, allAdam48762/finite
moments/weights,2770protected hashes/source182 checked. Bothsame78540querysegments/
157080terms and62826source segments/125644terms/62822name+62822value exposures.

GPU evaluator77837 TERMINAL0, compressed64 main evaluation report complete.
GPU76742 LIVE soleGPU sequentialnotebook/transfer/reservedcomp THENraw1 main/
notebook/transfer/reserved. Compnotebook done: full66/256/allfour7/64; completion
158/256/allfour18/64;blockedboth0/64. Transfer nowrunning. CPU48597 LIVEcore10
comp aggregate+public-source-replay (notebook/extra verifiers stillpending).
Do NOT duplicate/restart or edit frozen sources. Source182 and benchmark185 stay
immutable. No optimizer/new training by this comparison. Need full raw1 evaluation,
both notebook/transfer/reserved+verification, paired comparison and request benchmark.
BOTH full gates now pass; reserved inputs remain gated through final_gate().

Preliminary original-dev reaggregation: comp62.32287183%/CE1.726689592 vs raw1
63.70945709%/CE1.652498788; comp-raw=-1.38658526pp docCI[-1.43790,-1.33644].
Canonical read-only comp re-eval has ONE fewer argmax hit(1158823vs1158824), tiny
FP32 difference; CE1.7266895929. Use final evaluator docs/report for final comparison,
not preliminary SHA after canonical re-eval replaces doc metrics. No fact-parity claim.
Partial comp natural outputs67/285positive,60/63missing,V12/Q19of64,pairedV11/Q11,
blocked0/64; not yet full independently verified/matched to raw. Legacy groups all0.
Historical/reasoning outputs saved, report pending; last-number reasoning53 probe
is reference-number agreement, NOT independently verified mathematical correctness.

User preference saved docs/wide512_final_architecture_preferences_20261020.md:
future remove explicit projection, keep64:1; native256 or matched encoder width are
unverified candidates, no current model edit. Projection occurs on prefill/append,
not each cached token; local reader scores ALL N oldIDs despite N/64 global slots.
Bit example roundtrips64IDs; groups4floats encode one16-bitID, no token dumped;
4low semantic precision bits replaced. FP32 remains mandatory for this format.
100M beating10B is an aspiration, not evidence. Growing retired, no subagents.

# Earlier execution record (status superseded above)

# Latest: wide51244616 LIVE; full-dev6000=55.052792% reaggregation/metadata VERIFIED

Previousgoalturnverifiedwait; THISgoalturnPROGRESS actualnextfull-dev+audit.
6000 CE2.183717909 /1023645correct /1859388labels =55.05279156%. Historicalsame-step
originalraw512+oldermemory CE2.335473625/52.79801741%, oldplain64 CE2.767914763/
44.96533268%. Delta+2.25477415pp vsoriginal, +10.08745888 vsplain64. Own4k->6k
accuracy+1.57057053pp/CE-.107117849; ALL17ownsourceacc+CE improve. Memoryteacher
recovers96.649960%/.212211CE from4k96.008089%/.269053; stillbelow2k97.072013%.
DoNOTcallteacher-memorytokens factualretrieval orfull2epoch/matchedraw1 parity.

14/17sourcesbetteracc/lowerCE vsoriginal512 at6k; trailingEVERYDAY-1.581155pp,
STORIES-.780750pp, REASONING_TOOLS-1.907032pp. All17betterthanoldplain64.
CPU20599 TERMINAL0/PASS development-step6000-audit.json: allDoc+17source totals,
44,448,180LMlabel exposures/all6000originalorders/targets/LRs/cursorschecked,
actualcheckpointSHA/format/recipe/source/cursor/allAdamcounters6000/finiteMODEL
tensors. METADATA+aggregateaudit, NOTnewpublicCPUreplay/fullmoment/protectedfile
audit (strongerthosepathscompleted2k). Source182 unchanged; benchmark185closure
alsofrozen/prepared5CPUtests+12inputs,no GPUtimingyet. Experimentrecordupdated.

44616 authoritativeGPUconfirmedLIVE latestcompressed6375/48762 ~1.301updates/s;
raw1audited1waitingsequentialturn. No duplicate/restart/newGPUworker, no training/
model/E/cache/corpus/original/native256/TEST edits. Native256staypaused8750. Goal
ACTIVE/UNMET. Needbothfull48762+361357104LMlabels, fulltraining/optimizer audits,
matchedquality+freefacts/counterfactual/binding/reasoning/transfer/independentreplay;
reservedonlyafterBOTHfullgates; finalrequestefficiencybenchmark+verify. Keep
TorchVersioncompatbootstrapbelow forfrozenfullCLI, no source migration. Next
periodicdev8000; don'trepeatpassed6k audit/source-tests whileunchanged.

# Earlier execution record (status superseded above)

# Latest: wide51244616 LIVE; full-dev4000=53.482221% reaggregation/metadata VERIFIED

PreviousgoalturnPROGRESS finalrequestefficiencyclosure5CPUtests/12casesprepared;
THISgoalturnPROGRESS nextfull-devactualresult+independentreaggregation/metadata.
4000 CE2.290835758 /994442correct /1859388labels =53.48222103%. Historicalsame-step
originalraw512+oldermemory CE2.469273915/50.79117430%, oldplain64 CE2.903171538/
43.04163520%. Delta+2.69104673pp vsoriginal, +10.44058583 vsplain64. Own2k->4k
accuracy+2.91203342pp/CE-.218455676. NOTnewmatchedraw1 orfull2epoch/freefactquality.

CPU58285 TERMINAL0/PASS development-step4000-audit.json: allDoc+17source totals
reaggregated; all4000originalsampleorders/targets/LRs/cursors compared; exactly
29,660,180LMlabel exposures; actualsavedcheckpointSHA/format/recipe/source/cursor/
allAdamcounters4000/finiteMODELtensors checked. ThisisMETADATA+aggregate audit,
NOTnewpublicCPUreplay orfullmoment/protectedfileaudit (those strongerpaths done2k).
No checkpoint/source edit or newtraining/optimizerupdates. Oldsource182unchanged.

14/17sources betteracc/lowerCE vsoriginal512 at4k; negativegroupsEVERYDAY-2.226525pp,
STORIES-.865434pp, REASONING_TOOLS-1.072706pp. All17betterthanoldplain64. Memory
teacher own2k97.072013%/.166144CE ->4k96.008089%/.269053: -1.063923pp,+.102909CE.
DO NOThide thiscaveat orcallteacher-memorytokens factualretrieval. Recordupdated
docs/token_memory_64_wide512_training_progress.md. No adoption/rootcauseproof yet.

44616 authoritativeGPUconfirmedLIVE, latestobservedcompressed4225/48762,
~1.294updates/s inclperiodicdev; raw1audited1 awaitingsequentialturn. DoNOTrestart/
duplicate/newGPUworker orchange182frozenmain/185supplementalbenchmark sources.
FullgoalACTIVE/UNMET. Completeboth48762+361357104LMlabels/arm, fulltrainingaudits,
matchedfullquality/freefacts/counterfactuals/binding/reasoning/transfer/independent
replay, reservedonlyAFTERBOTHfullgates, thenefficiencybenchmark+verify. Alloriginal/
native256/E/corpus/cache/TEST untouched. KeepTorchVersionbootstrapbelowforfullCLI.

# Earlier execution record (status superseded above)

# Latest: wide51244616 LIVE; final request-efficiency closure PREPARED/VERIFIED

PreviousgoalturnPROGRESS actualstep2000 full-dev+all17CPUsourceverification.
THISgoalturnPROGRESS: remainingend-to-endMODELREQUEST efficiencybenchmarkcode/
tests/protocol/data/independentarithmeticverifier implemented inNEW files ONLY:
tools/benchmark_wide512_requests.py, tests/test_wide512_requests.py,
docs/wide512_request_efficiency_protocol.md. Main182frozenmodel/training/eval
sources UNCHANGED (sourceguardchecked); no newGPUworker, no trainingupdates.

FIVE CPUtests PASS: actualtinyModelrequests both1/64modes, ordinary/phase63 raw,
one/two rollovers, exactpayload/rawsuffix/schema, cached/publicLP equality,
unchangedweights/RNG/nogradients; emptyhistory; badrequests; incompletefullgate
rejectsBEFOREmodels/E/GPUinit; independentcacheformula/samplearithmetic rejects
corruptedKV/encoder/latencyreports. ActualCUDAinitializedFalse. Source-bound
receipt ownrun/request-efficiency-cpu-tests.json; prior4-mechanicsreceipt archived
explicitlyv1 beforeclosure completed. No full-run/source migration occurred.

Prepare54768 TERMINAL0:12deterministicoriginalDEV timingworkloads underownrun/
request-efficiency: oldtokens128/512/2048 ×rawphase0/63 ×batch1/4,128forcedtoken
advances, firstlength-eligibleoriginaldocs; no output/reconstructionfilter or
reservedTEST. CPUindependentall12caseauditPASSED exactcorpuslices/dimensions/
firstdocpolicy/caseids/source185/test5/noCUDA. Additional185-sourcebenchmark
closure NOW FROZEN incl3newfiles+main182; don'tedit afterpreparedmanifest.

GPUbenchmarkaction prohibiteduntilBOTH48762/361357104full-training audits and
singleworkerlock. Itusesactualfinalweights+originalfrozenE, one warmup/six measured
alternatingrounds. ActiveGPUreaderonly, Eresidentonlycomp. IncludesH2D/encoder/
packing/cacheallocation/globalprefill/rawsuffix/128advances/rollover E. Memory
peaks capturedBEFOREpayloadaudit. Sameproductionmax-new-tokenreservepolicy.
Checksallpayloadids/rawtail/counts/schema everyrequest. IndependentCPUverify
recomputesallsixsamples/timeparts/medians/ratios/Ecallshapes/exactKVformulasincl
raw64/reserve/boolmask/int64rawIDs. ThisisfixedtokenMODELrequestcost, excludes
tokenization/disk/checkpoint/device-weighttransfer/greedysampling/textdecode;
no EOSearlyexit, no whole-serving/64xtotal/neural-byte-losslessclaim. No timing
ornewqualityresultsYET. Needrunnextbenchmark+verifyAFTERfullbothcomplete.

CLI (CPU preparealreadydone, don'trepeat):
python tools/benchmark_wide512_requests.py benchmark
python tools/benchmark_wide512_requests.py verify
BenchmarkinternallyallowsTorchVersion ONLYforstrictweights_only load, unlike
frozenfullrunnerwhichstillneeds externalbootstrapbelow. No optimizerinstantiated.

44616GPUhandle confirmedLIVE latestcompressed3475/48762 ~1.310updates/s, raw1
waitingataudited1. Don'tduplicate/restart/launchGPU orrepeatalreadypassed2k audits.
GoalACTIVE/UNMET; finalfullbudget/matchedquality/freefacts/binding/reasoning/
reserved+independentverification+efficiency required. LatestqualitystillVERIFIED
2k50.570188%, promisingearlylanguageonly. Original/native256/corpus/E/cache/
checkpoints/TEST untouched, growingretired,noagents. Main2epochjobkeepsrunning.

# Earlier execution record (status superseded above)

# Latest: wide51244616 LIVE; step2000 FULL DEV50.570188% independently VERIFIED

Previousgoalturnverifiedwait; THISgoalturnPROGRESS: firstactualfull-dev result+
independentcheckpoint/Adam/exposure/aggregate/publicCPUverification completed.
Step2000 CE2.509291434/940296correct/1859388labels=50.57018761%; original512+older
memory SAMESTEP CE2.840776576/45.19965709%, oldplain64 CE3.191832439/39.05483955%.
Delta+5.37053052pp vsoriginal, +11.51534806pp vsplain64. ALL17sourceshigheracc/
lowerCE thanBOTH historicalsame-steprefs. TheseareNOTnewmatchedraw1control or
final2epoch/freefact/reasoningparity. Don'tcallteacherforcedmemory97%retrieval.

CPU85708 TERMINAL0/PASS capturedactual2000immutablecheckpoint, FROZEN owntraining
auditorall2000orders/LR/targets/cursors/physicalorders/auxdenominators/objectives,
allAdam2000/FP32moments-shapes-finite/weightsfinite/2770protectedchecked. Exact
14,841,890LMlabels/3181querysegments6362terms/2522sourcesegments5044terms/
2522name2522value. Reaggregated EVERYdoc+17sources; publicCPU replay13single-doc
sourcesmatchedGPUhits/targets/CEtol1e-4. CPU59112 TERMINAL0/PASS supplements4
multi-segment sources(memory2segs,reasoning_chat2,reasoning_tables2,multiturn3).
ALL17 sourcegroups nowpublicCPUreplayed. Allsources/model/oldnative/E/corpus/cache/
TEST unchanged. Zeroauditoptimizerupdates, no source migration ornewworkerGPU.
Artifacts development-step2000-audit.json +additional-replay.json +historical-
step2000-references.json underownrun. Newexperimentrecord docs/token_memory_64_
wide512_training_progress.md. Actualcheckpointcopy undercheckpoint-monitoring/
(pathinprimaryaudit); don'teditimmutablepassedreceipts/snapshots orrepeataudits.

44616 authoritativeGPUagainconfirmedLIVE, latestobservedcompressed2450/48762,
~1.294updates/s includesfirstperiodicdevtime; raw1pausedaudited1 awaitingturn.
DoNOTduplicate/restart/launchanotherGPUworker/edit182frozensources. Thisturnhas
actualearlyLANGUAGEqualityprogress butgoalACTIVE/UNMET. Needfull48762 EACH,
matchedraw1+freefacts/counterfactual/binding/reasoning/efficiency+allindependent
audits. ReservedstudentTEST onlyafterBOTH48762/361357104exposureaudits. Original512
referencehasoldermemory; literalIDs areaddedmetadata, notneural/byte64xlossless.
KeepTorchVersioncompatbootstrapbelowforallload-bearingactions, firststepnotrerun.

# Earlier execution record (status superseded above)

# Latest: wide51244616 LIVE; actual500 checkpoint/Adam audit PASS

PreviousgoalturnPROGRESS: originalpostwarm250log auditpassed. Thisgoalturnalso
PROGRESS: capturedstableLIVEcheckpoint500+originalprefix intoimmutableCPU-only
checkpoint-monitoring/20261018T092057Z; no originalsource/checkpoint edits, noGPU
ortrainingupdatesusedby audit. CPU19418 TERMINAL0/PASS usingFROZEN actualown
verify_wide512_run.training withprocess-localRUNoverrideto snapshot; initial/
manifest/orders bytesidentical, capturesha recorded. All500orders/labels/LRs/
cursors/physicalorders/auxiliarydenominators/objectives, allAdam500, FP32moment
shapes/finite, modelweightsfinite, all2770protected hasheschecked. Exposures:
3,684,531LMlabels/785querysegments1570terms/628sourcesegments1256terms/628name628value.
NOTqualityevidence. No source migration. Snapshotno livefile raceaftercapture.

44616 authoritativeGPUhandle againconfirmedLIVE; observedcompressed625/48762,
CE2.93434, ~1.333updates/s cumulativeearlytrain. Raw1audited1 awaitingsequential
turn. Nextmeaningfulqualitymilestone fulloriginaldev2000; don'trepeat already
passed500 audit/duplicate snapshots or create anothermodel/workerwhilewaiting.
All182 sourcebytesfrozen, original/native256/E/cache/corpus/TEST untouched. Keep
TorchVersionweights-onlycompatibilitybootstrapbelow; no model/trainer edits.
GoalACTIVE/UNMET. Fulloriginal2epochs/strongraw1/freefacts/binding/reasoning/
efficiency/evaluation audits required; can'tadoptfromcorrectness ortrainingCE.

# Earlier execution record (status superseded above)

# Latest: wide51244616 LIVE; post-warmup250 LOG audit PASS, no quality result

PreviousgoalturnwasPROGRESS: fullpipelineexport/data/allfirstgates+actualfulltrain
started. ThisgoalturnalsoPROGRESS: zero-updateCPU independentcapturedfirst250log
prefix audit PASSED. Save compressed64/log-audit-250/{training-prefix.jsonl,
report.json}. EACHof250originalsamplerorders/targets/LR matchesoriginalbaseline;
epoch/batch/cursor,stablelen-firstphysicalorders,micro8,actualauxiliarytables/
denominators/objectives/finite logged metrics checked. Warm200==3e-4, step201cosine
decreases. Totals1,831,425LMlabels/419querysegments838terms/335sourcesegments670
terms/335name335value; all182frozen sourcebytesmatchmanifest. Read-onlyLOG audit
ONLY, NOTcheckpoint/Adamstateaudit or qualityevidence. No source/model edits.

AuthoritativeGPUhandle44616 confirmedLIVE multiplepolls; observedcompressed250/
48762 CE3.81456 ~1.365updates/s cumulativeearlytrain, notend-to-endfinalspeedclaim.
Raw1stillpausedaudited1 awaitingsequentialturn. DoNOTduplicate/restart/launch
anotherGPUworker. Firstfulloriginaldev at2000 ispending. Native256/old/corpus/E/
cache/TEST unchanged. GoalACTIVE/UNMET. NeedfinalfullcomparisonAND meaningfulraw1
factcontrol; matchingtwofailedreadersisnotqualityparity. Finalefficiencyreviewmust
distinguish64:1 GLOBALmemorypositions fromtotalKV/raw64overhead/local64-wordwork,
and includeprefill/E/rollover foranyend-to-endclaim (priorcomponentbench excludes).
Keepcompatibilitybootstrapbelow forallload-bearingactions; nofrozenfilemigration.

# Earlier execution record (status superseded above)

# Latest: wide512 FULL fresh2-epoch training44616 LIVE soleGPU; first audits PASS

First audits12252compressed/core10 and16621raw1/core12 TERMINAL0/PASS: EACH7035LM
labels/3querysegments6terms/3sourcesegments6terms/3name3value; original/physical
orders/LR/cursor/recipe/allAdam1/momenttypes-shapes-finite/2770protectedchecked.
No firststep repeated orcheckpoint/sourceedited. Compatibilitybootstrap ONLY
allowlists standardtorch.torch_version.TorchVersion forstrictweights_only loader.

Fulltrainer44616 NOW LIVE soleGPU, PID7092/25020venvshim, core15, .7quota. Actual
compressedstep2 OBSERVED (CE7.8461). Startsfromaudited1 and does48761 NEW perarm,
sequential compressed64 thenraw1, exact48762 ORIGINALupdates/2epochs each. DoNOT
duplicate/restart/launchanotherGPU worker. No productionqualityresultyet. Preserve
sessionhandle; statusfilecanbelagging. Source182filesimmutable, sources exact.
training-launch.json +loader-compatibility.json recordfullrun/loader provenance.

ALL future load-bearingCLIactions needbootstrap (no frozenfileedits):
$env:PYTHONPATH='src;tools'
& .\.venv-cuda64\Scripts\python.exe -c "import torch,runpy; torch.serialization.add_safe_globals([torch.torch_version.TorchVersion]); runpy.run_path('tools/run_wide512.py',run_name='__main__')" ACTION FLAGS
Currentlytrain --arm both --resume --stop-after 48761 alreadyRUNNING; don'treissue.
Afterterminalcompletion needowntrainingaudit EACH thenfull-dev/freefacts/
counterfactual/notebook/reasoning/transfer/resultsindependentCPUreplay. ONLYafter
BOTH full48762/361357104labelauditPASS reserved studentTEST+verify+pairedcompare.
Native256 remainspaused8750/untouched; old/growing/corpus/E/cache/checkpoints
unchanged. GoalACTIVE/UNMET; new fullpipeline+export/data/allfirstgates arePROGRESS,
notqualityadoption. DoNOTmarkgoalcompletefortraininglaunched orpassingtests.

# Earlier execution record (status superseded above)

# Latest: wide512 first1 EACH saved; CPU first audits LIVE (loader compatibility)

Data51498 TERMINAL0/PASS alloriginalsegments/4,987,062blocks/initials/auxcoords.
GPU79608 TERMINAL0: exactly1update EACH, bothpaused1, CE7.92344/7.91687, immutable
first.pt snapshots. DoNOTretrain/repeatstep1. Initialfirstaudits98561/71920 FAILED
beforeunpickle: recipe torch.__version__ serialized TorchVersion(strsubclass),
strictweights_only loader rejectsbydefault. No source/checkpoint edits. Trusted
standardtorch.torch_version.TorchVersion ONLY isallowlisted in externalinvocation;
weights_only=True retained. loader-compatibility.json recordsbootstrap/runprovenance.
Firstaudits12252compressed/core10 and16621raw1/core12 NOW LIVE; noGPUworker.
FirstattemptbootstrapwithoutPYTHONPATH failedimportbeforework; use src;tools.
ForALL subsequentload-bearingCLIactions, use safe compatibility bootstrap:
$env:PYTHONPATH='src;tools'
& .\.venv-cuda64\Scripts\python.exe -c "import torch,runpy; torch.serialization.add_safe_globals([torch.torch_version.TorchVersion]); runpy.run_path('tools/run_wide512.py',run_name='__main__')" ACTION FLAGS
DoNOTeditfrozen recipe/trainer tofixmetadata or rewrite immutablefirst files.
AfterBOTH actualfirstaudits PASS, full resume48761 NEW each viaabovebootstrap
train --arm both --resume --stop-after48761 (flagneeds space before48761).
No old/native256/cache/source edits. GoalACTIVE/UNMET, noqualityresults yet.

# Earlier execution record (status superseded above)

# Latest: wide512 FULL-RUN PIPELINE exported; CPU data audit LIVE

User confirmed23M READER, frozen E separate. New canonical tools/wide512_run.py,
run_wide512.py CLI, owntrain/verify/evaluate_extra/compare helpers and fullprotocol
docs/wide512_full_training_protocol.md implemented. FIVE newCPU pipeline tests
PASSED (actual both-mode copy/generator LP equals publicpacked/actual loss;
physicalwithin-update stablelayouts+last26; scope restoresoldowners; reserved
full-exposure gate; sourceclosure). Syntaxpassed. Existing8 model/objectiveCPU
andfullsizeGPU/layout gates remainimmutable. Newraw1 hasCOPYreader; ownlanguage
usesreader.distribution BOTHarms, neveroldgen-onlyraw1 path. No code/proof edits.

Prepare83792 TERMINAL0/exported dump/token-memory-64-wide512-unified-v1; fresh47
BOTHactualinitials identicaltensor/RNG/emptyAdam,23209216params/Eseparate,182
frozen sourcepaths/2770protectedfiles, ORIGINALorders/effective32/48762updates/
361357104LMlabels EACH. Comp length-firstmicro8,raw1memory-firstmicro4, actual
physicalorderslogged/audited; .05query+.02source originalTRAIN-only auxiliary.
Exactformat token-memory-64-wide512-full-original-v1. Manifest/preflight/sourcezip
andinitial*.pt exported. All source/helper/eval/test/protocol bytes NOW FROZEN.

CPU DATA audit handle51498 LIVE; doNOTduplicate/restartfromsilence. NoGPU worker,
NOproductionupdatesyet. Afterdataaudit terminalPASS run train bothstop1skipinitial;
audit first EACH, immutablefirst.pt snapshots; BOTHfirstaudits requiredbefore
resume. Then train bothresume48761 NEW tofulloriginal2epochs. Full source/data/
recipe/RNG/cursor/Adam guards. DO NOTeditguardedfileswhilelive; no source migration
hasbeenused. ReservedstudentTESTreads gatedbehindBOTH full48762 exposureaudits.
Ownextrareserved verifier checksfrozenmanifest/files/sources/inputs andpublicCPU
generation. Fullcomparison includesoriginal512 full-dev reference, disclosed as
raw512PLUSoldermemory. Needtraining/full-dev/freefacts/counterfactual/notebook/
reasoning/transfer/independentreplay/reserved+pairedcomparison beforeanyquality
claim. GoalACTIVE/UNMET. Noold/native256/checkpoint/corpus/cache/TEST edits, no
growing resurrection, noagents. Native256 remainspaused8750; neverresume/editit.

# Earlier execution record (status superseded above)

# Latest: wide512 MODEL + CPU/GPU/LAYOUT gates VERIFIED; NO live worker

49771CPU,52566GPU,29944LAYOUT allTERMINAL0/PASS. Newsrcwide512 implemented512core/
256localdictionary/8layers/8heads/FFN11525/exact23209216reader/Eseparate/raw64/
1globalposper64. Genericraw1 SAMEparams/initials, full512wordembedding residual
+adapter,1position/word; notoldplainraw1 model. No extraKVhistoryfields.
8CPUgates,fullsizeCPU/GPUallparammax comp7.0e-7/raw8.46e-7, GPUmicromax3.92e-8,
public/cache1.91e-6; exactpayload/rollover; initialweightsunchanged/Adamempty.
Cachedfreshsamebatch2prefix:V279.449ms vswide34.003ms, cache6366356->4850836B,
23.8%less; componentonly excludesprefill/E/rollover. NOTquality orend-to-endspeed.
CompcodeRMS7.681925->memory.019990/rawEMB.019978; nativeoldinputscalehypothesis
nowmeasured, NOTcausalproof. Raw1mem.028062fullEMBresidual. Genuinecounts8/8 and
512/512 auditedviaoriginaldata selection. Highest512label/examplestress comp12
alloc6.945e9/res7.900e9B; raw4 alloc7.138e9/res7.411e9 inclmoment/scratch buffers.
Actualoriginaleffective32 den7035/3query/3source: comp length8 .715s vsoriginal8
.959;raw memory4 1.820 vsoriginal1 3.263. Maxallgrad2.24e-8. Sameexamplemultisets/
labels/denominators; onebatch3samples excludesAdam/E/IO/eval/save. No prodsteps.
Source tools/preflight_wide512.py +layout, objective/model/test/protocol andall
completedpreflight sources/results nowIMMUTABLE. No old/native/cache/data/TEST
changes. Growingretired,noagents. FullgoalACTIVE/UNMET; thisturnsubstantial
PROGRESSactualmodel/8tests/fullsizebench; noquality orfull2epochclaim.
NEXT: prepare NEW full-original two-epoch trainer/run/export/data/initial/auditor/
evalsourceclosure. Candidatecomp length-firstmicro8,raw1memory-firstmicro4,
effective32 ORIGINALorders/labels/LR, fresh47 BOTHarms/sameinitials,48,762updates/
361,357,104LMlabels each. Actualmicroorderexplicitlylog+audit. No oldpilotextends.
CRITICAL: oldtrain_native256_lexical_full.language hasraw1 GEN-ONLY branch because
oldraw1 modelhadNOreader. Newraw1 DOES havecopy/queryreader; newcommonlanguage/
verification MUSTscoremodel.reader.distribution forBOTHarms, testagainstactual
model.loss_sum/publicpacked. Do NOT reuseoldraw1gen-onlyeval silently.
ReuseoriginalData/Labelsread-only; .05query+.02source mainobjective unchangedboth,
noSTOPaux. Needdata/initial/source audits and actualfirst1updateboth+independent
Adam/cursor/labelcheck beforefull continuation. Prepare evaluation+verification
helpers BEFOREstarting; don'teditguarded sourceswhilelive. SingleGPUworker.
Results doc docs/token_memory_64_wide512_preflight_results.md; protocol preserved.
Allpreviousmodel/STOP/conditionalstudiesimmutable/notadopted; no new worker now.

# Earlier execution record (status superseded above)

# Latest: wide512 GPU preflight52566 confirmed LIVE soleGPU

52566 authoritativeGPUhandle LIVE, process3084/24484venvshim, core15, .7GPUquota.
DoNOTduplicate/restart. No optimizerupdates; currentwide model/objective/tests/
protocol/preflight sourcebytesFROZEN. CPUledger49771TERMINAL0/PASS8+2748guards.
GPUreport pending underdump/token-memory-64-wide512-preflight-20261018/gpu.
Old/native/TEST untouched. No training/quality/end-to-end speed claim yet.

# Earlier execution record (status superseded above)

# Latest: wide512 CPU ledger PASSED; GPU preflight LIVE (2026-10-18)

49771TERMINAL0/WIDE512CPUGATESPASSED8. Source/protected2748hashchecksPASS.
NewGPUpreflight launched fromtools/preflight_wide512.py gpu; see toolhandle
inlatestturn; SINGLEGPU, zero productionoptimizersteps. DoNOTduplicate. All
wide model/helper/tests/protocol/preflight bytes FROZEN. No trainer/run yet.
ActualnewsrcModel implements23M512core/256local/raw64/global64:1 +same-parameter
raw1control. FullCPUcorrectness/objective/serializedAdam checks passed. GPU
allparamgradient/capacity/cache/body-speed pending; noquality/end-to-endspeedclaim.
Old/native/sources/checkpoints/caches/data/TESTunchanged,no growing/noagents.
FullgoalACTIVE/UNMET; currentturnPROGRESSimplementedactualmodel+objective+8tests.

# Earlier execution record (status superseded above)

# Latest: wide512 IMPLEMENTED; frozen CPU preflight ledger LIVE (2026-10-18)

Newseparate src/models/token_memory_64_wide512 package implements512width/8layers/
8heads/FFN11525,256localdictionary,two reads2/5,exact23209216reader/Eseparate.
Old256codes unpackpayloadBEFOREnorm/proj; insertIDsfirst256ofexisting512Vslots.
Newgenericraw1controlsameinitialtensors/paramsusesFULL512wordembeddingresidual+
adapter,1oldword/globalpos; comp64still1code/64,raw64,cache8layersnohistorybank.
Memorygaininit1/sqrt256matchesroughinitialRMS. Old/native/source/checkpoint/cache/
corpus/TESTunchanged. No growing/subagents. Noquality/speed/productiontrainingyet.
FiveCPUmodeltestsPASS inclfullbudget/RNGidentity,geometry/raw1coords,independent
packed/publicallgrad,future/padding,cache/chunked/empty/blocked/payload,rollover.
FixednewemptyMreshapebugBEFOREanytraining. ThreeCPUobjectivechecksPASS botharms
(genericoldwordcoords,auxonlyselector/gates,globalmicrograd,exactserializedAdam).
Source tools/wide512_objective.py new; no oldhelperedits. Direct unittest23243
TERMINAL0/8unique tests. Newfrozenledger49771 LIVECPUcore10,printed8PASS15.956s;
nowfinalsource/2748protectedinputhash checks. POLL49771,notduplicate. CPUout
 dump/token-memory-64-wide512-preflight-20261018/cpu/report.json onlyvalidafter
terminalPASS. Allcurrentwide model/helper/test/protocol/preflightbytesFROZEN.
AfterCPUledgerterminalPASS run tools/preflight_wide512.py gpu SINGLEGPU; no
trainerexistsyet. PreflightfullsizeCPU/GPU/micrograd+compB8/B12,raw1B1/B2 capacity
andcache/genuineprefix/decodebenchplanned. Ownfreshdifferentarchitecturebenchmark,
notquality. Verifygenuineprefix2rowscounts; oldSTOPsame selectionboth8/8.
Protocol docs/token_memory_64_wide512_protocol.md. Fulloriginal2epochs/broadquality/
strongmatchedraw1reference remainactualgoal; no completion/adoptionclaim.
PreviousgoalturnPROGRESSverifiedSTOPregression+conditionaltradeoff; thisturn
PROGRESSactualnewsrcmodel/objective/test8/gatedpreflight; fullgoalACTIVE/UNMET.
AlloldjobsremainTERMINAL,doNOTrestart. NewCPUinventorydocsupersededbyimplementation.

# Earlier execution record (status superseded above)

# Latest: all workers COMPLETE; STOP rejected / conditional trade-off VERIFIED

96131TERMINAL0wideconditionalDEV;21149TERMINAL0independentauditPASS all1740cases,
6CPUvsGPUreplays,doccounts/scores/source/input/donorexact. NOliveworker. DoNOT
restartANYcompletedjob. STOPpaired512alsofullyverified39092/2748guards.
ConditionaloldV2same1000weights: natural225/285,missing30/63,pairedV49/Q39same;
notebook197/256/fullallfour37/64,completion256/256/allfour64/64same. Prefixaccess
3->95/224,original0/224both;blockedprefix0/56. Language44.213472->43.741220%,
-.472252pp docCI[-.502633,-.442032]. NOTqualitypreserving/adopted; no newtraining.
Docs native256_generator_stop_availability_results +native256_query_stop_results.
Alloldsource/models/checkpoints/cache/data/native/TEST unchanged. FullgoalACTIVE/
UNMET; thisturnPROGRESSmatchednegativepilot+actualCPU/GPUtradeoff+speeddiagnosis.
No growing/subagents. All completedstudy sources/results nowimmutable.
NEWCPUmoduleinventory only: dump/native512-local256-budget-20261018/report.json
PASS actual23209216: global512/8layers/8heads/FFN11525 core22312448; local256
queryreader764416; old256->512norm/proj/channelgain/typeadapter132352. Encoder
separate;1globalposition/64;KVonly103slots3375104 vsnative4429824bytes. NO full
model/forward/cache/training/speed/quality implementation. Docnative512_local256_budget_candidate.
Nextsafe research action: implement separate width512-core/256-local-reader
candidate with generic raw1 control and actual CPU/GPU causal/cache/gradient/
capacity/speed gates BEFORE training. DoNOT merelystartanotherfailedSTOPpilot,
reducecompression orclaim old1000weightsare2epochs. Full fresh2-original-epoch
comparison/strongrawreference/broadfacts/reasoning remainactualendgoal. Neither
newCPUbudgetcandidate nor prior shared-embedding packing component is adopted.
Layoutanalysis preservesall48762batches/labels +tinygradient/resume; fullGPUgate
pending. FP32backendprobe alreadyefficient: no guessedmathfallbackoptimization.

# Earlier execution record (status superseded above)

# Latest: STOP pilot VERIFIED rejected; conditional wide DEV still LIVE

96131 confirmedLIVEsoleGPU (13:12local), zero-update sameV2step1000 conditional
EOSavailability evaluation. FullDEVpart43.741220%/CE3.075815155 vsold44.213472%
(.472252ppcost, provisionalnotindependentlyverified). Naturalexact225/285,
missing30/63,V49/64,Q50/64,pairedV49/Q39,blocked0: SAMEscoresasold. Transfer
original0/224;prefix95/224 vsold3;blockedprefix0/56/shuffled0/56. Notebookenabled
full197/256 allfour37/64,completion256/256 allfour64/64: SAMEscoresasold.
96131 nowblockednotebook/finalguards; nofullreport/terminalconfirmedyet. POLL
SAMEhandle,neverduplicate. Noqualitypreservingwin/adoption, no newtraining.
Afterterminal CPUindependentaggregate/case+token/EOS/hash/scores/donor-replay,
selectedGPU-public-vsCPU-replay andsamecheckpointlanguagepairedcomparison needed.
CPUcachedreceipts6sourcecases from37191 available forGPUcompare; usearchivedAST
conditional function, oldModel/codec. Fullstageexcludeshistorical/reasoning/reserved
TEST. DoNOTclaimfullgoalsuccess fromprefix95; languagecostandunconstrained0remain.
ALLoldmodel/helper/protocol/test/CPUmath+newGPUscriptbytesFROZENwhilelive.
STOPpaired39092TERMINAL0/PASS2748guards:lossregresses43->3/285facts; noextend.
Resultsdocnative256_query_stop_results.md. Native/original/cache/data untouched.
FullgoalACTIVE/UNMET; thisturnactualmatchedpilotverification/newCPU+GPUevidence.
Futurefullbudgetchoice stillrequired; bestV2trained1000isnot2epochs. Noadditional
trainingrunstarted. CPUlayout/backendresultscomponentonly, notadopted.

# Earlier execution record (status superseded above)

# Latest: paired STOP pilot VERIFIED regression; wide conditional EOS eval LIVE

AllSTOPjobsTERMINAL0:81748fullSup,17380notebook/transfer,34780results/public/
notebook,53964transferverify,39092pairedcompare. comparison512PASSED2748guards.
Samefresh47architecture/initials/orders/3771495LMlabels/allAdam512 both. Ordinary
vsSup: fullDEV38.922592->38.877093%,CI[-.080989,-.010329]pp; positive43->3/285;
missing0both;pairedV4->0,Q1->0/64;notebookfull27->0/256,completion255->242/256;
allfourfull1->0/64,completion63->53/64. Allsevenaccess0/512,reasoning0/53both.
NOadoptionorautomaticextension. Both512checkpointsimmutable. Resultsdoc
native256_query_stop_results.md. Thisisolatesauxcoef0vs.02,SAMESTOParchitecture,
NOTV2architecturevsSTOP and NOTtwoepochcompletion. Original/native untouched.
NEW96131 confirmedLIVEsoleGPU, zero-updateWIDEDEV conditionalEOSavailability
ablation onpreservedV2step1000. Rootdump/native256-generator-stop-availability-dev-
20261018; archivedscriptcorresponding-source/probe.txt. Alloldmodel/source/cache
immutable. p.RUN scopedonlytonewroot forlanguage docs toavoidoldoutputoverwrite.
IncludesfullDEV/all540natural,224original/prefix+56blocked/shuffledtransfer,
notebook512enabled/128blocked. NoreservedTEST/reasoning53/historicalinTHISstage;
NOadoption/broadquality/freshtraining/speedclaim. Afterterminal independentCPU
aggregate/input/publicreplay +samecheckpointlanguage/notebook/transfercomparison
required. Models/helpers/mathartifact ALLFROZEN while96131live. DoNOTduplicate.
CPUbackendprobecompleted0:torch2.14cu132/RTX5070FP32 alreadyefficientSDPA at
128/551/3008 bool/alignedfloat, soNOmathfallbackfix. Docnative256_fp32_attention_backend_check.
CPUlayoutcheckgradient/resume andconditionalEOSsixcache/blockedcheckscompleted;
nooptimizationadopted. FullgoalACTIVE/UNMET; thisturnrealdiagnostics+pilotcompletion.

# Earlier execution record (status superseded above)

# Latest: supervised full evaluator COMPLETE; notebook/transfer GPU LIVE

81748 TERMINAL0 FULL EVALUATION COMPLETE512. SupfullDEV38.8770929%,CE3.569743280;
positives3/285,missing0/63,pairedV/Q0/64;accessallseven0/512;reasoning0/53.
Supnotyetindependentlyverified. 17380 LIVEsoleGPU canonicalSupnotebookTHENtransfer.
Whennotebookreportready CPUresults(core10) canoverlaptransfer; after17380TERMINAL0
CPUverify-transfer(core12). Final compare_native256_query_stop_pilot.py --step512
afterALLreceipts. Old81748 doNOTrestart. Bothtraining/checkpointsimmutable512,
models/helpers/protocols/testsourcesFROZEN whilelive. Native/original/TESTuntouched.
CPUconditionalEOScomponent92080/37191finishedPASS onoldV2step1000, selected
prefix15/28vs0, notebookALL8outputsidentical; notbroad/GPU/adopted. See docs.
FullgoalACTIVE/UNMET; thisturnactualnewCPUevidence+Supfullcompletion, notsuccess.

# Earlier execution record (status superseded above)

# Latest: supervised full eval reasoning LIVE; conditional EOS CPU checks pass

81748 confirmedLIVEsoleGPU at12:47 local, doNOTduplicate. Reasoningphase since
12:35; fullreportnotyetwritten. Afterterminal runSupnotebookTHENtransferGPU,
CPUresults+verifytransfer, finalpaired512comparison. Bothtrainingarmsimmutable.
NewseparateCPU92080/37191TERMINAL0: conditionalgeneratorEOSfallback onoldV2
step1000 (ifnoEOScopyprobpositive) givesselectedprefix15/28vs0, original0/28both;
ALLEIGHTnotebookoutputsidentical4/8exact. Sixpublic/cached+blockedCPUchecksPASS,
cachebytes/schemaandweightsunchanged. NoGPU/fullDEV/speed/adoptionclaim; numeric
underflowcantriggerfallback. Doc native256_generator_stop_availability_cpu_diagnostic.
PrecedingunconditionalvariantlostRemyanswer; retainfailure. Bothnotproduction.
Currentallmodel/helper/protocol/testsourcesFROZEN, native/original/TESTuntouched.
FullgoalACTIVE/UNMET; actualCPUevidence+verifiedwait thisturn, notsuccess.

# Earlier execution record (status superseded above)

# Latest: supervised full eval LIVE; two CPU diagnostics completed (2026-10-18)

81748 confirmedLIVEsoleGPU, supervised512 fullquality nowreasoningphase. Natural
savedprovisional3/285positive vsverifiedordinary43/285;missing0/63both; Sup
pairedV/Q0/64. AllsevenSupaccess0/512. SupfullDEV38.8770929 vs38.9225917 ordinary.
NotindependentlyverifiedSupyet. After81748TERMINAL0 SupnotebookTHENtransfer GPU,
thenCPUresults+verifytransfer andfinalpairedcompare. DO NOT duplicate/extend.
Allcurrentmodel/helper/protocol/testbytesFROZEN; old/native/TESTuntouched.
CPU25666/33967TERMINAL0: full2epoch within-update layout preservesall48762batch
multisets/361357104labels; lengthsortattentionproxy.7302, memorysortreaderproxy
.6253; actualtinyallgradmax2.98e-8/exact8stepAdamresume. NOT GPU speed/adopted.
CPU55421TERMINAL0: read-only generator-controlledEOS oldV2step1000 selectedDEV
prefixaccess15/28vs0, original0/28both, notebook3/8vs4 (Remyprematurestop).
Noadoption; modelweights/cache/sourceunchanged, smallsubsetnotbroadquality.
Docs native256_batch_layout_analysis +native256_generator_stop_cpu_diagnostic.
FullgoalACTIVE/UNMET; thisturnnewdiagnosticevidence+verifiedGPUwait, notsuccess.

# Earlier execution record (status superseded above)

# Latest: ordinary ALL quality checks VERIFIED; supervised full eval LIVE (2026-10-18)

17240 TERMINAL0 allordinaryaggregate/public/notebookchecksPASSED;5740 TERMINAL0
transferverificationPASSED inclCPUpublic/donor/casehash/score/guardchecks. Ordinary
512fullDEV38.9225917%,CE3.534536832;positives43/285,missing0/63,pairedV4/64,Q1/64.
Notebookfull27/256 allfour1/64;propercompletion255/256 allfour63/64;blocked0.
Allsevenaccess0/512;transferprefix1/224complete,original0/224;reasoning0/53.
No broadreadingwin. Both512checkpointsimmutable, noautomaticextension.
81748 LIVEsoleGPU supervisedfullquality512, canonicalmain. FullDEVpartobserved
38.8770929%,CE3.569743280 (notindependentlyverifiedyet);freefacts/transferpending.
Do NOT duplicate/restart81748. OnceTERMINAL0 runSupnotebookTHENtransferGPU,
CPUresults(core10)+verify-transfer(core12), thenpairedcompare512afterallreceipts.
ALL model/helper/protocol/testbytes FROZEN whilelive; mutablestatusonly. User
confirms23209216READERparams; encoderseparate. Noold/native/TESTchanges/noagents.
FullgoalACTIVE/UNMET; currentturnPROGRESSverifiedordinary+Supstarted, notcompleted
fullbudgetorqualitytarget. SeparateCPUsharedembeddingpackingcomponentdocumented,
notimplementedfullmodel, noquality/latencyclaim.

# Earlier execution record (status superseded above)

# Latest: supervised full eval LIVE; ordinary CPU checks LIVE (2026-10-18)

44786 TERMINAL0 ordinarynotebook+transfer. 81748 LIVEsoleGPU supervisedfull
evaluation512 canonicalimport/main; DO NOT duplicate/restart. Ordinary17240
CPUresults/public/notebook(core10) LIVE;5740 CPUverify-transfer(core12)LIVE.
Ordinary transferprefixcorrectvalue156/224 (includesmissingpositives),exact1/224;
unconstrained0/224,blockedprefix0/56,allfour0. Broaderqualitygainnotshown.
Both512training/checkpointsimmutable; models/helpers/protocols/testsFROZEN.
After81748terminal supervisednotebook+transfer sequentialGPU, thenCPUresults/
verify-transfer. Finalpairedcompareafterallreceipts. FullgoalACTIVE/UNMET.

# Earlier execution record (status superseded above)

# Latest: ordinary notebook complete; transfer GPU / results CPU live (2026-10-18)

44786 LIVEsoleGPU nowordinarytransfer; notebookbothmodes completed. Ordinary
notebook full27/256,allfour1/64;completion255/256,allfour63/64;blocked0all.
17240 LIVECPUcore10 ordinaryresults(aggregate/public/notebook) canoverlap
transfer because separateoutputs. Do NOT duplicate. Once44786terminal launch
CPUordinaryverify-transfer(core12) and supervisedGPUfull evaluator(core15).
Allmodels/helpers/protocolsfrozen, both512checkpointsimmutable. Fullgoalunmet.

# Earlier execution record (status superseded above)

# Latest: ordinary full evaluation recovered; notebook/transfer live (2026-10-18)

Prior session83062 is unavailable after context recovery. No Python process was
present and a completed full report is saved; do not rerun that evaluator.
Ordinary fullDEV38.9225917%, positives43/285, missing0/63, pairedV4/64/Q1/64;
allsevenaccess0/512, reasoning0/53. Provisional until independent results replay.
CurrentsoleGPU44786 canonicalmain ordinary notebook THEN transfer. Both training
arms immutable512, sourcebytesfrozen. Next CPUordinary results+verifytransfer
while supervisedfullGPU, then supervisednotebook/transfer and CPUchecks, paired
compare. Do NOT extend training. FullgoalACTIVE/UNMET; no old/native/TEST edits.
CPU shared-embedding packing component passed in separate JSON artifact; not a
complete model, noGPU/latency/quality result. Doc shared_embedding_packing_component.

# Latest live: STOP BOTH512 VERIFIED, ordinary quality eval (2026-10-18)

Trainer42581 TERMINAL0, BOTHarms512(.786ordinary/.782supervisedupdates/s). Ordinary
audit96037 andSup31224 TERMINAL0/PASS inclALLAdam512,actualorders/LR/cursor/source/
query/EOScounts; BOTH3771495LMlabels/14519EOS/3756976other/800querysegments1600slots/
643sourcesegments1286terms/643name643value. Bothdirectactualsourcebytechecks176
PASS, source-file-verification-512.json each. Immutablecheckpoints, NOextension.
GPUordinaryquality83062 LIVEsoleworker; fullDEVpartdone38.9225917%,CE3.534536832,
723722/1859388labels; memoryteacher95.032094%. NofreeQA/broadcomparisonyet,
superevaluationNOTstarted. Nextpoll83062 NEVERduplicate/restart. Afterterminal
runordinarynotebook+transfer GPUsequential viaCANONICALimport/main(args). Then
CPUordinaryresults(fullaggregate/public/notebook)andverify-transfer canoverlap
SupGPUfullquality83062successor, thenSupnotebook/transfer andCPUresults/verify.
Finalcompare_native256_query_stop_pilot.py --step512 afterALLreceipts. Every
GPUactionstep512/armcompressed64/conditionexplicit; canonicalmoduleentry only.
FullgoalACTIVE/UNMET: currentturnPROGRESSbothtrained/audited+budgetinventory+eval
started. ALLsourcesFROZEN. Noold/native/encoder/cache/corpus/TESTchanges/noagents.
DoNOTscore reservedstudentTESTorclaim2epochscomplete/qualitywin from512teacherCE.
CPUbudget8x512inventory reportonly(actualparams23209216/KVratio.761905), NOT
implemented forward/cache/latency/qualitycandidate. Statusdoclatestbelow.

# Earlier execution record (status superseded above)

42581 LIVEsoleGPU supervised250~.779updates/s. Ordinary512fullaudit96037TERMINAL0
PASS:3771495LMlabels/14519EOS/3756976other/800querysegments1600slots/643source
segments1286terms/643name643value/allAdam512/originalordersLRcursorschecked.
Additionalactualbytecheck ALL176declaredsourcesPASS, receiptordinary/compressed64/
source-file-verification-512.json. Noqualityevalyet. WaitSup512+42581terminal,
thenSupCPUtrainingaudit +directsources, ordinaryGPUfullqualitycanoverlapSupCPU
only; oneGPU. Allcurrenthelpers/protocols/modelcodeFROZEN. Usecanonicalimport
main(args) exclusively forpilotactions. FullgoalACTIVE/UNMET; noold/native/TEST
changes. CPU8x512budgetinventoryisnotimplementedforward/quality/latencyevidence.

# Earlier execution record (status superseded above)

42581 LIVEsoleGPU nowSUPERVISEDphase, observed25~.805updates/s. Ordinaryphase
printedPAUSED512 (terminalphase, .786updates/s); CPUordinarytrainingaudit96037
LIVE canonicalimport/main oncore10. No secondGPU.
AfterSupPAUSED512+42581terminal startSupCPUfullaudit. Ordinary/source/checkpoint
immutable whileCPUaudit. Allmodel/helper/protocolsourcesFROZEN. NewCPUbudget
inventoryonlycandidatepointbelow. FullgoalACTIVE/UNMET; noqualityresultsyet.

# Earlier execution record (status superseded above)

42581 stillLIVEsoleGPU, ordinary300 (~.777updates/s), supervisedqueuedsequential.
No sourcechanges/qualityresults. Continuepoll42581; CPUordinarytrainingauditwhen
phasePAUSED512 canoverlapSupGPU, thenSupfullaudit/evaluationafterbothphases.
Samecanonicalimport/main(args) restrictionsandgatesbelow. GoalACTIVE/UNMET.
PreviousturnPROGRESS; currentverifiedwait+newCPUbudgetevidence. Readonlyinventory
dump/native512-query-reader-budget-20261018/report.json PASS actualconstructed
modules23209216params:8layers/width512/heads8/FFN10980, projected256semanticcode,
twoqueryreaders+STOP1323008readerparams. KVonlysame103positions3375104vsnative
4429824B/sample,ratio.761905. NOforward/cache/training/latency/qualityclaim; no
ongoingmodelsourceedits. Futureefficiencycandidateonly. Doc
docs/native512_query_reader_budget_analysis.md. Allnative/original/TEST untouched.

# Earlier execution record (status superseded above)

Authoritative42581 stillLIVEsoleGPU, last ordinary125 (~.781updates/s), supervised
queuedsequentiallyinsameprocess. NOqualityresults; bothfirst/data/gpu gatespassed.
Nextcontinuation MUSTpoll42581,neverduplicate/restartfromfiles/timeout. When
ordinaryphaseprintsPAUSED512, CPUtrainingauditordinary canoverlapsupervisedGPU;
afterbothphasescomplete auditSup. Usecanonicalimport r.main(args) ALWAYS for
thispilotdriver, neverCLI/runpy/module__main__ (aliasbugpreserved, no sourceedits).
Trainingaudit invocation:
sys.path[:0]=['src','tools']; import run_native256_query_stop_pilot as r;
r.torch.set_num_threads(1); r.main(SimpleNamespace(action='audit',condition=
'ordinary'/'supervised',audit='training',arm='compressed64')). PinCPU10/12 via
evaluate_native256_query_selection.pin; auditmodule.runner mustbe r.
ThenGPUactions evaluate/notebook/transfer step512 eacharm, exactlyoneGPUworker;
CPUresults afterbothfull/notebookoutputsready, CPUverify-transfer aftertransfer
outputs; all via SAMEcanonicalmain and completeargs(step512,armcompressed64).
Finally tools/compare_native256_query_stop_pilot.py --step512 (alreadycanonical
imports). NoreservedTEST/newmodelsource editswhilelive. Allworkerhandlesabove
firstcompleted audits terminal. FullgoalACTIVE/UNMET; thisturnPROGRESSthrough
newactualtraining, notqualitysuccess. Statusdocrecordsallpreviousexecution.

# Earlier execution record (status superseded above)

Firstaudits92011ordinary/35034supervised TERMINAL0/PASS: allAdam1,7035LMlabels,
32EOS/7003other,3query/source,6slot/source/3name3value,correctrecipe/coefs/cursors.
Resumedtrainer42581 LIVEsoleGPU, sequentialordinary511NEWthen supervised511NEW,
eachfromactual1to512hardbound. Lastordinaryobserved50,~.77updates/s. Poll42581
only,neverrestartfromstatusfiles/timeout. Canonicalimport/main(args) essential.
OnceordinaryprintsPAUSED512 itisphase-terminalandCPUtrainingauditcanoverlap
supervisedGPU; nosecondGPU. Afterbothphasefinishtrainingsauditseach, fullquality/
notebook/transfer +independentreplays, pairedcomparisonalreadyprepared. Allsources
FROZEN. Noqualityresultyet; previous/currentturnPROGRESS,fullgoalACTIVE/UNMET.
Source/data/export/firstgatespassed. Noold/native/TEST changes; nogrowing/subagents.

# Earlier execution record (status superseded above)

Correctcanonicaldataaudits98108/22071 TERMINAL0/PASS withprintedordinary/supervised.
Actualfirsttrainer44586 TERMINAL0, BOTHarms paused1. SAME7035LMlabels/32EOS/
7003nonEOS/3querysegments/3sourcesegments; initialCE7.88947/query1.70469/source
6.74285both. OrdinarySTOPcoef0loss0objective8.109566489; supervisedcoef.02,
balancedSTOP2.018149853objective8.149929486. Firstauditsordinary92011 LIVEcore10,
supervised35034 LIVEcore12. NoGPUworker. Waitpassedall_actual_stop_denominators_
checked receipts before511NEWupdates perarm. Alwayscanonicalimport/main(args),
neverdirectCLI/runpydriver (aliasdiagnosis below). Model/source/protocol/eval
FROZEN. CurrentturnPROGRESSactualproductionfirstupdates,fullgoalACTIVE/UNMET.
NooldV2/native/data/cache/TEST changes. AfterownfirstPASS resumeordinary511then
supervised511sequentialoneGPU, fulltrainingauditsandevaluationpipelinealready
prepared. Preserveactualfirst/currentmodelstate; doNOTduplicatefromstatusfiles.

# Earlier execution record (status superseded above)

CAUTION newpilotdriver mustbe IMPORTED canonically; call r.main(SimpleNamespace),
NOTscript/runpy/__main__. Frozen driveras__main__ +auditorimport createsdifferent
module withCONDITION=None. Firstdataaudits68040/68512TERMINAL0butINVALIDcondition
selection(supervisedcomparedself). Preserved data-audit-cli-module-alias-invalid
.json and ROOT/entrypoint-diagnosis.json. NOproductionupdates;initialsunchanged.
No sourceedits/newvariant needed: canonicalimport executesexactguardedcodeand
sharesr.CONDITION withauditor/evaluator. Correctedcanonicaldataauditordinary98108
LIVEcore10, supervised22071 LIVEcore12. Polltheseonly. MustbothPASScorrectcondition
beforefirst1. Allfuturetrain/audit/evaluate/notebook/results/transfer actions via
canonicalimport r=run_native256_query_stop_pilot, r.main(args). Existingcomparison
CLIimportsrunnercanonicallyandcanrunaswritten. Model/protocol/sourcebytesfrozen.
GoalACTIVE/UNMET; currentturnPROGRESS(gates/export+entrypointfixbeforetraining),
noGPUworker. Prior/old/native/TEST untouched. Earlierlive labels superseded.

# Earlier execution record (status superseded above)

Export82852 TERMINAL0, BOTHordinary/supervised exported. Independentdataaudit
ordinary68040 LIVEcore10, supervised68512 LIVEcore12; noGPUworker yet. Pollboth,
neverrestartfromfiles/timeout. Source/protocol/tests/model/trainer/evaluator frozen.
AfterPASS each actualfirst1 via newdriver, independentfirstaudits, then511NEW
each onlyifownfirstaudit passes. Runroot/source/contracts below unchanged.
PreviousgoalturnPROGRESS; currentPROGRESS(fullGPUgates+freshpairedguardedexports).
FullgoalACTIVE/UNMET; noold/nativeupdates/TEST scoring. Mutable status doc
docs/native256_query_stop_pilot_status.md (earlierlive labels historical).

# Earlier execution record (status superseded above)

Full-sizeGPU22696 TERMINAL0/PASS: allparamCPU/GPU4.10e-7,micro1.86e-8,normrel
6.85e-6,lossdiff<=2.39e-7,initialexact/emptyAdam/2589oldguardspass. CachedoldV2/
STOPbatch2 82.674/83.012ms same6366356KVbytes; actualprefixbothrowsoldcount8,
noextraKV. ComponentonlyexcludesAdam/E/IO/prefill/rollover. Noqualityclaim.
Newpairedpilotdriver run_native256_query_stop_pilot.py/auditor/transfer evaluator/
comparison/protocol/2driverCPUtests prepared;3combinedobjective testsPASS18414.
SourcefilesNOWFROZEN. Export82852 LIVE sequentialordinary->supervised; ordinary
alreadyexported, supervise pending. NoGPUtrainingworker yet. Poll82852,dontduplicate.
ROOT dump/native256-query-stop-pilot-20261018/{ordinary,supervised}, uniqueformats
native256-query-stop-pilot-CONDITION-v1, eachcompressed64layout. SAMEfreshSTOP23M
model/init/RNG/data/orders/LR, STOPaux0vs.02; sameLM+.05query+.02source. Noextradata.
Nextafterexportterminal: independentdataauditsboth+pairedexactinit; actualfirst1
each+firstaudits; then511NEWupdate/armeach, total512hardbound. Afteractualtraining
audits fullDEV/natural/legacy/historical/reasoning/notebook+transfer, independent
publicreplays andpairedcomparison. Allhelper/evalfilesalreadycreatedbeforeexport.
DoNOTresumeoldV2/native orscore reservedstudentTEST. FullgoalACTIVE/UNMET;
currentturnPROGRESS(actualcombinedobjective/GPUgates/guardednewpilotexport).
Mutablestatus docs/native256_query_stop_pilot_status.md. Allold/originalprotected
filesunchanged, no growing/subagents. Full2epoch/strongraw/broadqualityremaining.

# Earlier execution record (status superseded above)

PreviousgoalturnPROGRESS(diagnosis+newSTOPprototype), fullgoalACTIVE/UNMET.
Newtools/native256_query_stop_training.py composesfrozenLM/query/source+.02STOP
usingexistingselectedCEh rows, nosecondtransformerforward. New3CPUtests18414
TERMINAL0/PASS: coefficient0exactoldcombinedAdam, detachedSTOPonly, globalmicro/
actualcombined8stepserializedAdam. OldSTOPprototype/source/CPUreceipt unchanged.
Newpreflight tools/preflight_native256_query_stop.py +docs/native256_query_stop_
gpu_protocol.md NOWFROZEN; GPU22696 LIVEsoleworker. Gradient/microgates passed;
capacitymixedB8/m39/raw5125.58GiBallocated/6.41reserved inclmoment+scratchsized
buffers, invalidpaddinglossexact0. Bodycoef0/.02 .35358/.35322s mixed,
.36105/.36429s padded,3samplescomponentonly. Cachedtiming/finalguardcheckpending.
Poll22696;doNOTduplicate ortrainuntilterminalpassedreceipt. No productionupdates,
old/native/source/cache/data/TEST untouched. NextifPASS newboundedpairedfresh
STOPpilot withsamearchitecture/data/schedule andSTOPaux0vs.02, notoldrunresume.
Neednewdriver/trainer/auditor/protocol/recipe/sourceguards beforeactualfirstupdate.

# Earlier execution record (status superseded above)

CurrentgoalturnPROGRESS: readonlyfailuresdecomposed andNEWseparateSTOPmodel/code.
FullgoalACTIVE/UNMET. ALLworkersTERMINAL0: heads96074/GPU87047/public6974,
stoppingGPU22600/independent26572; CPUprototype52904PASS6tests. NoGPU/CPUworker
remains. DoNOTrestart/extendcompletedquerypilot; checkpoint1000immutable.
Diagnosis dump/native256-query-transfer-diagnosis-20261018; ALL3584heads audited:
familiar/values/names/positions512/512entity/topic/record, missing256/256present
signal/noneabsent;wording462entity0topic,composed256entity0topic,allscopevalid.
224subsetoriginal32cap0exact/0EOS;value-freeprefix176/208positivefirstSINGLEvalue,
0/16unknown;3/224strict. Blocked0/56prefixvalues;donorcodeexactreplayverified.
Decisionanalysis704teacherendpoints: accessoriginalgeneratorEOS0/224; prefixed
accessgeneratorEOS212/224/final10/224/202overridden;notebook256/256both. Access
pointerEOS0,notebook.991615mean. Allmixtureidentities/CPUpublicsamples/features/
2589priorguardspass. NOTfreequalitysuccess orsemanticencoderlosslessness.
NEW src/models/token_memory_64_query_stop (reader/transformer/init) + tools/
native256_query_stop_objective.py; protocol docs/native256_query_stop_protocol.md.
512finalSTOPweights fund1FFNunit31368->31367, exact23209216reader/Eseparate,
raw64/OLD64:1/noextraKV. OutputlocalSTOPredistributesONLYnonNULLcopytoEOS;
NULLgeneratorfallbackpreserved;empty/blockedexactoldfallback. Prior2readlayers
andcacheclass/arraysunchanged. Freshnewarch, NOcheckpointmigration/trainingyet.
Prospectivetrainingaux .02*balancedoriginalEOS/nonEOSCE, detachedh,globalclass
denominators; no newfacts/inferencegold. CPU6PASSED inclbudget/ownfresh,indepLP/
allactivegrads,NULL,cache/public/chunk/realrollover/sameKV,causalallgrads,
actualmixedobjective/microbatch/exactserialized8stepAdam. CPUreceipt dump/
native256-query-stop-cpu-preflight-20261018/report.json (source key singular).
Model/objective/tests/protocolNOWFROZENbyreceipt. Next: NEWfull21-layerGPUactual
combined LM/query/source/STOP gradient+micro/capacity/timingpreflight, preserving
oldguardedsources. Thennewboundedpairedpilotifgatespass; doNOTresumeoldentrypoint.
Initialvalueonlyformat/missing/broadreasoning/freshfull2epochs remainunresolved.
Nooriginal/native/cache/data/TESTchanges,nogrowing/subagents. Report
docs/native256_query_transfer_results.md. NoreservedstudentTESTscoring.

# Earlier execution record (status superseded above)

PreviousgoalturnPROGRESS: fullpilot/notebook/public/pairedcomparison verified;
goalACTIVE/UNMET. Newtools/diagnose_native256_query_transfer.py +protocol/tests
createdBEFOREexecution, NOWFROZEN. TwoCPUtestsPASS;prepareTERMINAL0. Dataset
dump/native256-query-transfer-diagnosis-20261018:ALL3584legacyDEVcases, subsetfirst
8sortedunits/group224original32cap+224value-freeprefix16cap;56blocked+56donorswap.
NOreservedTEST/training/modelchanges. CPUheads96074TERMINAL0/PASSED. GPU87047
LIVEsoleGPU; pollsamehandle. Then runhelperverify CPU; noautomaticextension.
Headsfull512/512entity/topic/correctrecordsignal forfamiliar/values/names/positions;
missing512entity/topic,256/256presentrecords,nomissingfalsesignal. Wording462/512
entity/recordsignal,0topic;composed256/512entity/recordsignal,0topic. Allscope512.
ThusALLfamiliar0outputsarenotexplainedbyheadselectionfailure. Output/copy/context
transferdiagnosisawaitsGPUcaps/prefixes+controls/publicverification. Fixedprefix
usesONLYquestionname/relation, noanswer. Noexistingfilesedited;allpriorimmutable.

# Earlier execution record (status superseded above)

ALLworkers TERMINAL0/PASSED: trainer24899/training55101/fullGPU91682/aggregate+
public58439/notebookGPU74041+CPU82295/selector60579/metadataGPU45266/comparison89820.
DoNOTrestart/automaticallyextend completed1,000pilot. Checkpointimmutable SHA
d9fd0c7b4743f1d542639ce46ab2ce848c382825455330da84fc5dac2a6684bc.
Finalcomparison verifies2,589 protected/input/source/outputguards, same7416257LM
labels/orders/LR/cursors/prompts. FreshownV2seed47,23,209,216READER/Eseparate,
old64:1globalpositions/raw64/noextraKV;3072queryweightsreplace6FFNunits.
ADDEDliteralIDmetadata+.05query/.02criticalsourcesupervision, notoldsemantic
encoderlosslessness oridenticalinitialcontroltensors/objective.
Verified natural positive225/285 vsSpan23; missing30/63vs13; changedV49/64vs3,
Q50/64vs1; pairedV49vs2,Q39vs0;blocked0. Notebookfull197/256vs15,37/64allfourvs0,
98/128bothswapsvs2; propercompletion256/256vs77,64/64allfourvs0,128/128pairsvs11.
Blocked0/64bothforms. Wholequeryraw/norolloverfull38/44,8/11allfour;completion
44/44,11/11allfour;gainnotsolelyrollover. Full-dev44.213472vs44.164101%,CE
3.057949714vs3.062043972;+.049371pp doc95CI[+.013191,+.086192],fixedweightsnotseed.
ALL7accessgroups0/512;numericreasoning0/53;historicalpositive10/103,changed2/32.
Selectorbothslots348/348DEV,1551/1551TRAIN; oldentity285/285positive/none63missing.
Strong narrow notebookbinding progress, NOTbroad/finalgoalcomplete. Fullgoal
ACTIVE/UNMET;2originalepochs/strongrawcontrol/TEST/novelliterals/efficiencyremain.
OriginalV2~.884updates/s;metadatavectorization componentchecks/timingpassed but
NOTusedbyqualityeval andfullfreeoutputparityNOTchecked; noadoption/e2espeedclaim.
Nextconcreteresearch: read-only diagnosisofcross-task queryselection/output
failure (allaccess0), distinguishquestionhead/domain/answerformat; no newrunbegun.
Native256/originals/TESTunchanged,no growing/subagents. Report
docs/native256_query_entity_results.md; matchedJSONrun/comparison-1000/report.json.

# Earlier execution record (status superseded above)

NotebookGPU74041 TERMINAL0: enabledfull197/256,37/64allfour,98/128bothswaps;
propercompletion256/256,64/64allfour,128/128bothswaps. Blocked0/64bothformats.
NoGPUworker remains. CPUpublic58439 LIVE core12 (replays throughreasoningdone,
finalprotectedhashcheckpending); notebookCPU82295 LIVE core10. AfterbothPASS,
run compare_native256_query_entity.py --step1000; doNOT duplicate/retrain.
This is a strong narrow notebook-binding signal, NOT broad/finalqualityparity.

# Earlier execution record (status superseded above)

Trainer24899/trainingaudit55101/metadataGPU45266/selectorCPU60579/fullGPUeval91682
all TERMINAL0/PASSED. No further updates; checkpoint1000 immutable SHA
d9fd0c7b4743f1d542639ce46ab2ce848c382825455330da84fc5dac2a6684bc.
New notebook GPU74041 LIVE soleGPU; aggregate+public CPU58439 LIVE core12.
Poll handles; doNOT duplicate. Next notebook independent CPU verifier, then
compare_native256_query_entity.py --step1000; no automatic training extension.
Provisional full-dev44.213472% CE3.057949714; natural225/285 positive,30/63missing,
49/64changedvalue,50/64changedquery,paired49/39,blocked0. All7accessgroups0/512;
reasoning0/53. These await independent replays, NOT broad/final parity.
Selector both query slots348/348DEV and1551/1551actualTRAIN exposures; trueold
entitysignal285/285positiveDEV,none63missing. Input-head diagnosis only.
Metadata-only vectorization GPU passes integer fields/allgrad1.93e-6/cacheLP9.54e-6
testedargmax/sameKV/unchangedweights; body267.48->251.87ms, cachedB1 79.78->62.68ms,
B8 80.86->60.25ms; excludesAdam/E/IO/fullgeneration. NOT adopted by qualityeval.
FullgoalACTIVE/UNMET; native/originals/TEST unchanged; no growing/subagents.

# Earlier execution record (status superseded above)

AUTHORITATIVE trainer24899 TERMINAL0 at1000, ~.884updates/s, boundedpilotpaused.
DoNOTrestart/extendyet. CPUtrainingaudit55101 LIVE oncore10; firstlaunchwrapper
forgot sys.path tools and failedBEFOREexecution; correctedlaunchrunning. GPU
metadataoptimizationparity/speed45266 LIVE soleGPUworker, onlystartedaftertrainer
terminal. No fullqualityeval/selector/notebookyet. NeedtrainingauditPASSED then
runoriginalV2fullquality+notebook, CPUselector andindependentreplays; pairedcomparison
last. Allguardedsources/checkpoint immutable now. CurrentturnPROGRESS(actual1000+
preliminaryCPUoptimization); fullgoalACTIVE/UNMET.

# Earlier live execution record (status superseded above)

PreviousgoalturnPROGRESS(newmodel/data/objective/preflight/timing). Currentturn
PROGRESS: freshguardedexport/newtrainer/data+firstupdateindependentaudits, actual
boundedtraining andseparateCPUmetadataoptimization. FullgoalACTIVE/UNMET; noquality
resultyet. Native/originals untouched,no growing/subagents. Export26376/data75609
TERMINAL0/PASSED. Two newrunisolation/closuretests passed; directnative/core/data/
encoderutilities addedBEFOREexport. AllnewguardedsourcesnowFROZEN.

Run dump/native256-query-entity-unified-v1 FORMATnative256-query-entity-full-original-v1.
FreshnewV2seed47/23209216reader/Eseparate/raw64/globalold64:1/KVunchanged, original
48762updates/2epochs/361357104LMlabels schedule. Different6FFNunits/queryweights
fromSpan; DON'Tclaimidenticalcrossarchitecturetensors/RNG. DataauditinheritsALL
originalblocks/labels onlyviaunchangedinput/data-sourceSHAs, separatelyrechecksall
48762originalorders/labels/LR,newinitialtensors/RNG,allquery/source/surfacereceipts.
Actualfirst52451TERMINAL0; independent63264TERMINAL0/PASSED:7035LMlabels/3query
segments/6slots/3sourcesegments/6terms(3name3value),allAdam1. Resumedtrainer24899
LIVE soleGPUworker; bounded999NEWupdatestoward1000,lastobserved925/~.89updates/s.
Poll24899, neverduplicatefromstatusfiles/timeout. Noevaluation/otherGPUrunning.

tools/run_native256_query_entity.py(train/audit/evaluate/notebook/results),
train_native256_query_entity.py +verify_native256_query_entity_run.py andprotocol
docs/native256_query_entity_training_protocol.md areEXPORTED/FROZEN. No editmodel/
trainer/helper/evalwhilelive. Objective .05query+.02criticalsource frozen; noNULL.
NEXTaftertrainerterminal: audittraining, originalV2fullGPUeval/notebook+independent
CPU/publicreplays, headselectorCPU andmatchedcomparison. Noautomaticextension.
Expected1000LM7416257/query1551segments3102slots/source1241segments2482terms.

Supplementalread-onlyhelperscreated/syntaxchecked BEFOREqualityeval:
tools/evaluate_native256_query_selection.py (actualTRAINexposures+348DEV,inputheads
ONLY, CPUdefault12), tools/compare_native256_query_entity.py (requiresallQA/training/
notebook/selectorreceipts,sameprompt/exposures,ownfresharchidentity,pairedCIs).
Theyarenotpartoftrainingmanifest; theiroutputsbindownadditionalhelperSHA.

SeparatetoolONLYmetadataoptimization, NOTusedbytrainer: tools/native256_query_entity_
vectorized.py +test/protocol; prefix/scatterintegerwordkeys+broadcastroleheaders,
explicitreversibleprocess-localoverride, fixedglobalroletables(nohistory),noextraKV.
CPU61836TERMINAL0/PASSED:4newtests(+9importedV2=13), exactallWordfields/Scope/active
fullmodelloss+ALLgradients/cache/rollover/contextrestore. GPUhelper
tools/benchmark_native256_query_entity_vectorized.py prepared; doNOTstartuntil24899
TERMINAL. Itmustrecordruntimeoverride/sourceSHA,checktrainedGPUfields/grad/LP/argmax/
cachebytes/unchangedweights thenmeasure. NoGPUperformanceclaimfromCPUtests. Full
freegenerationparitynotyetchecked; originalV2qualityevalmustremainunchanged.
Status docdocs/native256_query_entity_training_status.md mutable. AlloldTEST/novel
studentoutputs stillNONE; finalfullbudget/strongrawcontrol/broadqualityoutstanding.

# Earlier: query-identity V2 PREPARATION VERIFIED; no corpus training (2026-10-18)

Previousvisibleturn onlyreportedpausedcompletedresults(no newexperimentalprogress).
Userresumedfullgoal. ThisturnPROGRESS:newseparatemodel/labels/allcaseaudits/objective/
GPUcapacity/equalbudgettiming. FullgoalACTIVE/UNMET, noquality/adoption/full2epoch
claim. No growing/subagents. Native256/originals untouched; ALL2,323 priorprotected
hashesverifiedunchanged bycombinedverifier24699 TERMINAL0. ALL workers terminal.

Modelsrc/models/token_memory_64_query_entity_v2 inheritsimmutableV1math andfixes
temporaryembeddingParameterregistrationininputs viaobject.__setattr__. V1 ordinary
zero-updategate60390PASSED butfirstauxiliarytestexposedregisteredalias; preserve
V1andV2both (V2dependsonV1). V2exactV1fresh47tensors/RNG; nooldtrainedweightsloaded.
23209216reader/E9426432separate/raw64/oneOLDglobalKVper64/noextraKV;6FFNhiddenunits
removed31374->31368tofund3072newqueryweights, +16384fixedglobalbufferbytes. CNN2
queryslots,last128causalsurfaceIDs,existingUserroleheaders, exactASCII<=8BPEword
keys/samesentenceentity+topicbias, excludecurrentUserbody, EOS/wordNULLmassconserved.
WordIDsareADDEDwordbitmetadata,notsemanticencoderlosslessness; literalcue/128scope/
aliases/multientitylimits. Noextraoldwordattention/KVbank. Noqualityclaim.

Data dump/native256-query-entity-labels-20261018 PREPARED91894 + ALLcaseindependent
30932TERMINAL0/PASSED:39270querylabels/62822criticalsource(31411name+31411value),
39270querysegments/31413sourcesegments.7859missingcasesqueryONLY,noNULLsource.
Alloriginalbytes/CSR/firstdivergencefields rederived. First1000originalupdates
7416257LMlabels/1551querysegments(3102slotlabels)/1241sourcesegments/2482terms.
Sourceslabelhelper/prepare/verifier/protocol/data immutable. No extraLMexamples.
Sourceonlyfirstdifferingfieldpiecesagainst2distractors,notgoldcontinuation.

Actualproductionallcasecoverage70106TERMINAL0/PASSED:bothquerywordseligibleALL
39270,correctoldentitykeyALL31411positives,ALL31413sourcepacketsvalidated. Untrained
selector18084/39270entity,17136/39270topicdiagnosticonly. OriginalcacheSHAverified.
src/protocol/tests/objective nowFROZENbyreceipts. No editingwhilefutureworkerlive.
Objective tools/native256_query_entity_objective.py:globaloriginalCE+.05queryCE+
.02criticalsourceCE,globalsegment/field/2layermeans. Auxh/embedding/oldpointer/norm
detached; AUXgradONLYselector/entitygates, presence/coreordinaryCE. FouractualCPU
testsPASSED63240 inclroute/micro/exactzeroauxAdam/8steps2toyepochresume. NineV2
invarianttestsPASSED3423 inclindependentmath+grad/causality/answerexclusion/cache/
realrollover/registration+exactfreshrepair (17totalincl8importedV1duplicates).

Actualfull21GPUgate6211TERMINAL0/PASSED under
dump/native256-query-entity-v2-preflight-capacity-verified-20261018/report.json.
CPU/GPUallclippedgradmax5.14090e-7, GPUglobal/micro1.86265e-8; weightsunchanged,
emptyAdam/zeroCorpusupdates. Preservedfailedzero-updategates:shortsourcetailslack
query; no<=128combinedquery/source; absolutefloat32normthreshold; nosource-bearing
raw512/m39. Diagnosis45234 +independentFP64norm28967TERMINAL0: trueCPU/GPU norm
relative1.72434e-7/rawgrad5.96046e-7/GPUrepeat1.19209e-7; CPUFP32normreduction
8.79e-6relativeerror. NewgateONLYnormrelative2e-5; retainabsloss/allgrad2e-5.
Capacity genuine [112342]+[217414]*7 batch8/m39/raw512,7query+14source terms,
median337.123ms/5.488GiBalloc/6.018reserved. All8source217414 originallym15 padded
INVALIDtom39 stress329.257ms/5.287alloc/5.773reserved, objectivepaddingdiff.0001221.
Includes2momentbuffers,excludesAdamstep/scratch/E/IO/eval/save. No speedwinclaim.

Equalbudgetbenchmark63387TERMINAL0/PASSED:
dump/native256-query-entity-v2-speed-20261018/report.json. BOTHdefault23209216,
samegenuineinputs,alternatingorder. BodySpan202.301ms/Query248.793(+22.98%);
cached1Span46.546ms/Query77.562(+66.64%),cached8 46.788/77.293(+65.20%). SameKV
bytes3441232(batch1)/27529856(batch8)at15old+1reservedslot/raw64;extraKV0.
Actualobjectivecapacitytimingdifferentinput/objective, DON'Tcompareitdirectlyto
ordinarybody. Preliminary219->256msV1sharedcorecontrolhad23206144params; equalbudget
timingaboveauthoritative. No endtoendtraining/generation speedorqualityclaim.

Combinedclosure+all2,323oldfilesverifier24699TERMINAL0/PASSED receipt
dump/native256-query-entity-preparation-verification-20261018/report.json.
Authoritativereport docs/native256_query_entity_preparation_results.md.
NOnewcorpustraining/export/checkpoint/trainercreatedyet. NEXTSAFEACTION: separately
guardedfresh47export/trainer/firstupdateauditor+boundedmatchedpilot(original48762/
361357104LMlabels/2ORIGINALepochs schedule), fullDEV/freefacts/missing/swaps/all-four/
legacy/historical/numeric+selectorchecks. AllqualityUNPROVEN,costmaterial. Noresume
ofsource-occurrencefailedpilot oroldnative. Fullfreshfinal/rawcontrol/TEST/novel-
literalstudentscoringstilloutstanding; reservedTEST onlyafteractual48762exposures.

# Earlier: source-occurrence1,000 VERIFIED failure; ALL workers TERMINAL (2026-10-18)

Previousgoalturn PROGRESS (newobjective/data/gates/firsttraining). Currentgoalturn
PROGRESS: completed/audited1,000,fullqualityeval+independentreplays+pairedcomparison,
conditionalteachermechanism+coefficientaudits andreservedfinaltestinputpreparation.
FullgoalACTIVE/UNMET, nofull2epoch/parity/adoptionclaim. NO LIVE CPU/GPU workers.
Trainer71460/audit12672/GPUeval60648/notebookGPU19417/aggregate+public20602/
notebookCPU93306/pairedcomparison2125 ALL TERMINAL0/PASSED. Teacher26419 and
independentcache13410 TERMINAL0/PASSED. Allfinalinputworkers terminal0. DoNOT
restart/extendcompletedpilotautomatically. Source/checkpoints/run immutable.

CANONICAL failedpilot dump/native256-source-occurrence-unified-v2, fresh47 actual
1000updates/32000originalsegments/7416257LMlabels/allAdam1000/orders/LR/cursor/
recipe/exposures checked. Aux1553segments=1243pos+310missing,16834terms; Source
reader23209216/frozenE9426432separate/raw64/globalold64:1/noextraKVorinfops.
Checkpoint SHA75649d532db43d44501abdcdb2c6efeddf3d2c60ed7398be495cedb37001e5b6.
Everyprotectedearlierfileunchanged, native256untouched. No growing/subagents.

VERIFIED: language42.651722%/CE3.178110542 vsordinarySpan44.164101%/3.062043972;
delta-1.512379pp fixed-doc95CI[-1.559751,-1.464757]. Positive20/285vs23,delta
-1.052632ppCI[-4.912281,2.807018];missing0/63vs13 delta-20.634921ppCI[-30.158730,
-11.111111]. Changedvalue5/64vs3,query4/64vs1,bothvalue3/64vs2,query0/64BOTH.
Notebook full15/256 BOTH, properprefix76/256vs77;bothswapsfull1/128vs2,prefix
10/128vs11;ALLFOUR0/64 BOTHformats/arms inclallrawName/wholeQuery/noRollstrata.
Prefixsamevalueacrossquestions119/128vs121;bothquestionscorrect0/128vs1. Full
samevalue86/128. 64blocked swappedpairs identical,blockedstrict0. All7legacy0/512.
Historicalpositives7/103vs3,changed2/32vs0,missing0/25vs8 (isolatedNOTbroadwin).
Numeric0/53/EOS2vs4 cap256; recentcolor2/8noEOS notstrict. Naturalcompletion
1/477 hasknowntrailing-spaceprefixconfound, don'tclaimitsgainasrecovery.
60uncachedpublicCPU +32notebookCPU replaysmatch,allscores/sourceschecked. Final
comparison dump/native256-source-occurrence-unified-v2/comparison-1000/report.json
also binds actualfull21CPU+fullshapeGPUpreflightproof inclsupplementalhelpers.
Report docs/native256_source_occurrence_results.md is authoritative.

Conditionalteacher/probability+actualweightdiagnoses verified asbelow; Copygate
alreadyhigh, firstNamefalls andgenericNULLpenalizesvalidIcopy. DoNOTincreaseflat
sourcealignmentlossorclaim23Mimpossible. New hypothesis ONLY(unimplemented):
docs/native256_query_entity_reader_hypothesis.md. A small queryentity/topic
selector overcausal128surfaceIDs fromexistingmetadata/raw64, exactorderedword
matching+samesentencebias excludingcurrentquery, EOS/word-copyNULLconservation,
confidence/presencecue, extraauxgradientsONLYnewpath(oldpointer/h/E detached),
criticalfirstdiff source labels+queryslotlabels(originaldataonly). Propose3072
newparams /remove6FFNhiddenunits31374->31368 exact23M, sameKV/global64:1. Flags/
IDF/rolemodelbuffersglobal(notKV) mustcountmemory. Hardmatching isliteralcue not
generalrelationtruth; testbroadaliases/longwords/multientitylimitations, noadoption
fromnarrowwin. No newmodel/code/runstarted. Fixcoefficientsprospectively, CPU
reference/causality/cache/rollover/optimizer/budget/benchmark BEFOREtraining.

RESERVED FINAL suite completeINPUTS only:1664cases224units/835codeblocks/707nonce
identifiersabsentwholeOriginalTRAINtext+BPE; allcaseauditpassed,CPUall835exact0,
GPUall835max1.907349e-5. Data dump/native256-novel-literals-test-20261018 and
encoder-inputs; sources/caches immutable. Original fullTESTinputs alsoverified.
NOstudenttestoutputs. Mustimplementstudentfinalevaluators withactual48762/
361357104labels/2ORIGINALepochs gate before fullTEST/novelliteralscoring. Same
freshfull-budgetuncompressed/strongbaselinecomparison+efficiency goaloutstanding.
See docs/native256_novel_literals_test_input_results.md. GoalACTIVE/UNMET.

# Earlier same-turn execution history (latest status superseded above)

AUTHORITATIVE:60648 fullqualityGPU TERMINAL0/complete. NotebookGPU19417 LIVE;
CPUaggregate+publicreplay20602 LIVE oncore12, scopedV2 helperfunctions. Noother
GPU. Teacherdiag26419 + independentcachedreplay13410 bothTERMINAL0/PASSED;9
selectedcriticalcachedreplayspermodel andALLsavedalgebra/categorymeanschecked.
Weight-audit tool alsoTERMINAL0. Reservedfinaltestinputworkers allTERMINAL0.
Poll19417/20602; afternotebookterminal0 run itsCPUverifier oncore10. Needboth
independentreceipts thencompare_native256_source_occurrence.py --step1000.
DoNOTautomaticallyextend/adoptthispilot. Allfrozenmodel/train/eval sourcesunchanged.

FullGPU provisional:42.651722%/CE3.178110542 vscontrol44.164101%/3.062043972;
positive20/285vs23,missing0/63vs13,changedvalue5/64vs3,query4/64vs1,bothvalue3/64
vs2,query0/64BOTH;blocked0/64. All7legacygroups0/512;stillneedsindependentreplays.
Naturalcompletion1/477 hasknownprefixtrailing-spaceconfound, NOT notebookproper
completion. Historical/reasoningfullreportnowexists, parseactualfiles later.

INDEPENDENT VERIFIED conditionalteacherdiagnosis (NOTfreeQA): firstNameLMcorrect
81/285(28.42%)->63/285(22.11%),sourceTop1.29474->.25965; sourcep.24224->.19979,
NULLp.0000208->.08760. FirstColorLM115/285->131/285,sourceTop1.30175->.34737;
allValueLM.92209->.92659,sourceCE.54804->.47704 butsourceTop1.73471->.72859.
MissingfirstI LM13/63->0/63;NULLCE13.94254->2.66346/NULLp.00000125->.09556,
butcorrectIcopycontribution.21108->.00340; genI.0000686->.002066. ThuslowerNULL
sourceCE doesNOTmeanunknownrecovery; genericword-copyNULL suppresseslegitimate
I-from-ackcopy. Copygatealreadyhigh(firstName.902->.873,firstColor>.996), so
insufficientcopygate isNOTsupportedasmainbottleneck. LatercolorsuseGOLDName.
AllDEVfirstsubstantiveName andI arebodyoffset0 (directactualtokenizationcheck),
so whitespace-phaseconfound hypothesis was REJECTED forcurrentDEV, not adopted.

Exactactualobjectivecoefficientaudit dump/native256-source-occurrence-weight-audit
-1000/report.json:779batcheswithaux; NameFirst1241exposures (2positiveonlytails
lackNameFirst) vs310NULL. Layer-averagedsourceCE totalcoeff NameFirst4.770684,
NULL15.366667,laterpositive57.762649;NULL/NameFirst3.221061 vsLMfirstword.24805.
ThisisSCALARCOEFFICIENTSnotgradientnorm/causal/convergenceproof. Per-positive-piece
averaginggivesfirstbindingdecisionsweakrelativeweight;NULLsemanticisnoWORDcopy,
notnecessarilyabsenceofrecord. No nexttraining/modelchangeimplemented yet.
Currentturn PROGRESS inclcompletedtraining/diagnosis+reservedfinalinputs;fullgoal
ACTIVE/UNMET, fresh2originalepochs/strongbaseline/finaltestquality stilloutstanding.

# Earlier current-turn progress (current worker/status superseded above)

AUTHORITATIVE currentworkers: GPUfullevaluation60648 LIVE, CPUread-onlyconditional
source/mixture diagnosis26419 LIVE oncore10. Alltraining/source-data-audit/reserved
test-input workers TERMINAL0. DoNOTstartanotherGPUworker. FullGPUprovisionaldev
793061/1859388=42.651722%,CE3.178110542,vscompletedcontrol44.164101%/3.062044;
languagecost~1.51238pp, notyetindependentlyqualityverified. Memoryteacher92.7987%
vscontrol95.1816% too; neitherteacheraccuracyisfreefacts. Naturalgenrunning; no
completefactualreportyet. MUSTfullGPUterminal -> notebookGPU -> independent
aggregate/publicCPU/notebookCPU -> matchedcomparison. Noadoption/extendfromloss.

Teacherdiag new tools/diagnose_native256_source_occurrence_teacher.py, prospective
docs/native256_source_occurrence_teacher_protocol.md. All348originalDEVmemory
dialogues/349segments, firstName285,firstColor285,missing63; conditionalold/new
pointerSOURCEoccurrence vsaggregatedwordmass andfinalLMcopy/generatorcontributions.
Nooptimizer/testscoring; laterpieces/colorshaveGOLDearlieranswer/Name context,
soNOTfreeanswer/bindingevidence. Sources/model/checkpointweightsread-only; output
dump/native256-source-occurrence-teacher-diagnosis-1000. Poll26419, don'tduplicate.

RESERVED FINAL NOVEL INPUTS nowfullyCPU/GPUencoded/audited: all1664cases/224units,
835genuinefull64blocks,707novelwholeword/BPEidentifiers (TRAINabsenceproved),
CPUindepreencoding835exact0/GPUall835max1.907349e-5. Prep34315/caseaudit28121/
codeprep42714/CPUcodeaudit93997 andGPUcodeaudit TERMINAL0. No studentTESToutputs.
Allnewreserved-case/code sources frozen via their manifests; doNOTedit. Original
traininputSHAs matchcanonicalsource-labelinputSHA. Laterstudentteststillrequires
actual48762/361357104labels. Encoderexportloadexplicitlycheckscheckpoint-embedded
architecture_sources; numericdependencyhashesmatchedcanonical11localsrcmodules.
No originalsrc/run/cache changes. GoalACTIVE/UNMET; currentturn PROGRESS.

# Earlier same-turn records (current worker/status superseded above)

Previousgoalturn PROGRESS (implemented/audited/launched). Currentturn PROGRESS:
trainer71460 TERMINAL0/PAUSED1000; trainingaudit12672 TERMINAL0/PASSED actual
7416257LMlabels/32000originalsegments/allAdam1000/exactorders/LR/cursor/recipe/
source/earlierguards. Auxactual1553segments=1243positive+310missing,16834terms,
allper-segment/globalmicrobatchdenominatorsverified. No extraoriginalepoch.
NewFULL GPU evaluation launched runnerV2 evaluate --step1000 (handletooloutput),
ONLY GPUworker; allothers terminal. NeedfullGPUterminal -> notebookGPU -> all
independentaggregate/publicCPU/notebookCPU -> pairedcomparison BEFOREqualityclaim.
Neverresume/extendthispilotfromlossalone. Alltraining/model/eval/helper/protocol
sourcesFROZEN. Native256/oldcompletedruns untouched, goalACTIVE/UNMET.

Whiletrainingran, prepared RESERVED FINAL novel-literal test, NOstudentoutputs:
tools/prepare_native256_novel_literals_test.py +verify helper,3CPUtestspass.
Prospective docs/native256_novel_literals_test_protocol.md. Data
dump/native256-novel-literals-test-20261018:224units/1664cases,known/novelliterals/
novelnames+literals each64four-wayunits(full+propernameprefix);32missingfullunits.
ALLquestionsraw, samequeryphase32,targetfitsnoRoll, matchedsourcewordbag/coordinates/
length/prefixacrossswaps. Arbitraryqz/Qzidentifiersinoriginalnotebookgrammar,
NOTnewtokenizervocabulary;707selectedidentifiers provenabsent ENTIREoriginalTRAIN
wholewordtextandactualBPEbytes. No student/decoder/successfilter;0structuralrejects.
Prep34315 +indepallcase28121 TERMINAL0/PASSED. CanonicalinputSHA matchesoriginal
source-labeldatasetoriginalinputSHA (directcheck). Cases/protocol/helpersnowFROZEN.
Dedup835raw64blocks; CPUsemanticprep42714,independentCPU93997 TERMINAL0/PASSED
(835/835exacterror0),fullGPUinputagreementalsoTERMINAL0 (all835,max1.907349e-5).
Inputcache encoder-inputs/manifest +independent-cpu/cuda-verification.json;
E9426432frozenexact,no updates/literalpackinglaterRAM-only. EncoderCodec.load_encoder
checkscheckpoint-embeddedarchitecture_sources;numericdependencyhashesalso11local
srcmodules matchedcurrentcanonicalprotectedmap; storage/settingsutilitymodules
notinthatmap, no numeric-model change made. Nooriginaldata/cache/source edits.
RESERVEDsuite may ONLYscorestudentactual48762/361357104labels+trainingaudit.
OriginalfullTESTinputcachealsoalreadyverified. FinalstudentTEST/novelliteral
evaluator stillnotimplemented; no earlytestmetrics/noadoptionclaim.

# Earlier current-turn training state (superseded above)

AUTHORITATIVE LIVE handle71460, runnerV2 train --resume --microbatch8 --stop-after
999 --skip-initial-eval. Lastconfirmedoutputstep175/48762,~1.258updates/s at
02:21UTC. Stop target1000TOTAL (999NEWafteraudited1), original2epoch48762schedule.
DoNOTduplicate/restart/extendfromstatusfileorlogalone; POLL71460 FIRST. Trainer
willpauseat1000. Needactualtrainingaudit, fullGPUevaluation+notebook, independent
aggregate/publicCPU/notebookCPU andmatchedcomparisonbeforequalityclaim. GPUworker
only; allothercurrentworkers TERMINAL. Native256pausedoriginal/oldcompletedruns
unchanged. This goalturn PROGRESS: implemented+all-label-auditedobjective, numerical
andcapacitygatespassed, actualfirststepaudited, boundednewtrainingstarted. Full
goalACTIVE/UNMET, noquality/adoption/full2epochclaim. No subagents/growing.

Current canonicalrun dump/native256-source-occurrence-unified-v2, uniqueformat
native256-source-occurrence-full-original-v2. Driver tools/run_native256_source_
occurrence_v2.py. SAME inference model Span64/23209216params/frozenE9426432/raw64/
oneglobalKVper64oldBPE; nonewinferenceops/cache. New objective globaloriginalCE +
.1mean-per-originalSEGMENTsource-occurrenceCE(avg2readlayers), auxH/embedding
DETACHED. Originaldata/exposures/orders/LR/FP32Adamunchanged; extraDERIVEDlabels
explicit. LiteralIDsstillADDEDmetadata, notoldsemanticencoderlosslessness.

All NEW run/model/helper/data/protocol/test sources FROZEN; doNOTeditwhiletrain/
eval live. Supplementalcapacityhelper/protocol/finalizer notinmodelsource97hash
map butBOUNDby canonicalpreflight.json and post-comparison inputguards. Mustnot
rerunCPUverify-gpu/finalizer now (wouldoverwriteboundreceipt or rejectlivefolder).
Allgates alreadydone: 5CPUobjective tests+2comparison tests PASS; ALL39270original
TRAIN memorydialogues/423848terms independentlyaudited; originalfullcache/data
proof transferred by exactSHA plusALL39294actualauxpackets rechecked inV2.
V1zero-updateGPUbitwiseassertionfailurepreserved; diagnosisordinaryrepeat3.725e-9
vszeroaux5.588e-9, exactloss/norm/weights/emptyAdam. CorrectedGPU1e-7gradientgate
passed. Full21CPUgrad3.017485e-7/objective3.33786e-7; fullcapacityB8/raw512/m39/
119terms peak5.136GBalloc/6.455GBreservedincl2momentbuffers(noAdamstep/scratch).
Allprotectedearlierfilesunchanged. Firstactual7035labels/40aux/3positive/Adam1
independentlyPASS; initialLMCE EXACT control7.8881953954696655. NoNULLfirstbatch.

On71460terminal0: runnerV2 audit --audit training, then evaluate --step1000
(GPU); afterterminal notebook --step1000(GPU). CanCPUaggregate/public audit while
notebookGPU works ONLYonimmutablefullevaloutputs; separateCPUaffinitycore10 toavoid
hostcontention(defaultGPUworkerpin_cpu15). Finalresults action includesaggregate/
public/notebookCPU sequential; orcallfunctionsinsideV2scope onCPU. Then
compare_native256_source_occurrence.py --step1000 (post-analysis, nottrain source).
Thischecks SAME actualfreshinitial/orders/LMlabels/prompts and allindependentaudits
butexplicitlydifferentauxsupervision/newlabelinputs; pairedfixedcheckpointCIsnot
seeduncertainty. Comparison also binds supplementalpreflight hashes. Optional
choice-analysis use existing choices(cases,outputs) withnewroot, don'trerunold
main which is hardcodedoldroots. Docs/native256_source_occurrence_status.md is
currentreport. OriginalTESTcodes independentlyverified already; no studentTEST
metrics beforeactual48762/361357104labels, finalnovelliteraltest stillpending.

# Earlier current-turn training progress (superseded above)

Firstindependentaudit66592 TERMINAL0/PASSED: allAdam1,7035LMlabels,3positive
alignedsegments/40terms, actualsource/cursor/recipe/order/auxnormalization andALL
earlierfileschecked. Newresume launched runnerV2 train --resume --microbatch8
--stop-after999 --skip-initial-eval, handlein tooloutput. This adds999NEWupdates
toownstep1 ->1000, original48762schedule retained; nooldpilotextended. Pollworker
authoritativehandle, no duplicate. Allsource/helper/protocolsFROZEN. Mustfinish
trainer -> audit --audit training -> fullGPUevaluate --step1000 -> notebook --step
1000 -> results --step1000 (CPUaggregate/public/notebook, maysplitviafunctions)
-> compare_native256_source_occurrence.py --step1000. Allqualityunprovenuntil
these finish. Source0.1training-onlychanged; model/inference/KVsameSpan64. Native
andcompletedpilots untouched, goalACTIVE/UNMET. Supplementalfullpreflightbound.

# Earlier first-update progress (current worker superseded above)

Firsttrainer64322 TERMINAL0, pausedactual1/48762: CE7.88820, alignment6.86702;
actual7035LMlabels/3positivealignedsegments/40auxterms/noNULLinfirstbatch. Initial
LMCE7.8881953954696655 is EXACT samecompletedSpancontrol firstbatch (directcheck),
sample_indices/targets/LR/cursor identical. No later-stepqualitycomparison yet.
Independentfirstaudit launched runnerV2 audit --audit first, session66592 LIVE.
MustfinishPASSED BEFOREresume999. NoGPUworker now.
Allfrozenmodel/helper/protocol/checkpoint/data sources unchanged. Combined
preflight.json[all_training_preflight_gates_passed]true, independentfull21CPU
gradientmax3.017485e-7/capacity actualB8/raw512/memory39/119terms bothPASSED.
FirstauditchecksallAdam1/original7035labels/cursor/auxrowweights/orders/sources/
earlierfileguards. CurrentgoalACTIVE/UNMET, noqualityclaim. Native256 untouched.

# Earlier same-turn preflight progress (current worker/status superseded above)

IndependentCPU74928 TERMINAL0/PASSED: full21layer actualobjective error3.33786e-7,
all23Mparamgradientmax3.017485e-7, weights/noAdam/ALLprotectedearlierfilesunchanged.
Finalizer44876 TERMINAL0/PASSED: preflight.json binds independentnumericalreceipt
ANDsupplementalactualB8/raw512/memory39/119termcapacity receipt + helper/protocolSHA.
Canonicalpreflight[all_training_preflight_gates_passed]=true. Original smalltail
GPUcapacitysample is superseded forcapacityonly, numericalproofretainedunchanged.
Newfirstactualtrainer launched runnerV2 train --microbatch8 --stop-after1
--skip-initial-eval, handle in tooloutput; pollthatworker BEFOREfirstaudit. No
otherGPUworker. Mustauditactual1 before resuming999. Noqualityresultyet.

# Earlier same-turn gates (current worker/status superseded above)

CURRENT: V2dataaudit72328 TERMINAL0/PASSED. V2GPUpreflight30944 TERMINAL0/PASSED:
coefficient0numericalgate passes, microgradmax2.607703e-8, weights/emptyAdamexact.
Its original "heavy" positive selector picked largestdocumentend butSHORTtail,
so its small capacitysample is NOT fullshape. Preservedunchanged; no adoptionclaim.
New supplemental capacity protocol/helper (outside frozen training sources) tests
mixed genuine [one39codecorpussegment + seven17termpositive segments], actual
B8/raw512/memory39/119terms. Worker72935 TERMINAL0/PASSED, weights/emptyAdamexact.
Ordinarybody samples.22536/.22368/.23076s, alignment.22573/.24188/.22914s;
peakallocated5135798784 vs5136044544,reserved6452936704 vs6455033856.
Includes2moment-sizedbuffers, noAdamstep/scratch/IO/eval/save, NOspeedclaim.
docs/native256_source_occurrence_capacity_protocol.md, helper+receiptSHA bindthis.

ONLY LIVE worker is full21layer independentCPU objective/allparameter-gradient
gate, launched runnerV2 verify-gpu, session74928 LIVE. Afterterminal0 MUST run
finalize_native256_source_occurrence_preflight.py
to bind independentnumerical+actual512/39capacityreceipts BEFOREtrain update1.
No optimization yet. Supplemental helpers createoutputs only; no frozen source
or earliercheckpoint is changed. Then train1/auditfirst/resume999 as below.

# Earlier numerical/export progress (current worker superseded above)

AUTHORITATIVE update: v1prepare27932/dataaudit99719 TERMINAL0/PASSED. GPUpreflight
90826 TERMINAL1 at BITWISE coefficient0gradientassertion, BEFORE ANY update.
New zero-update diagnosis61913 TERMINAL0: repeatedordinaryCUDAgradientsdiffer
3.725290e-9; zeroauxvsordinary5.587935e-9. Loss/normexactall8passes; weights/empty
Adam exact. Receipt dump/native256-source-occurrence-parity-20261018. V1export
andfailure.json preservedimmutable. NO model/objective/math/source fix needed.

Corrected NUMERICAL-GATE export (notarchitecturevariant) uses new
tools/run_native256_source_occurrence_v2.py, run
dump/native256-source-occurrence-unified-v2, uniqueformat...original-v2.
Prospective docs/native256_source_occurrence_cuda_protocol.md preservesCPUexact
zero-coefficientAdamparity; GPUgradient tolerance1e-7 WITHactualerrorlogged and
exactloss/norm. Independentfull21layerCPU/allparamgate/micro/causal/NULL/weights/
emptyAdamchecks unchanged. New preflight_native256_source_occurrence_v2.py only
changes thatassertion/receipt; oldprefight/sourcefiles NOTedited.

V2prepare22323 TERMINAL0, exactfreshinitial/RNG/orders, protectedV1+diagnosis SHA.
V2data-proofaudit72328 LIVE (CPU); pollsamehandle. It verifies ALL unchanged
input/cache/sourceSHA, inherits immutableV1full5million-block/label audit with
explicitprovenance, independentlyrechecksactualfresh/2orders andEVERY39294aux
productionpacket. DoNOT duplicate fullunchangedcache audit. NoGPUworker now.
AllnewV2source/protocolsFROZEN as exportprepared. ComparisontoolnowpointsV2;
2CPUcomparisontestsrepassed afterimportchange. Next afterauditterminal0:
runnerV2 preflight -> runnerV2 verify-gpu (CPUrequiredwritespreflight.json) ->
train --stop-after1 --skip-initial-eval -> independentfirstaudit -> resume999.
Thenactualtrainingaudit/fullGPUeval+notebook/independentresults+pairedcomparison.
No training/adoption/qualitywin yet. Native256/oldruns untouched; goalACTIVE/UNMET.

# Earlier setup this turn (current worker/status superseded above)

Previous goalturn PROGRESS: completed span/raw comparisons and original TESTinput
audits. GoalACTIVE/UNMET. This turn implements training-only SOURCE occurrence CE
for the SAME pooled-key architecture/inference/KV,23Mreader/Eseparate/raw64/old64:1.
No subagents/growing/native/original edits. Old completed pilots remain immutable.

Protocol docs/native256_source_occurrence_protocol.md written BEFORE preparation.
New native256_source_occurrence.py +prepare/independent-data-audit helpers and
5CPUtests PASS: coefficient0 exactordinaryAdam, independentmean-embedding actual
objective/allparameter gradients/detachedcore, micro/globalnormalization/future/
padding/NULL guards, serialized8update2toyepochAdamresume andgenuineoriginalbytes.
Firsttest failure was FLOAT LOGGING conversion only; fixed before export, rerunall5.

Data dump/native256-source-occurrence-data-20261018 PREPARED40067terminal0,
independentaudit39708terminal0 PASSED ALL39270originalTRAINmemorydialogues;
423848terms=415989positiveName/valueBPE+7859NULLfirstsubstantivemissingtokens.
39294alignedORIGINALsegments (24answercrossings); noaddedQAexamples/data/epochs.
First1000same7416257LMlabels,1553alignedsegments/16834auxterms. Everytarget/source
IDmatchesoriginalbytes andsourceisalreadyvisible. Auditorindependentlyparsedall
questions+sentences+answers; allCSR/source/inputhasheschecked. These data helpers,
protocol+tests are now FROZEN via data manifest. No decoder/encodertraining.

New run dump/native256-source-occurrence-unified-v1 via
tools/run_native256_source_occurrence.py. Preparation27932 TERMINAL0, all actual
freshinitial/RNG/orders exact control+emptyAdam, earlierrun guards unchanged.
Production-packet/full-originaldata audit99719 LIVE. Poll99719, don't duplicate.
New run/model/trainer/preflight/audit
sources FROZEN as export is live. Sourcescope/exceptionrestoration syntaxcheckPASS.
No GPUworker. CUDA memoryprobe6684terminal0 (11.57GBfree/12.82GBtotal), noPythonGPU
worker in inventory; no other app/process modified. All earlierworkers terminal.

Objective originalglobalLMCE +.1*meanSEGMENTsourceCE, average2readlayers; actual
sourceposition vsaggregatedwordID. AuxiliaryqueryH/sharedembeddingDETACHED;
ordinaryCEupdatesallweights. Pos/missingper-segmentmeans/countslogged, global
denominatoracrossmicrobatches. Inference model is EXACT completedSpan (no new
productionparams/operations/KV); literalIDsremainADDEDmetadata,notcodec lossless.
Fresh47/exactcontrolinitial+orders/freshAdam/original48762schedule, firstpilot1000.
Mustfinishprepare -> runner audit --audit data -> runner preflight -> independent
verify_native256_source_occurrence_gpu.py (CPU, writes requiredpreflight.json) ->
train --stop-after1 --skip-initial-eval -> independentfirstaudit -> resume999.
Thenactualtrainingaudit/fullGPUeval+notebook/independentaggregate+public+notebook/
pairedcomparison. DoNOTtrain before gates, anddoNOTextendoldcontrol.

New post-analysis compare_native256_source_occurrence.py syntax+2CPUtestsPASS;
actualcomparison notrunyet. docs/native256_source_occurrence_status.md records
auditeddata/code/setup, notquality. It explicitlyallows extraDERIVEDsupervision
andnewlabelinputs butchecks SAME original
data/exposures/LR/initial/actualorders/prompts and allprioraudits. Noqualityclaim.
Full two-original-epoch/strongreference/originalTEST/novelliteralgoal still unmet.

# Previous goal turn: span1,000 and original TEST inputs VERIFIED; ALL workers TERMINAL (2026-10-18)

This goal turn PROGRESS: new span fresh1,000 trained/evaluated/independentlyverified,
matched comparisons against oldlexical/raw saved, query-invariance diagnosis, full
original TESTinputs prepared+GPUagreement. GoalACTIVE/UNMET, noadoption/2epochclaim.
No subagents/growing/native/source/cache modifications. DoNOTextend/restartthis
completed spanpilot automatically. Its checkpoint/records must remain immutable.

All student workers TERMINAL0: span trainer9299,trainingaudit29521,fulleval94813,
aggregate4873,public99538,notebookGPU98039,notebookCPU30825,3armcomparison20331.
Raw eval/verifiers/comparison alsoallterminal0. NO LIVE GPU or CPU worker now.
Original TEST ALL-prefix encoder audit66169 FINISHED0/PASSED: all48459blocks
independently rederived/reencoded, exactCPUerror0, all7646rows/1757421labels,
frozenEncoderweights/buffers and allsource/input/cachehashes unchanged. Input-only,
zero studenttestpredictions. Do NOT duplicate/restart any completedworker.

Span1,000 exact7416257originalLMlabels/32000segments/allAdam1000/orders/LR/cursor
verified; sameactualfresh47NNstate/RNG andsourceinputs asoldLexical, original48762
schedule.23209216trainableReader/E9426432separatefrozen/1globalKVper64oldtokens.
SHAa9d853aa45c3d9e3831d6a5e2b0b61f6da907317f04a11180db793c2d00244a8.
All original/live helpers/model/protocol/test sources FROZEN; no edits needed.

VERIFIED span-vs-oldLex at1,000: language44.164101 vs44.136780%,CE3.062044 vs
3.065599;delta+.027321pp,fixeddoc95CI[-.011717,.064771] (notseeduncertainty).
Positive23/285 vs21,delta+.701754ppCI[-2.807018,3.859649];missing13/63vs0.
Changedvalue3/64vs3,query1/64vs3;bothvalue2/64vs1,query0/64BOTH. Blocked0/64.
All7legacygroups0/512. Historicalpositive3/103vs8,changed0/32vs3,missing8/25vs1;
numeric0/53/EOS4,cap256. Recentcolor3/8hasZEROEOS(notstrictrecovery).

ControlledNotebook span vsold: FULL15/256vs22,bothswaps2/128vs1; PREFIX77/256vs78,
bothswaps11/128vs1 (delta7.8125pp unitCI[3.90625,12.5]);ALLFOUR0/64 BOTHformats
andallraw-name/whole-query/no-rollstrata. Prefixanswersidenticalacrossquestion
changes121/128vs120; sameUNIQUEsourceowner107vs117. Bothquestionscorrectsame
memory1/128 BOTH. ThusstrongerswapreadingforonequestioncanbeRECORDSHORTCUT,not
questionbinding. Query q0/q1 successpairs6/5,recordordinal0/1/2=6/4/1. FullName
firsttoken50/256vs78. Blockedexact0,all64matchedswapoutputsidentical(CPUverified).

Evidence dump/native256-lexical-span-unified-v2/comparison-1000/report.json;
each eval independentaggregate/public/notebook receipts,60public+32notebookCPU
replaysmatch. New analyze_native256_notebook_choices.py +2CPUtestspass including
FIXED-FIRST-RECORDshortcut passingone swap butzeroqueries/allfour; output
eval/notebook-binding/choice-analysis.json(descriptiveonly). 3armcomparetool3CPU
testspass; actual source/checkpoint/fresh/Adam/exposure/prompt guardsallpassed.
Report docs/native256_lexical_span_results.md updated(noadoption). Rawvsoldlex
matchedcomparisonalreadyverified+10.505553pp languageCI[10.202668,10.811576],
butrawpositive0/missing59 andallfour0 too. Strongfinalmixed51261.013% retained.

Cost unchanged: pooling adds~18%batch8bodytime/+703.39MiBtemporarypeak, noKV
entries; wholecache25.23xsmallerthanraw, OLD64x. Single-token latencystillworse
thanraw; no end-to-end speedwin. LiteralIDsADDEDmetadata, notoldElosslessness.

NEXT research NOTIMPLEMENTED: source-occurrence alignment fromoriginalTRAIN
metadata(history/answer), keepingordinaryLMCEandoriginaldata/orders/2epochbudget.
CopyCEsumsallduplicatewordoccurrences, sohighteacherwordaccuracydoesnotforce
correctowner; common'in/the'positionsprovideweakbindingsignal. Sparsecorrect
sourceName/value positions +NULLmissingdecisioncouldforcebinding. Startwith
extra gradients confinedtoReader(queryH/sharedembeddingdetached); this doesNOT
guaranteelanguagepreservation. Need prospectiveprotocol/data+independentALLcase
audit,coefficient0exactoldparity/micro/allparam/future/resumechecks,zeroupdateGPU
andactualfirstauditBEFOREtraining. Reusecompletedspan1,000 aswordCEcontrol;
doNOTretrainit orchangeitsinference. See hypothesis MD, not an adoption claim.
Directraw-query-token summary/copy isanotherUNTESTEDconditionalfollowup; don't
silentlycombinechanges. Original full2epoch/TEST/heldliteral/strongbaselinegoal
stillrequired; no universalcapacity/impossibility claim fromtheseearlytests.

TEST INPUTS: prepare46144terminal0(7646rows/1757421labels/48459codes),encoderCPU
1490.9s; all original data and frozenEweights unchanged. IndependentCPU66169PASSED
reencodesALLblocks(groups127vs256) fromoriginalbytes, maxerrorEXACT0. Receipt
dump/native256-original-test-inputs-20261018/independent-cpu-verification.json.
New GPUinput-only helper
verify_native256_original_test_gpu.py completed0:1165structuraluniform/per-source
blocks, maxCPUcache-GPU2.288819e-5,Encoderstateexact/no updates/no studentmetrics.
OriginalTESTstudent evaluator NOTimplemented/reservedactual48762+361357104labels.
Same-format genuinelynewliteral-value test also stillneeded forfinalacceptance.
See docs/native256_original_test_input_results.md. This audit adds noqualityclaim.

# Earlier same-turn execution history (latest status superseded above)

PreviousgoalturnPROGRESS(prepared verified span). This turn PROGRESS: actual fresh
span1,000 trained/audited, raw full/notebook evaluation+independent audits finished,
matched raw-vs-lexical report saved, original TEST inputs prepared without student
predictions. Full23Mreader/Eseparate/64:1/2ORIGINALepochs/broadgoal ACTIVE/UNMET.
No subagents/growing/native changes. Do NOT adopt or extend from early language.

ONLY LIVE GPU: span full evaluation94813, via run_native256_lexical_span_full_v2.py
evaluate --step1000. Fresh trainer9299 FINISHED0/PAUSED1000. Trainingaudit29521
FINISHED0/PASSED: allAdam1000/7416257labels/exactoriginalorders/LR/targets/cursor
and guarded earlier files unchanged. Span SHAa9d853aa45c3d9e3831d6a5e2b0b61f6da907317f04a11180db793c2d00244a8.
Actual preflight81718 FINISHED0: micro8 gradient5.588e-9, weights/Adamunchanged,
peak5332224000allocated/6381633536reserved inclmomentbuffers(noAdamstep/scratch).
Firsttrain1042 andaudit60649 FINISHED0, exactfirst7035labels/allAdam1; then999NEW
updates, original48762schedule retained. This is NOT two original epochs done.
All span model/trainer/protocol/test sources remain FROZEN; don't edit whileeval.

SPAN PROVISIONAL full-dev821182/1859388=44.1641013%,CE3.062043972 vs oldlexical
44.1367805/3.065599038: +508hits/+0.027321pp, not meaningfulbinding proof.
Saved natural(caseinputsidentical): positive23/285,missing13/63,changedvalue3/64,
query1/64,bothvalue2/64,query0/64;falseunknown45/285,blocked0/64(EOS36). All7legacy
groups0/512 each. Stillneedsfullreport, notebookGPU,aggregate/publicCPU/notebook
audits andpairedcomparison BEFORE any quality claim. Wait94813 terminal then
run span notebook via V2 driver notebook --step1000, CPU results action (covers
aggregate+public+notebook), then new paired span comparison. Don't duplicateGPU.

RAW COMPLETE: fullGPU75351/notebookGPU46723/aggregate35749/publicCPU93566/notebook
CPU61468 ALL FINISHED0/PASSED. 60public uncachedCPU replays and32notebook replays
match; all64blocked swapped question/format pairs identical. Raw1,000 full-dev
33.631227%/CE3.694164868;positive0/285,missing59/63,changed0/64both,paired0;
falseunknown280/285. Properprefix/full notebook BOTH0/256/allfour0/64;EOSfull256,
completion110. Historicalpositive0/103,changed0/32,missing24/25,blockedmissing24;
numeric0/53/EOS1. Not a universalcapacity/architecturelimit; weak earlycontrol.

v2comparison70559 FINISHED0 saved dump/native256-lexical-full-unified-v2/comparison-
1000/report.json. Actualsampler/labels/LR/cursors equal; allprompts/targetsidentical;
perdocreaggregation/independent audits requiredandpassed. Oldlexical-minusRaw lang
+10.505553pp, fixedcheckpointdoc95CI[10.202668,10.811576] (4852docs/5000draws,
notseeduncertainty). Oldlexical21positivevsraw0 butmissing0vs59; allfour0BOTH.
Don't substitute this for final2epoch/strongmixed51261.013%/broadqualitytarget.

NEW post-analysis tools/compare_native256_lexical_span_full.py +test/protocol,
3CPUtestsPASS. Source/checkpoint/freshstate/Adam/actual-log/input/prompt guards;
paired docs/strictchangedpairs/legacy+notebookindividual/bothswaps/allfour/visibility
strata versus oldlexical andraw. Defaults bothmatched1,000; latercanselectraw1
withoutretrainingobsoletecontrol. NOactual spancomparison yet(allresultauditsneeded).
Doesnot editfrozenmodel/trainer. TemporaryDirectoryACL tests initiallyfailed;
log tests now serializedStringIO withonlyfileopening mocked; actualcomparisoncode
still readsreal files. Don't request permission for test scratch cleanup.

ORIGINAL TEST INPUTS prepared46144 FINISHED0: dump/native256-original-test-inputs-
20261018, original7646rows/1757421labels/48459genuine64blocks, CPUfrozenE9426432.
No studenttestpredictions/metrics/optimizerupdates. Helpersprepare_native256_
original_test.py andverify_native256_original_test.py +protocol are FROZEN by
manifest; noeditingwhileauditLIVE. CPU independentALL-prefix audit66169 LIVE on
core10; derivesoriginaltoken/mask/documentbytes, reencodesgroups127 vs256. GPU
cacheagreementstillrequiredlater; final originalTESTstudent evaluator NOTyet
implemented, reservedforactual48762/full361357104labelaudits. No pilotTESTtuning.

Two UNIMPLEMENTED conditional research notes, NOTnext-runapproval/adoption:
docs/native256_raw_prefix_copy_hypothesis.md (copy branch lacks causalrawprefix;
bewaremissinghallucination) andnative256_source_occurrence_hypothesis.md (copyCE
sumsduplicatewordoccurrences; derivedcorrect-sourceposalignmentcouldteachbinding,
reader-onlyextra-gradientsmaypreservelanguagebutneedtest). Finishcurrentevalfirst.

# Earlier same-turn preparation history (worker state superseded above)

Authoritative newest state supersedes old worker/root notes below. GoalACTIVE/UNMET,
current turn PROGRESS; no real span training yet. No subagents/native/growing edits.
ONLY LIVE GPU is raw1 fixed1,000 evaluator75351. It completed full dev and all seven
legacy groups(0/512 each), finishing remaining controls/historical/numeric suite.
Do NOT launch another GPU until75351 is TERMINAL, then run its notebook evaluation.
Raw full-dev33.631227%/CE3.694164868 is preliminary pending aggregate/public audits.
Saved natural cases are IDENTICAL to compressed cases; provisional strict raw
positive0/285, missing59/63, changedvalue0/64, query0/64, bothpairs0; false-unknown
280/285. This is abstention, not successful fact reading. Compressed remains verified
21/285 positive,0/63missing, no robust all-four. Read-only provisional scorer79182
FINISHED0; full raw report and independent audits still required.

Span full-v1 ALL-DATA audit7936 FINISHED1 BEFOREtraining: frozen auditor compares
raw and compressed packets, but v1's single-arm tuple rejected its raw data-control
batch. V1 is preserved ZERO-update, with failure.json. No model/math/label error
or weight update. Its initial failure receipt compared source CALLBACKS outside
the required scope; that false comparison flag is NOT a changed-file result.
Appended source-context-verification.json proves EVERY archived source hash is
unchanged and correct scoped callback EXACTLY reproduces manifest. Preserve both.

NEW tools/run_native256_lexical_span_full_v2.py and tests/protocol implement ONLY
data routing: Data subclass scopes accepted formats to both inside originalbatch,
restores on every exit. Run/model factory still compressed64 ONLY; raw model is
rejected. Two tests43748 FINISHED0 PASS on genuine HEAVIESTtrainrow: identical
input/labels/masks,64xvisibility/source slots/identical IDs, tuple restoration and
nested driver failure restoresallglobals. Model/gate/kernel/recipe unchanged.

CORRECT ACTIVE SPAN ROOT dump/native256-lexical-span-unified-v2, unique format
native256-lexical-span-full-original-v2. Preparation41827 FINISHED0, exact actual
old fresh weights/RNG and byte-identical original orders, empty Adam. V1 remains
frozen; all V2 helper/protocol/test sources now frozen too. Data audit33295
FINISHED0/PASSED: all780186train/7831dev rows,180678552/1859388labels,
4937894/49168codes, all48762originalorders/targetcounts/LR, exactfreshemptyAdam.
No real span training yet. Next: GPU preflight (waitraw+notebook), actualfirstupdate --stop-after1
--skip-initial-eval, CPU firstaudit, then --resume --stop-after999 to1,000.
Use V2 driver ONLY for this new run. Existing original48762/two-epoch schedule,
23Mreader/Eseparate/assistantCE/oldcache unchanged. No extra QA/epochs. Full broad
goal still needs final two-original-epoch/test/held-literal evidence and paired
span-vs-both-controls analysis. Do not adopt from teacher loss/partialprefix alone.

# Earlier same-turn verified gate/training record (worker/root notes superseded)

Goal ACTIVE/UNMET. This turn made real progress: compressed1,000 quality verified,
targeted name-key diagnosis, separate span model, full CPU/GPU/independent gates,
raw1,000 training audited and fresh span original-corpus preparation. No adoption
or full two-epoch quality claim. No subagents/growing/native modifications.

ONLY LIVE GPU: raw full evaluator75351 via evaluate_native256_lexical_v2.py,
on immutable raw1 step1,000 copy. Full dev stage reports625335/1859388=33.631227%,
CE3.694164868; source breakdown/documents saved. Natural/legacy/historical/reasoning
evaluation still running; don't launch another GPU. Then raw notebook evaluation,
aggregate/public/source/notebook independent audits and v2 paired comparison.
Compressed44.136780% is already independently verified; raw language is preliminary
until its result audit. This is early fixed-step evidence, not final parity.

Raw trainer76505 FINISHED0/PAUSED1000; CPU trainingaudit96291 FINISHED0 PASSED:
all Adam1000,7416257 labels, exact original orders/LR/targets/cursor, all2097 earlier
protected originals unchanged. Raw SHA12ad6e39da45b7aed6ed2a8b04d2795a66154842c9e65c4623df25823fe90e7e.
Do not restart or automatically extend the paused v2 arms. Old native256 remains
untouched/paused8750. Compressed v2 fact/binding failure is described below.

Span model remains UNTRAINED. src/models/token_memory_64_lexical_span keeps nine
learned slots pooling causal ranges0,1,2,3,4,5..8,9..12,13..20,21..32. Exact23209216
trainable reader weights/fresh47 state/RNG/buffers and same global KV arrays;
frozen9426432 encoder separate. Six model CPU tests plus full21 CPU genuine-input
check passed(loss exact/all144 parameter-gradient max3.576279e-6).

Corrected GPU gate45927 and independentCPU51525 BOTH FINISHED0/PASSED, export
dump/native256-lexical-span-preflight-v2-20261018. GPU micrograd2.98e-7/futuregrad
1.907e-6/zero future semantic-input gradient; independent explicit mean-embedding
CPU-to-GPU allparam max4.530e-6. Public/cache/actual64 rollover/metadata/bytes pass;
all2113 guarded earlier files unchanged. First gate90854 FINISHED1 before body
profiling due exact64MiB residual. Entire failed export is preserved/protected.
Separate forward/backward diagnosis proves cuBLAS scratch: clear ONLY workspace
releases64MiB to ZERO, unchanged settings/weights. v2 clears it before each isolated
body, retains zero-live-allocation assertion, and includes recreated scratch in
peak. Model/math/old guarded sources unchanged. All span model/gate sources frozen.

Measured untrained loaded cost at2496 old words: whole cache4473448 bytes vs raw
112856128(25.23x whole,64x OLD). B4 forward span55.035ms/old53.757/raw237.190;
B1 decode span55.867/old52.836/raw45.151ms. Single-token decode remains slower.
Isolated B8 body279.020 vs236.476ms(+17.99%); peak5330576896 vs4593019392 bytes
(+703.39MiB,new4.9645GiB). Two moment-sized buffers included; Adam step/scratch,
E/data IO/eval/save excluded. Three body samples only; B4 particularly variable.
No end-to-end throughput/quality claim. docs/native256_lexical_span_results.md.

NEW fresh run dump/native256-lexical-span-unified-v1 PREPARED82157 FINISHED0.
tools/run_native256_lexical_span_full.py +tests/test_native256_lexical_span_full.py
+docs/native256_lexical_span_full_protocol.md. Three tests19260 FINISHED0 PASS:
scope/format isolation, actual saved old fresh weights/RNG, exact reused notebook
data provenance. Preparation proves every initial tensor/RNG and orders.npy BYTE
EXACT versus lexical v2, empty Adam. Only span compressed64 arm; no duplicate raw.
Original48762 schedule/2 original epochs/effective32/micro8/assistant CE/FP32AdamW
unchanged; unique format native256-lexical-span-full-original-v1. No extra QA/E
training. All new helper/protocol/test sources now FROZEN by prepared manifest.

LIVE CPU: new span ALL-DATA AUDIT7936 on core13; poll that handle. After it passes,
GPU preflight must wait until raw evaluator and its notebook GPU work finish.
Then actual span train --stop-after1 --skip-initial-eval, independent first audit,
resume --stop-after999 to1,000. These are NEW-update counts on original schedule.
Driver requires explicit stop limit, preventing implicit extension. Its scoped
callbacks restore all original globals, even on failure. Reused notebook critical
checks OLD data-preparation hashes/files, while new outputs bind NEW model sources;
no old dataset/manifest relabeling. CPU preparation/audit has no CUDA initialization.

The v2 comparison tool covers old lexical64 versus raw1 only. A paired span-versus-
both-controls comparison still needs implementation after span results exist;
do not misuse v2 callback guards. Final acceptance still needs full original2epochs,
strong mixed512 reference, original TEST and genuinely held same-format literals.
Literal IDs are ADDED FP32 metadata, not proof old encoder lossless or byte64:1.

# Earlier same-turn lexical1,000 record (latest worker state superseded above)

PreviousgoalturnPROGRESS(v2firstupdatesverified/resumed). CurrentturnPROGRESS:
compressed1000actualtraining/evaluation/independentresults COMPLETE; diagnostic
finds missingOwner inlocalkeys; separateone-changepooled-keycode+5CPUtestsPASS.
Full23Mreader/Eseparate/64:1/2ORIGINALepoch/broadqualitygoalACTIVE/UNMET.
No subagents/growing/nativechanges. DoNOTadopt/automaticallyextendv2comp1000.

Raw1resume76505/PID14928 FINISHED0/PAUSED1000 onSAMEoriginal48762schedule,
lastCE3.80495/~.316updates/s(no controlledspeedclaim). DoNOTrestart/extend it.
CPU trainingaudit96291LIVE, checkingallAdam1000/orders/labels/LR/cursor/oldhashes.
AllotherworkersTERMINAL:comptrainer76455/compaudit18359,rawquarter97499,
fullevaluator64260,notebook45058,aggregate57675,CPUpublic64967,spanCPU2231.
DoNOTduplicate/restartraw. Afteraudit: same frozenGPUevaluation/extraNotebook, independent
results andmatchedcomparisonstillrequired. OnlyONEGPUworker, sourcesFROZEN.

Compressed1000checkpointSHAf660aaeff5fecd71aaf3df5768d87948331bc939942f63041666fdfad7d7aaee.
AllAdam1000/7416257labels/originalorders/LRs/targets/cursor/2097oldhashesverified.
FULLDEV820674/1859388=44.136780489%,CE3.0655990383. Oldplain64same1000
35.105475565/3.511000690; oldnative25633.170806739/3.733724880 (its1000actual
batchorders/labels/LR/cursor comparedexact). Gain+9.031305/+10.965974pp,
ARCHITECTURE+representationcomparison/old1000weightsnotretained/no pairedCI.
NOTsame-stepraw1/final2epochs/broadfactqualityclaim. NewliteralIDs are added
FP32metadata,NOTproofoldElossless. Source/oldruns/cachesunchanged.

Verifiedfreefacts:21/285originalpositive;0/63missing;3/64changedvalue;3/64query;
bothcorrectvalue1/64/query0/64;blocked0/64. All285positiveEOS. Historical8/103
positive/3of32changed/1of25missing;compressedcolors0/8. All7legacyaccessgroups
0/512each;differentuntrainedvalue-onlyformat, notglobalcapacityproof. Numeric
probe0/53/EOS3of53(cap256),limitedproxy. Full/source/aggregateverifiersPASS;
60source-selectedCPUpubliccasesmatchcachedGPUoutputs, allsource/metricschecked.

ProperBPEprefixnotebook:full22/256 vscompletion78/256;bothswaps1/128 EACH;
allfour0/64 EACH. Blockedfirst16units0/64 EACHformat andall64swappedquestion/
formatpairsoutputIDENTICAL. Independentreplay32enabled/blockedCPUpredsmatch,
everyoutputtext/token/EOS/scorechecked. Cleaninputstrata fullrawName+noRoll40/
completion62allfour0;wholequeryraw+noRoll11each0. Whole-querycoding notsolecause.
Frozenmaincompletion0/477hasstandalone-spaceprefix; differentcases/condition,
doNOTattribute0->78solelytokenization. SupplementalprefixexactfixedName/relation
and128swappedpairsidenticalaudited. SourceWordbags/positions/lengthscontrolled.

Exploratoryname/valueanalysis:correctname95/285,value60/285,both21;68correctName
wrongwholeSourceValue;39otherNamecorrectValue;42othercompleteRecord;39otherName
differentSourceRecordValue;20nonSourceValues;56otherformat. NOTpartialsuccess.
INPUTONLY key-span-diagnosis:570pairswithin285positivecases, ALL9tokenkeys
containNOownerName atfirstvalue-discriminatingtoken;completeNameneeds11..20
tokens. Esemanticmemory/queryhstillavailable,NOTwhole-modelimpossibility/solecause.
Humanreport docs/native256_lexical_1000_results.md; fullartifacts under
dump/native256-lexical-full-unified-v2/compressed64/evaluation-1000.

NEWUNTRAINEDsrc/models/token_memory_64_lexical_span(3files). SAME9learnedkeyslots
nowpoolcausaloffsets0,1,2,3,4,5..8,9..12,13..20,21..32. Scalarwordscoreprefixsums
meanvalidranges; future/inaccessible scoresZERO BEFOREscan. OriginalWordBits/
twoReadLayers/raw64/Efrozen/global1slotper64/cachearrays/23209216trainableparams
unchanged. Full21layerfreshNNstate/ALLbuffers/TorchRNG EXACToldFast47. No new
KVbank; temporarypoolindices/counts/scores add work/memory. NOTfreecompute.
Prospective docs/native256_lexical_span_protocol.md. SixCPUtestsPASS(7350exit0):
independentactualembedding-meanvsScalarPrefix/allReader+embedding+inputhgrads;
cache/public/ACTUAL64roll/BOSexcluded/metadata+byte/inventorybothmodes;future
IDs/raw/allparamgrads+futureSemInputgradZERO;empty/blockedfiniteALLgradspresent;
micrograds+actual8Adamserializedresumeacross2epochs;fullfreshbudget/RNG/buffers.
AdditionalFULL21CPUgenuine-inputcheckfinished0: batch2/raw128/3oldcodes/10labels,
lossEXACT/all144paramgradmax3.576279e-6/freshState+RNGexact/zeroUpdates.
Output dump/native256-lexical-span-cpu-preflight-20261018/report.json passed;
sourcesFROZENbythisreceipt inclnewGPUgate/reference/verifier/protocol/tests.
New tools/probe_native256_lexical_span.py +verify_native256_lexical_span.py and
native256_lexical_span_reference.py/preflight_native256_lexical_span_cpu.py READY.
No GPUbenchmark/preflight or realcorpusSpantraining yet. BEFOREtraining execute
genuinesourcefullsizeGPUnumerical/cache/roll/micro/future/referencegrad
andcompleteoldFast-v-Span-v-rawloadedtimings+isolatedheavybody/peaks. Independentverify.
Newpreflightrootplannednative256-lexical-span-preflight-20261018(NOTcreated).
Do noteditoldguardedkernels/helpers; newpackage only. Raw76505isnowTERMINAL0.
AfterCPUaudit completes, executeSpanGPUgate thenCPUindependentgate, andraw1000eval.
Thenfreshmatchedboundedoriginal-scheduleSpanpilot maytestonefeaturechange;
mustNOTsubstituteconditionalprefixwinforfullQA or1000for2originalepochs.
Finalcoveragealso needs sameNotebookformatnovelliteralvalues andoriginalTEST
evaluationatfullweights; currentfullhelperhardcodesDEV, thosepathsnotprepared.

# Earlier same-turn state and implementation history (superseded by above)

PreviousgoalturnPROGRESS: v2data/GPU/bothactualfirstupdates audited,compressed
resume76455started. CurrentturnPROGRESS:76455FINISHED0 pausedEXACT1000(still
ORIGINAL48762schedule),lastbatchCE3.20155,~1.161updates/sresumeaggregate(earlyCPU
auditcontention/IO/saveincluded,no matchedspeedclaim). Sourcesstrictlyfrozen.
Compressed1000CPUtrainingaudit18359PASSED/FINISHED0:7416257labels/allAdam1000/
alloriginalorders/targets/LRs/cursor/2097oldprotected unchanged. CheckpointSHA
f660aaeff5fecd71aaf3df5768d87948331bc939942f63041666fdfad7d7aaee.
rawSAMEcheckpoint1resume97499LIVE/PID28696 stop-after249 NEWupdates ->250,
lastobserved75 CE6.98766/~.277updates/s(earlyCPUauditcontention,no matchedspeed
claim), thenPAUSEforcompressedGPUevaluation. Raw250
is an intermediate cursor on SAME48762schedule, not a new pilot/reset/epoch;
aftercompeval resumeSAMEraw250 +750to1000 for matchedcomparison. This gives
earlierqualityevidencewhileCPUcompaudit overlaps actualrawtraining. DoNOTstart
anotherGPUworker while rawresumeLIVE. No newauxdata/loss/architecture changes.
UPDATE97499FINISHED0 pausedEXACT250. Intermediateactualcursor/Adam250/
1831425labels/all250batchesmatchauditedcomp verified; receipt raw1/cursor-
verification-250.json. rawcheckpointSHA9a0f43de1426661279071efb2e5ec076c20c294417dca167441baeaa8a48282b.
Fullcompressed1000GPU EVALUATION64260LIVE/PID28812 via newthinCLI; doNOTstartanotherGPU
worker/editguardedsources. Alltrainingworkers terminal/paused(comp1000/raw250).
FULLDEVteacher stage FINISHED:820674/1859388=44.136780489%,CE3.0655990383.
Saved-doc reaggregationmatches. Oldplain64same1000 originalupdates35.105475565%/
CE3.5110006898 ->+9.031304924pp/CE-.4454016515. DescriptiveEARLYarchitecture+
representationcomparison, noold1000weights/per-docCI; NOTrawmatched/final2epochs
orfreefactclaim. Receipt evaluation-1000/early-development-comparison.json
bindsdoc/oldvalidation/checkpoint/dataauditSHAs. Natural540freeanswers currently
running(nooutputsfileyetatlastpoll),thencompletion/legacy/historical/reasoning/
oldprotectedhashes. DoNOTassumeevaluationcompletebefore64260terminal.
Finalacceptancestillneeds sameformatnovelvalues and full originalTEST evaluation
atcompleted2epochweights(currentfrozenfullhelperhardcodesDEV; notimplemented
originaltestpathyet). Do not substitute repeatedly inspectedDEV/syntheticQA
for finalbroadheldoutevidence. No such newtesttrainingdata/run started.
GoalACTIVE/UNMET; no full2epoch or qualityclaim; nativeoriginal untouched.

NEWCPUonlyheldoutprepare18272FINISHED0: dump/native256-lexical-binding-check-
20261018,64fourwayunits/512full+completioncases+128blocked;67structurally
eligible originalDEVnotebookrows. Exactsame sourcewordbag/record+valuepositions/
queryphase/rawtail acrossswappedmemories, bothnamequeries, factsfullyold.
ConditionalprefixBPEboundarybeforeANYvalue, name/fixedrelationonly. No training
data/loss/model/KVchange. NEWhelper+independentverifier+protocol+3CPUtestsPASS
counterfactualconstruction/value-freeprefix/constant+lastrecord+truncatedfail/
unequalcoordinatesreject/independentscores. Allthese sources NOWFROZENbynew
manifest (inclpublicreplayhelper/tests); doNOTedit. FulldataauditPASSED/FINISHED0:
all512sources/queries/targets/prefixoffsets/wordbags/fixedvaluecoordinates/
unselectedrecords/background/128blocked controls independently rederived.
Evaluate copiedfixedcheckpointAFTERfullhelper'soriginaleval; strictEOS/allfour/
bothswaps/blockedoutputinvariance/publicCPUreplayrequired. Correlatedextra
heldoutcontrol,notnewindependentdataset orcurrenttrainingfit. Source/model
success NEVERusedtofilter. Prospective docs/native256_lexical_binding_check_protocol.md.
NEWcompare_native256_lexical_full.py +2CPUtestspass pairedclusterbootstrap/
strictEOS/allfour(incllegacyIDonlyrows). Requiresbothindependentpublic+aggregate
andnotebook audits; checksACTUALsame-steporders/labels/LR/cursors/checkpointAdam,
allsourceprompts/targets, doc/source accuracy/CE, strictEOSchangedpairs/allfour,
fixed-checkpointdoc/unitCIs. Retainsstrongoldmixed51261.013%reference separately;
weaknewrawparityaloneNOTenough. Syntaxcheckedafterstrictpairedcountsaddition.
No actual comparisonresults yet. No architecture/source/trainingedits.
AdditionalPROSPECTIVEprefixauditPASSED:all512prefixes exactlytokenization of
queriedname+' keeps the notebook',128same-question swappedprefixpairs identical;
manifest-bound prefix-independence-audit.json. No modeloutputs inspected.
InputvisibilityauditPASSED:full256cases namesfullyraw168/wholequeryraw52/
targetsfitbefore64roll248;completion256cases namesfullyraw248/queryraw44/
targetfit256. Cases NOTfiltered. Prospectiveallfourcleanstrata fullrawName+noRoll
40units/wholeRawQuery+noRoll11;completion62/11. Failureswithcodedquerytextnot
uniquefactbindingdiagnosis. ComparisonrequiresbothadditionalauditSHAsandreports
theseinput-onlystrata; thirdCPUtestPASS requiresALLfourinputsqualify.
Nextafter97499TERMINALpaused250:read/audititsactualsavedcursor/Adam, thenrun
FROZEN evaluate_native256_lexical_full.main under run_native256_lexical_full.scope
onGPUcompressed64 (new evaluation-1000 copy oflast1000), THEN notebook-binding
evaluate onthatcopiedcheckpoint. OnlyONEGPUworker. Aggregate/publicCPU/source
andnotebookverificationrequired. ThenSAMErawresume +750 ->1000 (notreset),
fullsame-stepevaluation/comparison. No full2epoch/qualityclaim untilproved.
NEWthinCLI evaluate_native256_lexical_v2.py --armcompressed64 --step1000
scopesONLYv2outputroot/unchangedrunner callbacks andcallsFROZENfull-evaluator.
Checksactualexpectedstep/source beforeeval, ownsource unchanged after; writes
entrypoint-verification receipt. No changed math/weights/model. Syntaxchecked.
Remainingcoverageafterinitialquality:legacynew-value accessphrase/value-only
formats alone cannot isolate newliteralvaluegeneralization in the trained
notebookfull-reply format. Need same-format novel literalvalues before final
broadqualityacceptance; no such additionaldataset/run started yet.

# Previous v2 initialization and first-update record

User reconfirmed23M READER budget, encoder separate. GoalACTIVE/UNMET.
Fullv1 preflight55694FINISHED0: compressed8 grad5.59e-9/body.3098,.3115s/
4.592GBalloc5.279reserve;raw2 grad3.73e-9/body.5706,.5438s/5.981GBalloc
6.161reserve. Exact initial weights/emptyAdam/zero updates both. v1 stays
zero-update/frozen; no actual full-run training has started. Allworkers terminal.
NEWrun_native256_lexical_full.py redirects process-local output/update/source
callbacks to NEWdump/native256-lexical-full-unified-v2, SAME full48762/original
2epoch recipe. Only redundant per-parameter CUDA finite synchronization removed;
SAMEclip_grad_norm error_if_nonfinite catches NaN/Inf BEFOREAdam. TwoCPUtests
PASS exact old/new8stepAdam+serializedresume and NaN/Inf rejection/unchanged
weights/emptyAdam. Newprotocol docs/native256_lexical_update_protocol.md.
v2prepare11678FINISHED0;dataaudit92408PASSED/FINISHED0:all780186train/7831dev
rows,4987062blocks,ALL48762orders/targets/LRs,exactfresh47initials/emptyAdam.
FullGPUpreflight51464PASSED/FINISHED0:comp8 grad5.59e-9/body.2447,.2609s/
4.592GBalloc5.279reserve;raw2 grad3.73e-9/body.6541,.6242s/5.981GBalloc
6.161reserve. Zero optimizer/weightsunchanged/emptyAdam. These body samples
EXCLUDEAdam/IO/eval/save; not a controlled throughput win. ACTUALcompressed
firstupdate launched with stop-after1/skip-initial-eval; audit before resume.
Firstcompressed13005FINISHED0 paused1;independent7677PASSED/FINISHED0:
allAdam1/source/order/cursor/LR/7035labels/2097protectedfiles unchanged.
Firstraw76633FINISHED0 paused1/same7035labels;independent35972PASSED/FINISHED0
(CPUonly scoped callback, no CUDA context),allAdam1/2097protected unchanged.
Compressedresume76455LIVE stop-after999 NEWupdates ->1000 ofORIGINAL48762
schedule, not a new epoch/pilot. Lastobserved100 CE5.28724/~.94updates/s
(resumeincludesCPUauditcontention;NOTcontrolledspeedclaim). Rawpaused1 ready;
doNOTstart raw/anotherGPUworker until76455TERMINAL. Sourcesstrictlyfrozen.
DoNOTassume1000complete/full2epochs/qualityyet. Nextverifyactualpaused1000,
matchedrawcontinuation, same-step language/facts before deciding remaining
fullschedule. Neitherarm shouldreset/restart or get extra original epochs.
Current-turnPROGRESS: v2data/preflight/bothactualfirstupdates verified,
independentfull-source/strictEOS/publicreplaytool+tests implemented; fullquality
goalACTIVE/UNMET. No growing/subagents/nativechanges. Human-readable status
docs/native256_lexical_full_status.md. Allother workers terminal.
NEWverify_native256_lexical_full_replay.py +2CPUtestsPASS strictEOSpair scoring
and public(noncache)64-rollover/blockedcoordinate controls. Separate actual
independent reconstruction of540natural source/edit/query/targets passed15759.
This verifier is supplemental, not yet applied to trained results. Requires
source-selected publicCPUgeneration replay+allsource/allfour/strictEOS metrics
before finalqualityclaim. Does not change any guarded trainer/eval/model files.
Neveredit guarded v1/kernel/source/protocol/test/exports or old/native artifacts.
No subagents; one GPUworker; native256original remains paused/untouched.

# Earlier same-turn preparation record (v1 gates now terminal)

PreviousgoalturnPROGRESS,currentturnPROGRESS:fullsizeGPU/cache/gradient/speed
checkscompleted,exactfastkerneltested/verified,newfulloriginal2epochcomparison
codeprepared. FullgoalACTIVE/UNMET,noquality/adoptionclaim/noOldRunchanges.
No subagents/growing. Reader23209216/Eseparate;64oldwordsglobalONEKVposition.

Fast48389FINISHED0,indep26051PASSED/FINISHED0. Sources/referenceexportsFROZEN.
GPUfast-v-refLPmax1.91e-6/grad2.38e-6;micro3.28e-7/future2.15e-6,zeroFutureSrc
grad;indepFULLBASECPU-v-fastGPUallparam5.72e-6. all2074priorprotectedunchanged.
FiveCPUfasttestsPASSinclFullstate/buffers exact,refgrads/cache/roll/future/Adam.
FastB1blocks39msforward44.77/decode45.02 vsnative1~37.44/37.30:stillLATENCYcost.
B4forwardfast45.33 vsnative1 131.56(~2.9x),prefill87.61vs146.67. Wholecache
4473448vs112856128bytes (~25.2x inclraw64/nextblock;64xOLDpositions). Fastbody
B4.13746s/B8.146917s,peakalloc2.57181GB/4.68634GB incl2momentsizedbuffers;
EXCLUDESAdam/IO/E/eval/save. IMPORTANTfastprofilekeepsoneidleReferenceLMthrough
loopvar/lossgraph (~92.84MB+buffers),so'isolatesfast64'caveatnotexact;reportthis.
Notraining/qualityfrombench. Noanotherkernel/archtuningbeforeactualqualitytest.

NEWtools/native256_lexical_full.py Data+PackedCodes+objective;NEWsrc/models/
token_memory_64_raw_reference 2files ordinary21x256/FFN32467/23M allraw oldword
embeddings/fullprefixGRADS/nolexicalhead. Fullbudgetprotocol
docs/native256_lexical_full_protocol.md isFROZENbyrawcapacityexport.
3CPUfulltestsPASS NumPy-v-Torchmetadata/oldcacheunchanged,actualfullsizeRaw-v-
OriginalNativeallparams+prefixgrad,globalCE/micrograds+serialized8Adamresume
cross2epochcursor. Rawcapacity39135FINISHED0:2496oldwords+512current/355labels,
micro1body.184386s,3.15676GBalloc/3.27785GBreserved incl2momentsizedbuffers,
emptyAdam/exactweights/zeroUpdates/fullprefixNOTdetached. Datahelper/Rawsrc/
test/fullprotocolareNOWFROZEN. DoNOTeditthoseorexistingold/model/kernel sources.

NEWtrain_native256_lexical_full.py +verify/evaluate helpers. Newrun
dump/native256-lexical-full-unified-v1,FORMATnative256-lexical-full-original-v1.
Armscompressed64Fast21Readervsraw1Conventional21Native,23209216TRAINABLEeach,
fresh47,encoderseparateonlycomp. EXACToriginalunified-v4/48762updates/2epochs/
361357104labels/32effective/PythonRandom47+epochorders/AdamW.01clip1warm200
LR3e-4cos3e-5,NOnewQA/Name/goldspanweighting/extraepochs. ROoldf32cache+SourceIDs
packedCPUINRAMperbatch,nonew5GBcache/noErecompute. Raw expands64oldwordslots,
visibility*64,allprefixgrads. Plannedmicrocomp8/raw2 (raw2capacityPENDING).
Uniquecheckpoint/recipe/source/arm/cursor guards;neveroriginalresumeentrypoints.

Prepare41302FINISHED0;independentdataaudit43408PASSED/FINISHED0:ALL780186train/
7831devrows/labels,4937894train+49168devcodeblockswordIDs/high28floatbits,
128stratifiedactualcomp/rawpackets,ALL48762oldorders/targets/LRs,exactfresh47
initials/emptyAdam. NoextraQA/epochs. NewFullGPUpreflightLIVE(afteraudit),DO NOT
duplicate/editnewtrainer/verify/eval/helper/test/protocol
whilelive. AfterprepareindependentdataauditALLoriginalrows/labels/allcachedblock
metadata+all48762oldorders/LRs/freshinitials;GPUprefightactualmic2vs1+b8comp/b2raw
capacity+zeroAdam/exactinitial. ThenactualfirstupdatePAUSE/auditbeforeresume.
Bothfull48k schedulesintend2ORIGINALepochs;partialcheckpointsNOTgoalcompletion.
EvaluatorsourceincludesfullDev,original348memory+192counterfactual/blocked,
known-name/relONLYcompletionprefix,legacy7accessallfour+blocked/shuffle,historical
312,smallnumericreasoningprobe. LegacyaccessNOTcurrenttrainingfit. Evalcopies
exactcheckpointbytesfirst;aggregateverifierexplicitlyNOTfullgeneration/source
audit;strongerindependentreplaystillneededbeforefinalqualityclaims. Alltraining
NOTSTARTEDyet. Native256originalpaused/untouched. AllGPUworkerscurrentlyterminal.

# Reference lexical GPU gate VERIFIED (2026-10-18)

PreviousgoalturnwasPROGRESS:NAMEisolationtrained/evaluated/verifiedpartialtradeoff,
newlexicalLM/cacheimplemented15CPUtests. Full23Mreader/Eseparate/2ORIGINALepochs/
64:1broadqualitygoalACTIVE/UNMET. No subagents/growing/native/oldsource/runchanges.

NEWtools/probe_native256_lexical_lm.py +verify helper +native256_lexical_benchmark.py
and2CPUbenchmarktestsPASS(actualNativeweights/public/cache/rollmath;native1
matchesdirectwordembeddings). Four23209216paramfreshmodels:native64original,
lexical64new,lexical1sameNN/freshweights/onewordKV,native1conventionalfullraw
nativeCore/nolexicalweights. NativebenchmarkcacheadapterONLY,nodeployedchanges.
Protocol docs/native256_lexical_lm_protocol.md detailsbeforetraininggates.
Run dump/native256-lexical-lm-preflight-20261018,formatnative256-lexical-zero-update-v1.
GPUworker3583FINISHED0,independent48797PASSED/FINISHED0. FullsizeCPU/GPUallparam
gradmax4.77e-6;GPUmicro4.17e-7/future2.44e-6;zeroFutureSourcegrad;cache/public
max1.91e-6,roll9.54e-7,emptyfinite,sourcecachevsE1.43e-5. ParamsALL23209216,
initialweightsunchanged/zerooptimizerupdates,all2053protectedearlierfilesintact.
Referenceexport+sourcefilesFROZEN. DoNOTduplicate/restart/editnewsource/helper/verify/test/protocol/
oldguardedsourceswhilelive. OnlyONEGPUworker. AllpreviousworkersTERMINAL.

InputgenuineTRAINdoc77644/segment112342,end2560:39old64blocks+64currenttokens,
samecontiguoushistoriescount1/39forALLmodels. Usesimmutablefulltrain.f32cache,
filelengthchecked;willreencodeEandcheckvecparity. Devmax2556haspartiallastwindow,
sofull2560TRAINsamplechosenforbenchmarkonly,NOqualityevaluation/filtering.
Tmpactivatelexicalgates/parity/rollover/empty/micrograds/futureallparamthenrestore
EXACTfresh47weights. Nooptimizerupdates. Batch1/4benchmarkfullforward,prefill
(source+raw63),cachedlasttoken,actual64rolloverEINCLUDED64modes/noE1modes,
2warmup/5alternatingordersavedrounds/cachebytes/peaks. HeavyB4oneactual512segment
originalassistantlabelswith2Adam-moment-sizedbuffers/NOAdamupdate. Independent
CPUfullmodel/public/cached/roll/bitmetadata/allparamgrad/median/sourceguards
REQUIREDafterterminalworker;doNOTclaimquality/speedbeforeactualreport+audit.
ReferenceB1blocks39 msnative64/lex64/lex1/native1:forward37.64/51.48/49.12/37.44,
decode38.56/51.04/49.75/37.36;B4forward37.56/51.21/137.05/133.37. Cachecomp4473448
vsraw112856128bytes(64xOLDSLOTS;~25.2xwholecacheinclraw64/nextblock). B4heavy
512/39codes/1420originalassistantlabels .151096s,3.05009GBalloc/3.27995GBreserved
incl2Adammoment-sizedbuffers,EXCLUDESAdam/I/O/E/eval/save. Noqualityclaim.

NEWsrc/models/token_memory_64_lexical_fast (3files) SAMEarchitecture/weights/
state-dict/budget/cache,exactunused-vocab-rowelimination+vectorized9laggathers.
NoTopK/nooldwordsculled;uniquerows/neighborindicesTEMPORARYnotcached. Source
vocab54IDsfor1block/782for39blocks vs4096;newkernelmustbenchmarksorting/sync.
5CPUtestsPASS referenceallparamgrad/forward,exactfullstate+buffers/budget,
cachedmetadata/rollbothmodes,futureIDschangeUbutnotearliergrads,empty/blocked/
micrograds/serialized8Adamresume. Toolsprobe_native256_lexical_fast.py +verify.
Run dump/native256-lexical-fast-preflight-20261018,worker48389LIVE. DoNOTduplicate/
restart/editfastsource/helper/verify/test/protocol/basefrozenfileswhilelive.
ReusesAUDITEDoldnumericalgateviaPROCESS-LOCALoutputroot/sourcecallbackoverride
restoredfinally;ALLoutputsgoNEWrun,oldexportunchanged. SourceNNparamsidentical.
FastvsfrozenrefCPU/GPUfullsizeactivereadgrads/outputs/cache/roll+complete5model
timings+heavyB4/B8required;independentverifierusesFROZENBASECPUequations.
Protocol docs/native256_lexical_fast_protocol.md. AllmodeltrainingNOTSTARTED.
GoalACTIVE/UNMET,noquality/adoption/full2epochclaim.

# Name-gradient isolation512 VERIFIED; lexical LM implemented/untrained (2026-10-18)

PreviousgoalturnwasPROGRESS: familiarbindinggainbutlanguagecostverified,head
ablationrulesoutcopyingALONE;wordbitsstoragegatepassedbutnoLM. GoalACTIVE/UNMET.
23MREADERbudget/encoderseparateconfirmed. No subagents/growing/native/oldchanges.
AllcompletedpilotsIMMUTABLE,neverextend/retrain. No final2epochqualityclaim.

NEWnative256_name_isolation.py +evaluate/verify/tests/protocol. Run
dump/native256-name-isolation-pilot-20261018,formatnative256-name-gradient-
isolation-v1,oneNEWarmisolated_name512. EXACTsamefresh47initialtensors/freshAdam,
parentname-anchorDATA/order/weights/LR reusedimmutable;oldvalue_only/name_anchor
comparatorsNOTretrained. OnlyNAMEqueryh.detach():NAMEcontrollerquery/gategrad
enabled,NOdirectNAMEcoregrad; ordinaryLM/answer/valuegradsretained. Indirect
sharedcontroller/copyinteractionsremain. E/packing/Dfrozen. Same23209216stored/
19227904trainable/one64tokencodeperslot/raw64/inference/cache. NOTwordbitsLM.
First128.75LM+1NAME;remaining384.75LM+.25answer+.25value+.25NAME. FP32AdamW.01
clip1LR3e-4cos3e-5warm30,effective16LM+16QA,micro8. TwoNEWQApassesNOTtwooriginal
LMepochs. Expected1872347LM/16896QAseen/8192NAMEtrained/12660answertrained.

FourCPUtestsPASS independentjointallparametergradients,NAMEcoreisolationwith
activecontroller/exactinference,microbatch+serialized8Adamresumecross128->129,
futurecodemask. Testsfinalrunfinished0. Prepare95531FINISHED0,dataaudit47575PASSED,
GPUpreflight82442PASSED/FINISHED0:gradmax1.12e-8,directNAMEcoregradabsent,controller
active,exactinitial/emptyAdam. Body3samples~.5635sisolated/~.5637ssharedEXCLUDES
Adam/cache/eval/save,nospeedclaim. Firstactual87708FINISHED0pausedstep1(LM8.312,
NAME9.339),independentfirstaudit40251PASSED/FINISHED0(allAdam1,4011LM/34QAseen/
16NAMEtrained/0answertrained). Resume98761FINISHED0at512(~1.74updates/s),audit
39154PASSED/FINISHED0(allAdam512/orders/LR/cursor/frozenD/labelcounts). Checkpoint
SHAa4d17e2df70b277aa05526562c629a93be15b3696062a6f3547a89408a8e5782.
Evaluator37108FINISHED0,independent81591PASSED/FINISHED0(36CPUpubliccases,
allcase/control/order/name/language/gen/historical/CIsrederived). All2005
protectedearlierfilesunchanged. AllworkersTERMINAL. DoNOTduplicate/
restart/editednewhelper/eval/verify/test/protocol/oldguardedsourceswhilelive.
Trainingdone;fullfreeEOS/allfour/name/positions/newvalues/
names/wording/missing/composed/trainfit/blockedshuffle+fullLMdev/generatoronly/
historical312+indepresults/CIsDONE. Control/shared/isolatedLM19.656629/10.047930/
16.481713%;isolatedCE5.309317624,genonly16.572442%/5.306840690. Isolated-control
-3.174916pp doc95CI[-3.257826,-3.093781];isolated-shared+6.433784pp
[6.304229,6.566021]. Copydisableonly+.090729pp. Familiar407/512/allfour68/128,
QAheldvalues402/512/allfour68/128;newnames119/512/allfour2,wording411/allfour72,
positions188/allfour11,composed5/allfour0. Missingpositive212/256/absent3/256,
allfour0. Actualtrainpositive105/128/allfour17/32;trainmissing60positive/0absent.
Blocked/shuffled0/64exacteach;historical0/103facts/0/32changed/0/8colors.
Descriptivepreclipnormwarm/latecontrol.6467/1.6018,shared21.917/4.200,
isolated30.196/9.184;remainingcontroller/globalclipcouplingspossible,NOTcausal
clipproof/convergence. Narrowtradeoff,NOadoption/languagepreservation/broadwin.
Report docs/native256_name_isolation_results.md. Protocol
docs/native256_name_isolation_protocol.md. Native256runremainspaused/untouched.

INPARALLELonlyCPU/newfiles: NEWsrc/models/token_memory_64_lexical/reader.py is a
trainablegenericshared-embeddinglexicalCOMPONENT;separateLMbelow,NOtraining.
UseswordbitIDs,9causaloldwordcontext,factoredlinearcontextmaps/nonlinearQUERY,
query-to-vocabprojections+scalarIDgathers,NOdenseperoldwordkey/valuebank,
temporarynormalizedVOCABtables. 2query/read
units,wordrecencybias,zero-initresidualgates,softvocabcopy+generator/NULL. No
hardcodednames/answers. Derivedwordkeys/values/scoresTEMPORARY/extracompute,
notpersistedKV; actualcacheintegration+completebenchmarkstillrequired. Exact
559616componentweights+CoreFFN31374=23209216verifiedactualmodulecounts;all
trainable,nofrozenD/newembedding. Encoderseparate. 8CPUcomponenttestspass budget/packed-
literalID/rolemetadata/zeroresidual/normalizedfiniteNULLandgradients/futurecode/
padding/queryconditioning+futurekeygradientzero. Seventhcompressed-vs1word-slot
localreadparitytestPASSED;8thindependentexplicitcausal-keyvsfactoredscoreparity
PASSED. Final6777FINISHED0. Newprotocol docs/native256_lexical_reader_protocol.md.
DoNOTclaimQA/quality/speed/KVintegrationfromcomponent. Nogrowing/oldedits.

NEWsrc/models/token_memory_64_lexical_lm integratesCore+reader,21layers/readunits
7/14,559616readerweights/CoreFFN31374=EXACT23209216allTRAINABLE;Eseparate. FP32
wordbitmetadataREINSERTEDafterVprojectionintoEXISTINGread-layerVslots;NOoldword/
featurebankretained/cacheattrsunchanged. words_per_memory64 compressed vs1true
rawreference,EXACTsamefreshinitialtensors. rawreferenceoneembedding/globalKVper
sourceword;readoutlocallexicalkeysidenticalforbothrepresentations. Mainpacked
raw64maskunchanged. Genericsoftpointer+NULL/generation,nogoldname/span/answers.
7CPUintegrationtestsPASSED(final70322terminal0AFTERfactoring):ACTIVATED(nonzero)read/recency/
mixturegates,actualfullbudget/freshraw-compidentity,cached/public/prefill/steps,
Vpayloadrecovery/cachebytes+exactarrayinventory,actual64generatedwordrollover
excludesBOS(componeEcode/raw64slotsNOencoder),allparameterfuture/padding/isolated
causalgrads,blocked/empty,actualmicrograds+serialized8Adamresume. NoLMtraining/
GPUpreflight/benchmarkyet. Protocol docs/native256_lexical_lm_protocol.md.
FullsizeCPUbudget/freshidentity;numeric/cache/gradtestsSMALL2layerCore+realreader,
NOTfullsizeCUDAproof. Report docs/native256_lexical_lm_results.md. Cache.codes()
explicitlyrejectslastlayernotcodebank;feedreturnsword+NULLprobabilities.
NextGPUgate(oneGPUworkerONLY),independentfullsize/cache/math/
sourceverificationandcompletebenchmarksBEFOREanylexicaltraining. Noqualityclaim.

# Name-key and name-anchor pilots VERIFIED; narrow binding, language FAIL (2026-10-18)

Previousgoalturnwasprogress:selective512trained/verifiedwordrenderbutnamebinding
failed. Thisturnprogress:source-keylinearlearnabilitygate,newdirectNamecurriculum.
Full23Mreader/2ORIGINALepochs/64:1qualitygoalACTIVE/UNMET. No subagents/growing/
native/oldsource/checkpoint/cachechanges. CompletedpilotsIMMUTABLE,neverextend.

NEWnative256_name_feature_gate.py +verify/tests/protocol. Probe55921FINISHED0,
indep44609PASSED/terminal0; 2CPUtestspass. Run dump/native256-name-feature-gate-
20261018. Twofreshlinear8class probes(raw125vsfrozenDnonlinear256),ORACLEQUERY
NAMECLASS,balanced512points/1024updatesAdamW.02/noWD. NO LM/E/Dtraining. Source
positionlabelsregexindependent; diagnostic64featurebanksNEVERproductionKV/attn.
Raworaclefamiliar489/512,114/128allfour;heldvalues469/512,103/128;heldpositions
335/512,48/128. Nonlinear512/512,128/128allthree. Trainraw3624/4096vs4024,
missing272/512vs460. FixedNULLmarginnotoptimized,notlinearimpossibilityproof.
Rawtrainmacro99.31%,nonlinear100%. Report docs/native256_name_feature_gate_results.md.

NEWnative256_name_anchor.py +evaluate/verify/tests/protocol. Newrun
dump/native256-name-anchor-pilot-20261018. Same23,209,216storedreader/onecode
attention/KV/unchangedinferusingCompatModel. BothfreshEXACTshared47initial(not
trainedcontinuation),E/DfrozenfreshAdam. value_only vsname_anchor512each. SAME
templatesBOTHentityquestions/equalphaseunit;train4096sourceexamples/codebank
reusedwithretargetedqueries;1792FRESHheldblocksdisjointpreviousaddresspilot.
SourceNamefirstBPEtargetderived,rawunshiftedscoreNAMECEtrainingONLY. First128
both.75LM;namearm+1.0NAME. Remaining384both.75LM+.25answer+.25valueaddr;
namearm+.25NAME. ControlNAME0.0. 16LM+16QA/micro8FP32AdamW.01clip1warm30
LR3e-4cos3e-5. 2NEWQApasses/NOT2originalLMepochs. Actualweightedlabelcounts
mustseparatewarmupzeroanswerexposures; namequeriesfirstdecisionsnotEOS.
Nointeractionloss/newparams/infer/KV/decoderfeatures.

3CPUtestspass zeroNAMEjointobjectiveexactoldallparamgradients,actualmicrograds/
exactserialized8Adamresumeacrosswarmup,futurecodeisolation. Prepare98557FINISHED0,
indepdata85245PASSEDbalancedtemplates512/512,128values30positives/pass,allsource
anchors/querysamephase/freshblocks/exactfreshinitial/originalLMorders. Fullsize
GPUpreflight46929PASSEDgrad5.59e-9/bothstages/exactweights/emptyAdam. Firstactual
value_only1trainer66426paused/FINISHED0,firstaudit94002PASSEDallAdam1,4011LM/
34QAseen/16NAMEseen/0answertrained/0NAMEtrained/frozenD. Resumedtrainer61662FINISHED0,
both512; audit12543PASSED(allAdam512/orders/cursor/LR/frozenD), each1872347LMlabels,
16896QAseen/8192NAMEseen;12660answer/valueTRAINEDlabels each, NAMEtrained0vs8192.
Evaluator1666FINISHED0,independent35915PASSED/FINISHED0,72CPUpublicgenerationcases,
allcase/anchor/allfour/control/language/historicalmetricsindependentlyrederived.
All1901protectedoldfilesunchanged. CompletedcheckpointsIMMUTABLE,neverextend.

Value_only->name_anchor strictvalue+EOS: familiar266->500/512,allfour4->120/128;
QAheldIDs264->484,allfour10->108;newwording285->498,allfour13->120. BOTHorders
248/256and252/256namearm,allfour59/64and61/64. Genuineknown-namebindinggain.
BUTnewnames252->129/512(allfour3->4),positions251->211(allfour2->21),composed
216->0(allfour0both). Missingpositives140->245/256,absent0->80/256;actualtrain
positive72->127/128,allfour4->31/32. Nameanchor503/512familiar,181newnames,
214positions,0composed. Bothblocked/shuffle64controls0exact. HistoricalALL0/103,
0/32changed,0/8colors. NObroadquality/adoption/2originalepochsclaim.

Fulldev1859388labels:control365493correct/19.656629%/CE5.189927408;name186830/
10.047930%/5.941782259. Name-control-9.608699pp fixedcheckpointdoc95CI
[-9.783904,-9.425124]. Readonlyheadablation30258FINISHED0,indep98705PASSED:
generatoronlycontrol19.454089%/5.199159038,name10.061106%/5.941511974.
Copydisableonly+.013176ppname; doesNOTrestorelanguage. Generatorstatesalsopoor,
notisolatedproofwhichgradientcausesdamage. Nooptimizerorcheckpointchanges.
Newverify_native256_name_language_head.py; alloldguardedhelpersremainfrozen.
Report docs/native256_name_anchor_results.md +research/goalstatusupdated.

SeparateNEWsrc/models/token_memory_64_wordbits component/test/protocol and
probe_native256_wordbits.py +verify helper. StoragegateONLY,noLM/inference
integration/training. Uses4lowmantissabits/FP32float:4floatsstore16bitpayload,
256floats64words. 12bitID+2inputrole+1inputmessagestart+1reserved; NEVERfuture
end/EOS/answer/goldspan. All256semanticcoordinatesretainedwith<=15ULPchange,
NOTexactoriginalfloats/NOTbytecompressionvsIDs. FP32mandatory,decodebefore
arithmetic/casts;futureVprojectionmustREINSERTmetadataexistingVslots,noIDbank.
Custombitwiseforward/identitySTEgradient. FourCPUtestsPASS inclall65536payloads,
all4096IDs/64positions,uint16,sign/finite/zero/subnormal/cachebytes/gradient.
StorageGPU/numericgate12191FINISHED0,independent27237PASSED/FINISHED0. CPU/CUDA
all65536payloadsand4096IDsatALL64positionsexact;existingKVfixture8860878bytes,
0added. Genuine512heldcodes/32768wordpositions:originalD32735correctBEFORE/AFTER,
32735/32736contentcorrect,0changedpredictions,maxlogitchange.000253677.
SameimmutableNATIVE8750fourdevsegments573labels:0changedpredictions,maxhidden
1.11461e-5,maxlogit1.39475e-5. NativeSHA57ccdacbf4ae79be637ace6d104848a5543bda33a8fe7486a39b24c7e8144052.
IndependentnumpyrecoversALL32768IDs,reaggregatesALLpredictions,CPU128decoder
positions+native2047replayed,all1901protectedfilesunchanged. NoLM/E/Dupdates,
noextraweights,noproductionreader/cacheintegration/QA/speedclaim. FP32bitpayload
storageonly;NOTproofoldencoderlossless. Report docs/native256_wordbits_results.md.
AllworkersfromthisturnTERMINAL. Neverrestartcompletedpilots;nextneedsseparate
query/span/fieldreaderwithlanguagepreservation,actualbudget/cache/speedchecks,
thenbroadfreshmatchedtraining. No new LM run begun. GoalACTIVE/UNMET.

# Fresh selective-address512 VERIFIED; word rendering works, binding FAILS (2026-10-18)

Currentgoalturnmadeprogress: actualbudgetedLM/cacheintegrated,trained2matchedfresh
512arms,independentlyverified. Full23Mreader/2ORIGINALepochs/64:1qualitygoal
ACTIVE/UNMET. Userconfirmed23MREADER/encoderseparate; includesfrozendecoderinside
readerbudget. No subagents/growing/native/oldsrc/checkpoint/cacheedits. ALLworkers
terminal; GetProcesspythonempty. DoNOTrestart/extend/adoptcompletedpilots.

Run dump/native256-address-curriculum-pilot-20261017,report
docs/native256_address_curriculum_results.md. Trainer62666FINISHED0 both512;
trainingaudit18846PASSED1872347LM/16896QAeach/allAdam512/orders/LR/frozenD.
Freshcore/controller47,pretrainedfrozenD,23209216storedreader/19227904trainable,
21layers256/FFN24619,onecodeattention/KVpositionper64tokens,lastlayerexisting
V=originalFP32code,noextracodebank. 16LM+16QA/micro8/.75LM+.25answer+.25address,
LR3e-4cos3e-5warm30AdamW.01FP32clip1. 2NEWQApassesNOT2originalcorpusepochs.

Generationallpositivegroups0/512/alwaysunknown. Selectivefamiliar244/512,
held128answerIDs251/512(first255),names267,wording259,positions256,composed248.
Mainallfourfamiliar9/128,held15,names19ARETEMPLATECONFOUNDED: trainingq0What/
q1Givecorrelatesqueryrole/template. Readonlysame-templatebothquestions/bothmemories
diagnosis21809FINISHED0; indep98493PASSED112CPUpublicfreeanswerreplays. Allgroups
inclPOSITIVEACTUALTRAINunitsZEROall-four/128BOTHtemplates/BOTHrecordorders.
FamiliarWhat254/512,GIVE253,held255/251; essentially50%recordchoice,queriedname
switchalmostneverchangesoutput(familiar0/256What,1/256Give). NOTrobustfactreading.

FrozenDgoldpos7680/7680valuetokens; CPUnameoracle79788FINISHED0 independently
derived15360/15360entityBPEs. Entityandvalueinfoexistsatgivenpositions;notQAproof.
Selectedsourcewordrender512/512familiarandheldvalues; familiarwrongrecord264,
outside4,heldwrongrecord257. Missingselectivepositive137/256,negative5/256;
generation0/256positive,256/256negative. Blocked/shufflefamiliar0/64. Historical
both0/103facts/0/32changed/0/8compressedcolors. No finalreasoning/qualitywin.

Mainold-eval54288failedzeroMreshapeAFTERgenerationQA+fullLM. FrozenfilesUNCHANGED.
Newtoken_memory_64_selective_compat.Model emptyMbranchONLY,nonewparams/positions/
KV,nonemptyexactsuper. TwoCPUtestspass. Recovery12746FINISHED0 preserved13completed
fileshashes,finishedhistory/selectiveeval. Mainindep90266PASSED; fullsizecompat
indep4633PASSEDBOTHtrainedstatesvsunchangedNativehidden/allcoregradsEXACT0,
emptycachedlogprobmax2.86e-6. All1769originalprotectedhashesunchangedandcheckpoint
SHAsmatchtrainingaudit generation690527df...f5e2/selectivec95ed6c6...e0b.
Sourcesbothselectivepackages,compat,training/eval/protocolsNOWIMMUTABLE.

Full7831dev/1859388labels generation18.324578%CE5.242833122 vsselective18.034536%
CE5.270522839; delta-.2900417pp doc95CI[-.3246316,-.2535656]fixedcheckpoint/one
seed. TheseareCOLD512models/half-LM16examplesupdate,notcomparablecompleted48k
baseline; new128heldIDsheldonlyQA,notcodec/language. Analysis33574terminal0.
No freshfull2epoch/genuineallrawreference/broadreasoning/endtoendefficiencyyet.

NextUNIMPLEMENTEDhypotheses docs/token_memory_64_address_next_hypotheses.md:
teachqueryNAMEANCHORfirstthenexistingvalueoffset,withsame-templatequartets;
orNEWstructuredlosslesstokenIDcodebranch (4096IDs->4base8digits/token->64x4=256
FP32onecode,canonicalpositions,selectedIDonly). Latterisattention/KVcompression,
NOTbytecompressionvsuint16IDs;noencodertrainingneededtheoreticallybutNOnewcodec/
LMimplementedorqualityproof. DoNOTmistakeproposedroundtripforgoalcompletion.
Coulddecodeall64nonlinearfeaturesTEMPORARILYforscoresbutextraMACsneedbenchmark;
neverexpandglobalmemory/KV. Keepgoalactive,pursueaddressbindingratherthanrepeating
word-rendertestsorblindlossweights. Masterresearch/goalrequirementsupdated.

# Integrated selective LM gate VERIFIED; new address pilot preparing (2026-10-17)

GoalACTIVE/UNMET. UserCONFIRMED23MREADERbudget,encoderseparatebaselineconvention.
No subagents/growing/nativeoroldsrc/checkpoint/cachechanges. Priorcompletedpilots
immutable,neverextend. NewactualLMpackage src/models/token_memory_64_selective_lm
importsunchangedfrozenselectivecomponents. Corefresh47/21layers/width256/FFN24619;
stored23209216includes3981312frozenworddecoder+36864controller;trainable19227904.
Actualforward+incrementalK/VwithlastlayerV=ORIGINALFP32code,noadditionalcodebank.
Oneposition/code,64temporarySCALARaddresses/query/code,NULL,onepointworddecode,
mixtureoutput. Addressnottrainedyet; no goldaddressforwardinputs. Cacheprefill/
token/64actualwordrolloverexcludesBOS,clearsoldrawK/V;onlycurrentraw64IDsretained.

SixCPUtestspass actualallparamcausal/isolatedpackedgrads,futurepaddingblockedfinite
gradients,hotpath/frozen,incrementalprefill/steps/rollover/cachebytes,exactserialized
Adamresume. Probe67378FINISHED0; indep22559PASSED/terminal0. Run dump/native256-
selective-lm-preflight-20261017,all1764protectedunchanged; fullGPUgrad2.98e-7,
weightsunchanged/zerooptimizerupdates. Raw64fullforward~41-43ms(native~41except
first63.66),cachednext~44-49ms,rolloverinclE~99-108ms. Heavybatch8/raw512/M39/
2048labelsF/Bclipmedian.128128s,peak3.344GiBallocated/3.631reservedincl2Adam-sized
buffers. ExcludesAdam/scratch,I/O/eval/save. NOspeed/qualitywinclaimed. Report
docs/native256_selective_lm_results.md. Integratedpackage/probe/tests/protocol
NOWFROZENbypreflightinventory:DO NOTedit/addfilesinthesepackages. Newtoolsseparate.

NEW tools/native256_address_curriculum.py +evaluate/verify helpers,new2CPUtests
PASSactualmixedobjective/micrograd/exactAdamresume+sourcecounterfactualconstruction.
Protocol docs/native256_address_curriculum_protocol.md. Newfresh512eachgeneration-
onlyvsselectiveOUTPUTarms,samecompletebudget/core/addrCE/orders/language replay,
notoldnativecomparison/notpackingtest. BothEpackingfrozen. 16LM+16QA/update,micro8,
.75LMCE+.25answerCE+.25addrCE,FP32AdamW.01clip1,warm30LR3e-4cos3e-5. 2NEWreading
passes4096examples,NOT2originalcorpusepochs. 128balancedwordIDs,2records,bothqueries
bothswappedmemories,positions0/8/16/24,128/1024unitshalfmissing=>6.25%examples,
source-derivedaddresslabels+NULLforEOS/unknown. Held128IDs/names/wording/positions/
missing/composed. FrozenDpretrained; heldIDsheldonlyfromQA,notencoder/language.

PREPARER96793FINISHED0; dataaudit53469PASSED; additionalreadonlytool
audit_native256_address_evidence.py outsidefrozeninventory replays64encoderblocks
atORIGINALcontiguous128batchshape EXACT0error (random16batchroundingnotbitparity),
all7680gold-addresssourcevaluesdecodedcorrect,NOTquestion/LMquality. Firstcode
audit17968failedtoostrictdifferent-batchrounding; helperONLYfixed,97911PASSED.
FullobjectiveGPUpreflight71195PASSEDgrad5.12e-9/exactinitial/emptyAdam.
Train88083FINISHED0/PAUSEDgenerationactualstep1,LM8.316answer8.209addr5.382,
gradnorm170.46clipped1,4011LM/34QA labels. Firsttrainingaudit72391PASSED/FINISHED0
allAdam1/frozendecoder/orders/labels. Resumedtrainer62666FINISHED0,both512.
Trainingauditor18846PASSED/terminal0,both1872347LM/16896QA,allAdam512/order/LR/
frozendecoder; checkpointsNOWIMMUTABLE generation690527df...f5e2/selectivec95ed6c6...e0b.
DoNOTduplicate/restart/extend. Run
dump/native256-address-curriculum-pilot-20261017. Source/preparefilesstrictfrozen
whilelive; doNOTedittraining/eval/helper/test/protocol. Trainingauditcomplete.
NEWreadonlydiagnose_native256_address_binding.py +verifyhelper (outsideguarded
inventory) fixesQUERY-TEMPLATECONFOUND: twooriginalqueriesuseddifferenttemplates;
nowBOTHnamesgetSAMETEMPLATE,bothswappedmemories/equallengthphase,unchangedcodes/
targets. Addedactualpositive trainingunitsbecausemainfirst128trainfitcasesare
missing-heavy. Diagnosis21809FINISHED0: selective~50%exactinclheldIDs,butZERO
all-four EVERYgroup andpositiveTRAINunits; nameswitchoutputsusuallyidentical.
Generationalwaysunknown onpositives/zeroexact. Wordrenderingworks,entitybinding
FAILS. NOTrobustretrievalgain. Bindingindep98493PASSED/FINISHED0,112CPUpublicreplays.
Mainfullevaluator54288FAILEDaftergenerationQA+FULLDEVcompleted, historicalsingle
emptyM=0 triggeredoriginalnewModel.memory.reshape(B,0,H,-1)ambiguouszeroelement.
No training/weightschanged. DO NOTeditfrozentraining/eval/productionnewmodel.
NEWsibling token_memory_64_selective_compat.Model overridesONLYzeroMhidden with
ordinaryno-memoryattention(no extra positions/KV/params); nonempty exactsuper.
TwoCPUtestspass actualallcoregradsvsUNCHANGEDNative.hidden andemptycache/public,
nonemptybitexact/no newparams. NEWfinish_native256_address_evaluation.py preserves
allcompletedgenerationoutputs/FULLDEVdocs byhash,recordsrecovery-provenance,
finisheshistorical+unstartedselectiveeval withcompatwrapper. Session12746FINISHED0.
Compat/recoveryfiles nowfrozenwhilelive. No resumedtraining/no duplicatedupdates.
Recovery/mainresults/compatibility/checkpointSHAauditsALLPASSED; seeLATESTtop.
FullfreeEOS/addresses/allfour/controls/trainfit/
fulloriginaldev/historical312+indepresultsrequired. Needcheckcheckpointdigests
againsttraining-verificationrecordbefore/aftereval; additionalread-onlyaudittool
maybeaddedoutsideguardedinventoriesifneeded. No claimfrompreflightororaclealone.

# Selective trained word reader VERIFIED; address/LM integration UNMET (2026-10-17)

Previousgoalturn wasprogress: positioncurriculumverifiedsmallLMgainbutpoorrobust
reading. Newturnmadeprogress: broadstudenttestandtrainedselectivedecoderproved
wordrecoveryatORACLEpositions; full23M/2originalepochs/64:1goalACTIVE/UNMET.
No subagents/growing/nativeoroldsrc/checkpoint/cacheedits. ALLworkersfinished.
DoNOTrestart/extend/adopt completedstudentorpreviouspilots. No LMtrainingthisturn.

NEWtools/native256_word_readout.py +verify/tests/protocol. Run dump/native256-word-
readout-pilot-20261017. Genuine4096train/512dev/512test64blocks. ExcludeheldIDs
light/dark/sun/star/book/house/school/road fromreadouttargetsONLY301labels/pass;
encoder/prototypesalreadyknowvocab. FOURfresh47~394753paramstudents: selectedh,
selectedh+z,z+sinposition,rawsourceembeddingORACLE. SAME2passes261843labels/epoch,
523686each/512updates/batch1024micro256/tail723/AdamW.01clip1/warm10/3e-4cos3e-5.
Allcodesusegoldpositions,NONErawtargetIDinputexceptexplicitraworaclecontrol.
PrototypebasisnormalizedfrozenDwordembedding;notfresh23M/fullcorpus2epochs.
FourCPUtestspass actualmicrograds/frozen/serialized2epochAdam/targetisolation;
GPUpreflightgrad7.45e-9,emptyAdam,.209GiBallocated/.234reservedcomponentonly.
Prepare21609/dataaudit/preflightFINISHED0,firstactualselected1+auditpassed;
resume76806FINISHED0four512. FulltrainingauditPASSED allorders/LR/Adam512/labels,
protosfixed. Eval9674FINISHED0/indep22485PASSED incl1024CPUtestrecompute/arm/mode,
1524oldhashesunchanged. Testwordexact selected13955/32768(42.59%),global11170,
codepos1078,raw32768. MacroIDaccuracy6.15/3.18/.037/100%;test2884IDs. ALLcode
students0/52held-IDtokens,0/64neversupervised,0exactblocks. Selectedtrainfit3409/8192
underfitnotcapacityproof. Priorfactsstudentfewhits/held0vsraw256/256oracleALL.
Report docs/native256_word_readout_results.md. Completedstudentsimmutable.

CRITICALNEW finding: originalcodecdecoder_layers=ONE. Its SINGLEselectedfeature
throughtrainedblock gives32728/32768dev,32735/32768test; nonspecial32728/32728dev,
32735/32736test. WorddecoderdoesNOTneed64contextpositions for these tests.
Thispivotsnextactiontoreusetraineddecoder,insteadoflearninginversefromscratch.

NEWsrc/models/token_memory_64_selective/{reader,budget,__init__}.py. SelectiveReadout
selects125rowsofW256->8000 ->shared125->256 ->fusedsingle-keyattentionA=Wo@Wv ->
trainedFFN/norm/wordhead. NOQ/K/softmax/expandedsourcepositionsinwordreadout.
Frozenparams3981312. AddressScorer36864UNTRAINED:foldquery125throughWonce,score
codes64SCALARaddresses/query/code,queryoffset17,NULL,visibilitymask. No64feature
vectors/code/KV expansion. CodeKVCacheSTORAGEFIXTUREusesONEkey256+VALUEoriginal
FP32code256/code+lengths;noextra retainedcode/features/history. Replacingordinary
projectedVisREQUIREDforLM; fixtureISNOTintegratedcacheprefill/decode/parityproof.
BudgetFixtureHASNOFORWARD. Actualparameterallocations23209216 readerwithFFNtotal
24619; TOTALincludingactual9426432encoderalso23209216withFFN6208. Quality/capacity
costunproven; budgetpreferencequestionstillpending,don'tre-askunnecessarily.

SixCPUtestspass fusedvsoriginaldecoder,foldedvsexpandedaddressalgebra,offset
boundaries/futureblockedgradients/NULL/emptycache/cachebytes/twoparamfixtures.
Protocol docs/native256_selective_reader_protocol.md. Probe66808FINISHED0;
indep77434PASSED allcases/sourcemetricsand2048CPUoriginalpointsamples/1726hashes.
Run dump/native256-selective-reader-preflight-20261017,export readout.pt.
FusedpredsidenticalunfusedpointALLdev/test;fullDtestpredsidentical,devoneSPECIAL
token disagreement. ORACLEfacts2048/2048(all8groupsinclnewwords/tworecords),ZERO/
shuffled0/2048. Goldqueried-recordPOSITIONSgiven:NOTfreevalue+EOS/QA/binding.
No optimizerupdates; frozenweightsunchanged. Componentaddress+forcedreadlatency
batch1 M1/M39=3.802/4.426ms;batch16=4.241/4.319ms. Includesvalidation,someGPUCPU
range-checksync;EXCLUDESmainLM/encoder/ingestion/integratedKV/addressSUCCESS.
Prototypeallocated.126GiBreserved.133 NOTfullmodel. No fullspeed/qualityclaim.
Report docs/native256_selective_reader_results.md,goalrequirementsMDupdated.

IMPORTANTsourceinventorytrap:probe.sources()GLOBSsrc/models/token_memory_64_selective
/*.py andfreezes3componentfiles+tests/protocol/probe/verifier inexport. DO NOTedit
ORADDfilesinthatsubpackage. FutureLMmustNEWsiblingtoken_memory_64_selective_lm,
importimmutablecomponents/weights. Oldpack/nativemodelsourceguardsalsopersist.

Nextconcretework (NOTimplemented): actualbudgetedLMforward+incrementalcache with
oneexistingmemoryVslotpreservingoriginalcode(notnormalized/projectedV);wordcopy
outputmixtureafterlearnedblock/positionselection,source-derivedaddresssupervision
inclNULL/EOS andlanguage replay. Trainqueries,notoraclepositions. Preservefuture/
blocked/padding/rollover64 causality,wordDecoderfrozen,counterfactualallfour tests.
Fullmodelbenchmark BEFOREtraining,thennewboundedmatchedpilot; finalfresh2original
epochs/actualallrawreference/broadfacts+reasoning+LM+endtoendefficiencystillmissing.
DoNOTsubstituteoraclewordaccuracyforthegoal. Keepgoalactive,don'textendfailedarms.

# Position-varied curriculum512 VERIFIED; small LM gain, no robust fact win (2026-10-17)

Evaluator33609FINISHED0; independentresults9554PASSED. Allworkersfinished incl
CPUoracle14381/independentCPUoracleverification andanalyzer18674. DO NOTrestart/
extend/adopt thiscompleted512arm. Currentgoalturnmadeprogress:newmatchedtraining
andnegativegeneralizationevidence,fullgoalACTIVE/UNMET. No native/old changes.
Newreport docs/native256_position_curriculum_results.md. Same23.2Mreader/64:1/no
newKVorinferenceoperations. Positioncurriculumonly; noarchitecturefixclaimed.

Freshtrainedoffset0/8/16/24 exactfixed320->varied345/1024,paired99->66/512,allfour
48->26/256. Heldoffset4/12/20/28/32/36 exact180->233/1536,paired0->2/768,allfour
ZERO384both. ExactdeltaCIs trained[-1.2695,6.25]pp,held[-.3906,6.9010]pp,all[-.0391,
6.0547]pp; balanced16casebatchbootfixedweightsNOTseeds. Fixedoffset0peak225->91/256;
newnames0 208->65,wording0 215->88. Newvalues0/512/missing0/512/composed0/256BOTH.
Newowntrainfit46/128,1/64pairs,0/32allfour substantialunderfitnotcapacityproof.
Two-regAB/BA65/55of256vsold116/115,allfourZERO. Historicalfacts0/103,changed0/32,
compressedcolors0/8; missing21/25 unchanged. No robust factual improvement.

Full7831dev/1859388labels: fixedCE2.678522355/46.488952%,varied2.667544632/46.689825%.
Language+.200873pp doc95CI[.181533,.219036]pp;14/17sourcesbetter, memorysource-.0264pp.
ReasoningtokenmetricsNOTsolvedreasoningtasks. OriginalDALL225280tokens/3520blocks
exact,casefact83072/83072tokens7040/7040bodies. NOTgloballylossless/notreaderability.
All1311evalprotected+1260oldhashescheckedunchanged. Results/paired-position-analysis
undernewrun/evaluation; noextraLMortrainingupdatesafter512. Checkpointimmutable.

NEWCPUoracletools diagnose_native256_selected_unpack.py/verify_native256_selected_
unpack.py,output dump/native256-selected-unpack-diagnosis-20261017. GOLDsource
positionssupplied,NOTreaderperformance. Select125rowsoftrainedunpackW256->125->256
matchesfullfirstexpandfeaturesmax9.54e-6. ExistingPOSTdecoder norm/wordheadonPRE
decoderfeaturegets0/2048vsfullD2048/2048. Independentfullmap/sourceposition audit
PASSED,zeroupdates/unchangedD/guards. ThisdoesNOTprovefeaturesemptyortrainedbridge
impossible; copyingunpackaloneisNOTimmediatefix. Selectedprefix64kMACsvsfull4.096M,
2.08Mstoredparams EXCLUDESaddress/head/decoder andNOTfullspeed/KVproof. No model
changefromoracle. Nextseparatehypothesis explicitaddresssupervision+TRAINEDquery-
conditionedreadout/bridge,preregisterparam/cache/speed/generalitychecksfirst;not
implementedorautomaticallytrained. Avoidfullattention-positionexpansion/sidecache.

The setup/training record follows; allformerLIVEhandlesarenowterminalabove.

Previousgoalturn wasprogress: selective-readingdiagnosisverifiedlayoutdependence,
firstrecordreplay,novelwordfailure. GoalACTIVE/UNMET. NewturnisolatesTRAINpositions
atidenticalinferencebudget; no subagents, nooldsource/checkpoint/cache edits, nogrowing.
Native256 unchanged. NEWONEfrozenpackingarm; REUSEcompletedbalancedfixedfrozen512
control, doNOTrestart/extendcontrolorotherfinishedarms. Samecopiednative8750initial/
freshAdam/originalfrozenpack; diagnosticcontinuationNOTfresh2originalepochs.

New tools/native256_position_curriculum.py +evaluate/verify helpers;9CPUtestspass
inclactualmixedobjective/frozen/microgradient/Adamresumeandnewposition/order/prefix
checks. Protocol docs/native256_position_curriculum_protocol.md. Data/run dump/
native256-position-curriculum-{data,pilot}-20261017. Same1024units/4096QAexamples,
two8192exposurepasses,identicalwordbags/rawqueries/labels/orders/LR/LMsegmentsto
fixedcontrol. Trainoffset0/8/16/24 via(batch+epoch)%4,2differentpositionsperunit;
eachposition2048exposures,class2eachper16.16LM+16QA,micro8,.75LM+.25QA+.05frozenDrec.
Reader23209216/onecodeper64/raw64/zeroextraKV orproductionoperations. Prefix/D/pack
frozen; originalfeaturecachesread-only,new9600blocks600MiB. SourcesNOWfrozen:doNOT
edithelper/trainer/evaluator/verifier/protocolwhiletraining/evaluationlive.

Prepare52975FINISHED0; independentdataaudit87139PASSED allcases/answers/positions/
wordbags/batches/orders/cache,ALL9600newblocksreencodedexact,1260protectedhashes.
AdditionalCPUtraining-vs-regressionblockoverlapZERO. Fullsizezeroupdatepreflight
38366FINISHED0:PASSED exactunchangedweights,livepackgrad9.54e-7,allmicrograd8.85e-9,
publicpackedlogit8.58e-6,3.482GiBallocated/3.807reservedinclAdam-equivalentmoments.
CachedF/B2.008/.933s EXCLUDESAdam/cache/eval/save,not endtoendspeedclaim.

Actualfirsttrainer80680FINISHED0 paused1. Independentfirstaudit5615PASSED allAdam1,
4011LM/32QA/256rec,source/cursor/LR/controlorderslabels,packfrozen1260hashes. Saved
first-update-verification.json. SAMErun resume52221 FINISHED0 at512/~1.45updates/s.
Fulltrainingaudit16642FINISHED0:PASSED allAdam512/orders/LR/labels,1872347LM/16384QA/
131072rec,everyposition2048exposures,packingunchanged1260protected. CheckpointSHA
cf46921b3bbaebd2bd79c07df2cc47512a2897d3f74852062aef93e673eeed4d. DO NOTrestart/
extend. Evaluator33609 FINISHED0 andresults9554PASSED. Botholdfixed512and
newvaried512immutable. Finaldiagnosticoutcomesabove;doNOTduplicateevaluation.
Evaluation
mustcoverfreshoffset0/4/8/12/16/20/24/28/32/36 alignedsamewordbags,each256cases;
freshnames/wording/values/missingat0+12,composed12,bothfixedandvariedsamefreshcases;
all8oldselectionregressions,newactualtrainfit,controls,firstdecision,fulloriginal
7831dev+savedfixedcomparison/docCI,historical312,originalDalluniquetestblocks.
Noautomaticextensionafter512,nogeneralqualityclaimwithoutbroad/freshfullbudget.
New docs/token_memory_64_goal_status.md listsfullscope/evidencegaps. CPUactual
encoder9426432+reader23209216=32635648inferenceweightparams. Optionalasyncbudget
clarificationpending:23Mreader(existingbaselineconvention)vs23Mtotalinclencoder;
doNOTassumetotal23Mparity. Currentdiagnosticmatchesreaderbudgetonly. Primary
CompressiveTransformer/PointerGeneratorpapersreviewedwithscope/costcaveats in
docs/native256_compressed_reading_next_hypotheses.md; noarchitecturechangefrom
these papers claimed/implemented. Pureallrawmatched2epochreference stillmissing.

# Selective-reading diagnosis VERIFIED; layout dependence and first-record replay (2026-10-17)

Previousgoalturn wasprogress: balanced512 established narrow familiar-word reading,
NOTbroad quality. Thisturn isnewread-onlyfactorialdiagnosis,zeroupdates,goalACTIVE/
UNMET. No subagents, nooldsource/checkpoint/cache edits, nogrowing, native untouched.
Prepare11080/dataaudit87338/eval44680/results47985allFINISHED0. All1,154protected
hashesunchanged; unchangedreader/packingtensors asserted. No remainingworker.
DoNOTrestart/extend/adopt completed512arms. No newtrainingstarted thisturn.

New tools/native256_selection_diagnosis.py +verify helper,4CPUtestspass,protocol
docs/native256_selection_diagnosis_protocol.md. Output dump/native256-selection-
diagnosis-20261017. SAMEexactbalanced512 frozen/packingweights.8conditionsx64units/
256cases,2,048cases/1,024freshblocks; allactualblocksindependentlyreencodedexact.
Eightbalancedone-tokenvalues,47bodytokens,query77,onecode/64tokens/noextraKV.
Cleanpositionchangesandtwo-recordorderspreservewordbag/rawquery. Distractor/
two-recordcounterfactualskeepbothvaluewords,askbothqueriesagainstbothmemories.
Noanswerinprefixlogits,firstargmaxmatchesfreegeneration; noDfeaturestomodel.

Frozen/packing EXACTvalue+EOS cleanfixed235/238of256; shift8=35/38,shift16=22/23;
factfixed+spareafter222/222;sparebefore+factshift8=37/32;twoAB116/112,twoBA115/111;
novel8words0/0. ALLshift/two-record/novelall-fourZERO;fixed52/54of64,spareafter48/49.
Allknownblocked32/256;fixedshuffled0/256. Two-recordfirstvalueREPLAY448/443of512,
secondvalue7/512both;questionchanges12/9of256. Smallinteraction.0184/.0334frozen,
.0165/.0355pack NOTsuccess,blockedexact0. Query/entityselectionfails;fixedposition
successNOTestablishesbinding. Forcednovel8choice17/19of256 vsblocked38/42;output
palettepruningaloneisNOTsupportedfix. OriginalfrozenDALL65,536tokens/1,024blocks
exactBOTHarms,includingnovel/movedfacts; notgloballylossless/notreaderability.
Packingcode drift.96873%; frozenrecompute3.79e-7numerical. Caselevelsourcevalue
checks2,560facts+512sparesALLcorrect. FullLMnotrepeatedunchangedweights; prior
verified46.488952/46.481154% remains. No universal23Mcapacity/architectureclaim.

Report docs/native256_selection_diagnosis_results.md. NextSEPARATEcurriculumcould
varypositionswithbalancedknownvalues/replayfirst,thensame-wordbagentityswapsand
broaderwords. Noautomaticextension. Finalclaimstillneedsfreshmatched2original
epochs/broadunseenvalues,binding,reasoning,languageat23M/64:1. Historical61.013%
referenceisraw512PLUSoldermemory,NOTpureallraw1:1. Broaderobjectiveunmet.

# Balanced fixed-position512 VERIFIED; narrow closed-palette reading, no adoption (2026-10-17)

Evaluation55291FINISHED0; independentresults66579PASSED. ALLpilotworkersfinished;
doNOTrestart/extend/adopt. All702protectedhashesunchanged inclnative/botholdpilots.
Frozen/packing free value+EOS: fixedID229/234of256,newnames221/222,newwording232/234;
knownaggregate682/690of768. Bothchangedvaluescorrect302/310of384,allfour141/147of192.
IDallfour50/53of64; trainfit124/128both,30/32allfour. Matchedfirst64IDgenuine54/56,
blocked8/8,shuffled0/0; shuffleddonorvalue52/55. Valueassociationp≈.0001MCfloor
all3knowngroupsboth (fixedweights,NOTseeduncertainty). Basicvectorreadingworks
WITHOUTencoderretrainingornewattentionpositions.64:1/computation/KVbudgetretained.
ClosedpaletteONLY: unseen8words0/256,composed0/256,missing0/256both. Missingoutputs
repeatUNRELATEDrecord'svalue235/256both,soquery/entitybindingNOTestablished.
All13priorphrasegroups0exact; historical0/103notebook,0/32changed,0/8compressed
colors;missing21/25unchanged. Positiveone-recordsuccessisNOTmulti-entitybinding;
twoqueriesareparaphrases,sameonefact,no palette-valueddistractors. May detect
valuepresence; no mechanisticclaim ofposition-specificselection/generalcopy.
Packing+8/768knownhits=+1.0417pp,balanced-batch95CI[-.3906,2.4740]pp; ID+5ALLgreen
(13->18/32),othersunchanged. Full-dev45.601133parent,46.488952frozen/46.481154pack,
CE2.726427/2.678522/2.678798. Pack-frozen-.007798pp,doc95CI[-.014842,-.000956]pp.
NoLM-onlycontrol; extraLMupdatesconfoundgainoverparent,partial8750notfull2epochs.
FrozenDfacts44484/44484tokens,2560/2560bodiesALL; freshslot24values896/896ALLincl
heldwords. Wholeblocks163838/163840parent+frozen,163835packing;notgloballylossless.
Packingmediancode drift.769014%. Sourceencodedheldvaluesexist,readerclosedset.
SixCPUchecks,GPUzeroupdate,actual1andfull512trainingauditsPASSED. Newanalyzer
tools/analyze_native256_balanced_pilot.py independentlyauditspervalue/donor/
source-replay/decoder-slot/conditionalCI, saved evaluation/readability-audit.json.
Fullreport docs/native256_balanced_pilot_results.md. BroadergoalACTIVE/UNMET.
NextpossibleSEPARATEtest fixedfact+distractor palette,wordbagswaps forselection,
thenpositionvariation/broadervalues; NOTstarted. No subagents.

# Balanced fixed-position512 setup record (2026-10-17)

User requests NEW512perarm, balanced one-token values/fixed fact positions,
language replay/64:1onecodeposition, frozen vs trainable packing. Native256 and
all finished pilots immutable; no subagents. New tools/*native256_balanced_pilot.py,
tests/test_native256_balanced_pilot.py; protocol docs/native256_balanced_pilot_protocol.md.
Paths dump/token-memory-64-balanced-fixed-{data,pilot}-20261017. Same copied
native8750initial/freshAdam/originalpack; NOTfailed-pilot continuation/fresh2epochs.
8values red/green/white/black/water/fire/earth/air, eachONEtoken and twice EVERY
16readingbatch.1024fourcaseunits4096unique,2shuffledbatchpasses8192exposures.
Facts15..26,value24,query77; swapped memories differONLYat24.16LM+16QA,micro8,
sameLMorders/recipe. Trainingpositiveonly; heldnew8values/names/wording/composed/
missing and13previoussingle/two-record regressions. Trainfit128balanced.
SixCPUtestsPASSED, actualobjective/frozen/Adamresume/balance/fixedpositions.
Preparation88170FINISHED0:2944freshblocks/184MiBfeatures, oldcaches read-only.
Independentdataaudit19136PASSED ALLcases:everybatchvalue2each,position15/24/77,
source/serialization/block/target/2pass/order checks,702protectedhashesunchanged.
Fullsizezero-updateGPUpreflight65370PASSED:unchangedweights,livepackgrad1.91e-6,
micrograd1.02e-8,3.482GiBallocated/3.807reservedinclAdamscratch. Two cached F/B
samplesmicro4/8=1.627/.636s EXCLUDEAdam/cache/eval/save,not endtoendspeedclaim.
Actualfirst-update trainer71652FINISHED0 pausedfrozen1. QA8.5178/LM2.8004finite;
packingfrozen. Independentfirst-updateaudit99456PASSED:allAdam1,4011LM/32QA/256rec,
balanced8values2each,exactcopiedparent/pack,702hashesunchanged. SAMErun resume20427
FINISHED0:BOTH512updates/~1.73updates/s,lastQA.0746frozen/.0752packing. Each16384QA
labels(8192values+8192EOS),LM/recasbefore. EndsourceguardsPASSED. No furthertraining
authorized; doNOTrestart/extend. Fulltrainingaudit65764PASSED:both512/allAdam512/
1872347LM/16384QA/131072rec,allorders/LR/cursors/balance/702protectedhashes.
Frozenpackexactunchanged,trainablepackchanged. Evaluation55291andindependent
results66579FINISHEDsuccessfully; terminalresultsabove. Checkpointsimmutable:
frozenSHA d6bde428995aa0ed3b00929c4ba1b0cdfd02c1948a2731cda325ea66b30bb24a,
packingSHA 1e5eee2c6b73444382cbdfbb1ba2b1af77020e92413948374f95461438401452.
Do not
duplicate workers or edit guarded helpers.
Prespecified10k unit label-flip association control added before training; constant
answers havezeroassociation, fixedcheckpointtestNOTseeduncertainty. Sources now
frozen; do not duplicate or edit guarded helpers while live.
No quality result; old completed pilots remain finished. GoalACTIVE/UNMET.

# Positive single-fact packing512 VERIFIED; common-value collapse, no adoption (2026-10-17)

Evaluator67735FINISHED0; independentresultsverifier99290PASSED. ALLworkersfromthis
pilotfinished. DoNOTrestart/extend/adopt either512checkpoint. All346protected
hashesunchanged(incloriginal159/native/oldpilot/caches). Both14/1536freshsingle
positive test,0/768correctchanged-valuepairs,0/384allfour; trainfit2/128but0/64pairs.
Trainmode'narrow candle'94/4096;constantmodebaseline12/1536,modelssame12hits
unchangedunderfactswap. Newmissing0/52both,all52nonunknown. Oldsame-two-id
falseunknown228/230->0,positiveexactstill0/230,missing24/26->0. Two-recordtest
positive4/1380frozen,5/1380packing,0/690pairs,0/156missing. ZEROallfoureverygroup.
Historical0/103notebook,0/32changed,0/8compressedcolors;missing21/25unchanged.
Full-dev parent45.601133/frozen46.618027/packing46.619694%;CE2.726427/2.671201/
2.671137. Packing-frozen+.001667pp,doc95CI[-.001184,.004485]pp(fixedweights,not
seeduncertainty). BothgetextraLMupdates; noLM-onlycontrol, don'tcreditcurriculum.
FrozenDfact39947/39948tokens,1919/1920bodiesALL;wholeblock122873/122880parent+
frozen,122872packing;1913/1920exacteach. Packingmediancode drift.787352%;active
gradients/changedweightsdidNOTestablishreading. Notgloballylossless; no universal
capacity/packing-impossibilityclaim.64:1onecodeposition/KV/inferenceopsunchanged.
Supplement behavior-and-prior-audit.json rederivedALLhistoricalscores,constant
prior,swaps,recipehashes; mainindependentauditchecksdecodedoutputs/language/recon/
sourceguards. Fullreport docs/native256_positive_pilot_results.md. Broadergoal
ACTIVE/UNMET. No next run begun; possible nexttestbalancedone-tokenvalues/fixed
positionbeforevariation,NOTautomaticcontinuation. No subagents.

User explicitly requests a NEW512-update-per-arm pilot, positive single-fact first,
language replay, frozen vs trainable packing,64 tokens->onecode/attentionposition.
No subagents; native256 and all completed diagnostic runs remain immutable.
New tools/*native256_positive_pilot.py and tests/test_native256_positive_pilot.py;
protocol docs/native256_positive_pilot_protocol.md. Six CPU tests passed.
New data/run paths: dump/token-memory-64-positive-pack-{data,pilot}-20261017.
All512 updates positive single-record warmup,1024new4caseunits/4096uniqueexamples,
two shuffled passes/8192readingexposures. No unknown QA labels; LM unchanged.
Same copied native8750 initialization as completed packing pilot, fresh Adam;
NOT a completed-native or fresh two-epoch architecture comparison. Reuse old
frozen-prefix features read-only; final codes recomputed with each arm's pack.
Fresh heldout single groups, mixed missing tests and old two-record regressions.
Preparation54121finished0:3072freshblocks,192MiBfeatures; oldfeatures reused.
Dataaudit2165passed ALLcases/2passes/orders/346protectedhashes. FullsizeGPU
preflight36792passed,unchangedweights,live/cachedpackgrad1.43e-6,micrograd1.12e-8,
3.490GiBallocated/3.814reservedinclAdamscratch. Bodytimings contended/excludeAdam,
not speedclaim. Actualfirstfrozenupdate trainer65621paused1finished0. Independent
Trainingaudit94918passed actualstep1:allAdam1,4011LM/80readinglabels,exact copied
parent,packingunchanged,346protectedhashes. Same-run resume trainer49816FINISHED0:
BOTH512updates,~1.67frozen/1.77packingupdates/s,sequentialcachewarmnotcontrolled
speedcomparison. Each1872347LMlabels/39512reading/131072reconstruction. No more
training authorized; do NOT restart or extend. Trainingaudit80543PASSED:allAdam512,
allorders/cursors/LR/labels/346protectedhashes,exactsameinit; frozenpackunchanged,
trainablepackchanged. Evaluation67735andindependentaudit99290finishedsuccessfully;
terminalresultsabove. Checkpoints frozenSHA624cdc884cf6c6027fa60239640c25d94cce3d93ae58dd3f3e979d1a7fa8630b,
packingSHA07de2f3df4447b8926734cfaac846117d1b4ff4cd48f914ab9d1560c3396ff20.

# Active goal: native256 frozen/trainable packing512 VERIFIED; no retrieval win (2026-10-17)

Evaluator89128FINISHED0; independentresultsverifier30223PASSED. Supplementary
behavior-and-fact-audit.json rederived ALLhistoricalscorekeys and actualfacttoken
reconstruction. ALLworkersfinished. DoNOTrestart/extend thiscompleted512pilot.
Full-dev parent/frozen/packing45.601133/46.602000/46.603452%,CE2.726427/2.672051/
2.671987. Packing-frozen+.001452pp,doc95CI[-.001675,.004556]pp(fixedcheckpoints,
NOTseeduncertainty). BOTH0/230positiveeachheldoutgroup,0/104actualtrainfitpositive,
0/51positivefourwayeachtestgroup. Testmissing152/156BOTH;familiarfalseunknown
228/230BOTH,trainfit98/104;relaxedvaluepresentZERO. Trainfit4all-fourunitsareONLY
single-recordall-missingunits,NOTreading. Historical0/103notebook,0/32changed,
0/8compressedcolorboth;21/25missingunchangedfromparent. No useful factualwin.
FrozenDfactreconstruction24963/24964tokens,895/896bodies ALLarms;notlossless.
Wholeblock57342/57344parent+frozen,57341packing;packingmediancode drift.930607%.
Packinggradientsactive/weightschanged butdidNOTunlockreadingin512. Original
native/source/cache/checkpoints/logs ALL159protectedhashesunchanged. Onecode/64,
noaddedproductionparams/KV. Report docs/token_memory_64_pack_pilot_results.md.
GoalACTIVE/UNMET. Partialnative8750init limits inference;doNOTclaimuniversal
capacitylimit or packing-impossibility. Next should establish actualpositive
single-factreading andavoidunknownshortcut BEFOREscaling; no new runauthorized
orstarted beyondcompleted512. No subagents.

User explicitly authorized new data/code, train512 perarm, test and report;
keep64tokens->one256code->oneattentionposition and leave native run untouched.
Read-only Windows process inspection foundNOactivePython/native trainer; PID1988
is dwm.exe, native status stale8950/log8954, checkpoint8750. Original native
sources/run/checkpoints/logs all protected by hashes. Do NOT restart native run.
New separate tools/*token_memory_64_pack*.py, fourCPUtestsPASS incl mixed-gradient,
frozenpack and exactserializedAdamresume. Protocol docs/token_memory_64_pack_pilot_protocol.md.
Data preparation2169FINISHED0: dump/token-memory-64-pack-data-20261017;8192fresh
readingexamples,128one-recordupdates then384two-recordupdates,16LM+16QA/update,
sevenheldoutgroups256each,trainfit128. Genuine64blocks,queryraw/answerfits64;
no decoder-success filtering.137033FP32pre-packfeatureblocks(~8.36GiB) include
8192originaltrainsegments andall7831dev; prefixfrozen sofeaturecachevalid while
packchanges. Original language examples also useCURRENTpackingweights.
Run dump/token-memory-64-pack-pilot-20261017 contains immutablecopyinitial.pt
of native8750; NOTcompletednative/fresh2epochs. BothfreshAdam. Frozenarmreader
only; packingarmreader+existingcompress.token/span_map (2080000weights), no new
productionparams/slots. .75LMCE+.25answerCE+.05frozenDreconstruction(fourblocks),
readerLR3e-5cos3e-6warm30,packLR.1reader,FP32AdamW.01clip1,micro8. Dtrainingonly.
Initialcheckers53149/23217 failed ONLYquerypunctuation parser andbitwise equality
acrossdifferentSDPAshapes; fixedcheckersonly,dataunchanged. Dataaudit8515PASSED,
159protectedfilesunchanged. Fullsizepreflight89997PASSED: exactinitialweights,
codeerror5.72e-6,livepackgradient9.54e-7,micro4/8allgradientmax1.02e-8,packed/public
logitmax1.24e-5,heavypeak3.485GiBallocated/3.814reserved. Microbatch8 selected,
twoheavybodytimingsmicro4=1.3912s/micro8=.6252s excludesAdam/I/O/eval/save.
Trainer14198FINISHED0pausedfrozen1. Auditor19949PASSED:4011LM/76QA/256reconstruction
labels,allAdam1,packingunchanged,159protectedunchanged. Resumedtrainer83488FINISHED0:
frozen512andpacking512,~1.64/1.74updates/s,finalQA1.2840/1.2832;NOTfreequality.
Trainingauditor64198PASSED: each1872347LM/34264QA/131072reconstructionlabels,
allAdam512,allorders/rates/labelschecked,frozenpackexactunchanged/jointchanged,
159protectedunchanged. Evaluationandindependentresultsverificationcompleteabove.
Alloldsource/model/cacheprotected. DoNOTrestart/extendpilot. No subagents.
GoalACTIVE/UNMET. No automatic extension or adoption.

# Active goal: four-way binding pilot VERIFIED; no robust binding win (2026-10-16)

Trainer95505FINISHED0,1810updatesbotharms,~3.6updates/s; trainingverifier27036
and independent results verifier PASSED. Both6,775,592LMlabels/163,376readinglabels,
28,960examples/arm,allAdam1810,orders/cursors/LR/sources/124originalhasheschecked.
DO NOT restart/extend this completed pilot. Both checkpoints immutable. Binding
loss improves some ordinary single-answer groups but fails robust counterfactual
binding: test positive exact ordinary136/460,binding163/460; test bindings22->32,
names90->110,wording101->125; both remain0/460newvalues and0/460composed. Four-way
test has binding1/57all-four only onwording andZERO all-four othergroups; ordinary
ZERO all. Full-language accuracy ordinary53.541058%,binding53.531969%,parent
53.504594%; fixed-checkpoint binding-ordinarydocumentCI[-.01930,.00175]pp.
Noquality/adoption claim. AdditionalCPUzero-updategradientdiagnosis bindinggradnorm
1.6059 vsanswerCE5.1020,ratio.31476,cosine.18336 isnotconvergenceevidence.
Primary-sourceICAE/AutoCompressor/LoMA caveatsinprotocol.

User requested a fresh native256 reader test: src/models/token_memory_64_native256
is new/separate; old256 codes enter attention directly, raw embeddings are256,
onecode/64tokens/raw64mask,21layers/4heads,exact23209216parameters. No encoder
retrain yet: this isolates the old256->512adapter hypothesis. Zero-update
preflight1792164038079915100 PASSED: isolated-vs-packed logits2.384e-6,grads
3.353e-8,initialweightsunchanged; heavy1.917GiBallocated/3.719reserved with
Adam moments. Cached body samples1.120/1.018/1.022s effective32 micro4 excludes
Adam/cache/eval/save; no speed/quality claim. Fresh2epoch run NOW started in
dump/token-memory-64-native256-unified-v1, PID6320; initial full-dev validation
atstep0 is running. Do not start another worker or edit native source/run files.
Protocol docs/token_memory_64_native256_protocol.md.

Previousgoalturn wasprogress: retrieval-only comparisonverified, some familiar
reading but0unseenvalues/languagecost. Newturn isolatesquerybinding atsameinference
budget. GoalACTIVE/UNMET, nooriginalsrc/checkpoints/caches modified, nogrowing.
No subagents. Protocol docs/token_memory_64_query_binding_protocol.md.

New read-only factorialdiagnosis completed81222exit0, independentaudit3261exit0.
tools/token_memory_64_query_binding.py +diagnose/verify helpers,5CPUtestspass.
For2recordpositivecounterfactualpairs askBOTHquestions againstBOTHswappedmemories,
identicallength/positions/rawqueryphase/commonanswerprefix. Allgenuineoriginalcodes.
Onlystructurefilter: train3620quartets/14480examples;testid60,binding62,names56,
wording57,values51,composed36; dev27/39/30/28;trainfitfirst32. Fullcaseauditpasses.
Bothraw/code specialists haveZEROall-fourcorrect EVERYgroup incltrainfit.
Familiar raw117/240answers,6/120bothquestions,questionchanges14/120;
code76/240,3/120bothquestions,questionchanges49/120. Raw121wrongrecord/2outside,
code86wrongrecord/78outside. Firstdecisionmeaninteraction raw.002926/code-.004047.
Allblockedcontrolsinteraction0,identicalmemory-swappedoutputs. Saved under
dump/token-memory-64-query-binding-diagnosis-20261016. Original124hashesunchanged.
Strongerevidenceofbindingfailure,NOTuniversalmodelcapacitylimit.

New train_token_memory_64_query_binding.py uses3620quartets2passes=1810updates/arm,
16originallanguagesegments+4quartets(16readingexamples)/update,micro8. Bothinitfrom
COMPLETEDretrieval-onlyCODEspecialist2048 withfreshAdam, NOTfreshoriginal2epochs.
.75LMCE+.25answerCE;bindingarm+.25meansoftplus(1-C), ordinary0. C=(d00-d01-d10+d11)/4
withdfixedvalueA-minusBlogitatfirstdivergence. Cancelsmemory/query/sharedpriors.
Newquartetspositiveonly;missingregressionstillmustcheck. Legacypositions,
23209216params,zeroextraKV/productionoperations. LR3e-5cos3e-6warm30,FP32AdamW.01,
clip1,seedsLM4701660/quartets4701661+epoch. Sameorders/labelsbetweentwoarms.
Uniqueformat raw64-four-way-binding-diagnostic-v1; neverresumeoriginalentrypoint.
5CPUtestspassinclactualmixedobjective/microbatch/exactserialized8stepAdamresume.
Fullsizezero-updatepreflight51447finished0,allparamgradientmax1.28e-6,exactinitial
weights unchanged;heavy2.239GiBallocated/2.566reservedinclAdamscratch. Threebody
samplesmedian.25775/.25693s excludesAdam/eval/save/encoder,nospeedwinclaim.

Run dump/token-memory-64-query-binding-pilot-20261016. Firstactualordinary1 paused
22377finished0; independenttrainingaudit92583finished0(allAdam1,3783LM/100reading
labels,cursor/source/124originalhasheschecked). Resumed trainer95505 FINISHED0.
Trainingverifier27036FINISHED0. Strictsourcesguarded:doNOTedittraining/helper/
production/evaluationfileswhileevaluationlive. Evaluatorrunning; aftercompletion
run verify_token_memory_64_query_binding_results.py.
Evaluationmustinclude fulloriginaldev,alloriginalretrievalgroupsinclmissing/new
values,allfour-waygroups+blockedcontrols,historical312. Noqualityresultyet.

# Active goal: retrieval-only raw/code capacity comparison VERIFIED; partial reading, no adoption (2026-10-16)

Evaluation92819finished0; independentresultsverification saved PASSED (session68549
no longer active). All124protected original hashes unchanged. DoNOTrestart/train
this completed diagnostic. Source/model/cache unchanged; goalACTIVE/UNMET.
Both2048-update specialists: familiar positivefreeanswers raw351/460 vs code224/460;
heldbindings347/460 vs51/460; newnames236/460 vs177/460; newwording360/460 vs223/460;
newvalues56/460 vs0/460; combined12/460 vs0/460. Strict value+EOS. Parent0allgroups.
Familiar paired171/256 vs60/256,missing52/52 vs3/52. Code first64 familiar controls
28/57positive enabled vs4/57blocked,8/32pairs vs0;swapexactdonorreplay.
Descriptive binding-breakdown.json: raw familiar single230/230,two121/230;
heldbindings single230/230,two117/230. Code familiar148/230,76/230;held39/230,12/230.
Thus23Mcanlearnnarrowretrieval; binding fails evenraw, codehasadditionalgap.
No universalcapacity/architecturecause established. Finitevalues/partitionlimits.
COMMONplain64 language parent53.504594%,rawspecialist52.081115%,codespecialist51.971563%.
Code-parent-1.533031pp doc95CI[-1.579569,-1.486134];raw-parent-1.423479pp.
Both0/103historical,0/32changed. DoNOTadopt/automaticallyextend specialists.
Fullreport docs/token_memory_64_retrieval_research.md updated; alloutputs/teacher/
controls/sourceguards and independent-verification.json under run/evaluation.
Next needs binding/valuegeneralization ANDlanguagepreservation; no next run begun.

Previousgoalturn madeprogress: reconstruction512pilot completed/verified,butno
factwin; no extension. Newturn testsdirectretrieval-onlysupervisionbeforeblaming
23Mcapacity. GoalACTIVE/UNMET. Modestcompute/VRAMallowed,maincompressionretained.
Nooriginalsrc/checkpoint/cache/dataedits,nogrowingresurrection,nosubagents.

New tools/token_memory_64_retrieval_only.py +prepare/train/preflight/verify helpers
and tests/test_token_memory_64_retrieval_only.py. Protocolwrittenbeforetraining:
docs/token_memory_64_retrieval_only_protocol.md. Data under dump/token-memory-64-
retrieval-only-data-20261016 prepared82743finished0.32768trainexamples/16384pairs,
4x256devgroups,6x512testgroups,54437genuineFP32blocks;eachgroup'spromptsandfactblocks
disjoint. One/twofacts,color/multiwordaccess/location;twofactswapskeepwordbag;
20%pairsoneentitymissing (~10%examples). Train16names,8colors,32accessphrases,
64locations. Testseparatesnewblocks/bindings/names/wording/values/composed.
NoheldbindingappearsANYtrainfact,distractorsincluded. Namescooccurwithinbinding
partitiongroups:syntheticlimitation. Wholequeryraw,allfactsold,targetsfitraw64.

Allcaseindependentaudit69496passes answerregex/serialization/token/block/prefix/
partitionchecks. Initialauditorincorrectlymatched'red'inside'stored';fixedONLY
auditortowholewords,dataunchanged. OriginalD allfacttokens/bodyexactALLdev/test;
train684291/684318tokens,32754/32768bodies;errorsretained. Notgloballylossless.
124protectedhashesunchanged.4CPUtestspassinclactualraw/codepublicgradients,
future/padding,actualobjective/microbatchandserializedtwoepochexactAdamresume.
FullsizeGPUpreflightfinished0,gradmax<1e-5,unchangedweights. Heavyraw2.307GiBalloc/
2.920reserved;code.872/.939inclAdamscratch. Threesamplebodytiming .1148/.1030s
EXCLUDESAdam/E/I/O/eval/save;notendtoendspeedproof.

Run dump/token-memory-64-retrieval-only-pilot-20261016. BOTHidenticalCOMPLETED
plain64step48762weights,freshAdam,2048updates/arm,twoNEWdatapasses65536examples/
354284answerlabels each. NOTfresh2originalepochs. Answer-onlyfullvocabCEinclEOS,
NOlanguage replay,FP32,AdamW.01,clip1,warm50,LR3e-5cos3e-6,batch32micro16,
samepairorder4701650+epoch. codearmoriginal64forward/legacy/onecode1slot/noextraKV.
rawarmSAMEsourcehistory/questions asraw<=512forward,no codes,highercomputecontrol
ONLY. Both23209216params,rawcontrolconfig512/codeconfig64;strictuniquefmt/armguards.
Neverresumeviaoriginaltrainer. Rawcontrolisnotnewproductionarchitecture.
Firstrawstep1trainer50340paused; auditor66792finished0labels161/allAdam1/cursor
andparent/sourcechecks. Resume74450finished0,2048each,~7.3raw/8.8codeupdates/s.
Evaluatorandindependentresultsverifierwritten/syntaxchecked; fifthCPUtestofactual
teacherrecord/freeEOSscoringpasses (initial4unchanged). Trainingverifier58256
finished0:both2048/65536examples/354284labels,allAdam2048,allpairorders/LR/cursors
match,124protectedunchanged. Evaluator92819 and independentresultsverifier complete.
Full evaluation includes
fullheldoutfree/teacher/counterfactual/blockedchecks+fulloriginaldevinCOMMONplain64
inferenceforeacharm'sweights, plushistoricaltransfer. Verified results above.
DoNOTduplicate/restart. All workers finished.

# Active goal: code-content reconstruction diagnostic VERIFIED; no factual win (2026-10-16)

Latest user allows slightly more compute/VRAM if the main compression advantage
remains. Asked whether23M is simply too small. Answer: constrained retrieval is
feasible; historical same23209216-param mixed-memory record shows100% familiar/
86.13% unseenbindings+wording with16answerwords andextra training. Not broad QA or
proof currentreader sufficient. Currentparentrawcompletion103/192 vscompressed0
also argues against parametercount alone. Preserve retired growing families.

New tools/token_memory_64_reconstruction.py, prepare/train/preflight/evaluate/
verify helpers and tests/test_token_memory_64_reconstruction.py. Data prepared
dump/token-memory-64-reconstruction-data-20261016:4096train/512dev/512test genuine
full64blocks, balanced17sources, dedupall5120blocks, originalcachetrain/dev exact,
test frozenE. No decoder-success filtering. Independentfullblock/cache/hash audit
passed;124protectedfilesunchanged. OriginalD reconstructs all261843train/32728dev
contenttokens (>2),32735/32736testcontenttokens;specialtokenerrorsremain. NOTlossless.

Protocol docs/token_memory_64_reconstruction_protocol.md written before training.
Twoexactcompletedplain64parentcopies,freshAdam;512updates/arm,2auxpasses,NOTfresh
baseline2epochs.16originalLMsegments+16auxblocks/update, .5LMCE+.5reconstructionCE.
EachblockoneallBOSposition-only andonecausalteacher-prefixexposure;524288auxlabels
half eachmode. Blockedarm hascodekeymaskedbutvisiblecount1/positions1..64 retained;
codearm enabled. AllBOSinputscontainnotargettext. Onecodeposition,oldlegacyforward,
no productionexpansion/decoderfeatures/newKV.23209216parameters. FP32,AdamW.01,
clip1,cos3e-5to3e-6,warm30;language seed4701604,aux4701623+epoch.

4CPUtests pass inclindependentcausalblockedallparametergradients,sourceisolation,
actualobjective/microbatch andserialized8step2epochAdamresume. Fullsizezero-update
preflight18240finished0,exactweights,graderror0,heavy2.238GiBallocated/2.535reserved
inclAdamscratch. Threebodytimings .2094/.2050s EXCLUDEAdam,not speedclaim.
Trainer11990pausedactualblockedstep1; independent6572passedallAdam/cursor/labels.
Resumed trainer session94055 FINISHED0,512updatesperarm,~4.6updates/s. DoNOTduplicate.
Run dump/token-memory-64-reconstruction-pilot-20261016. Sourceguardsstrict;
doNOTedittrainhelper/guardedsources whilelive. Trainingverifier28853finished0:
each1868770LMlabels/524288auxlabels,262144eachmode,allAdam512;124protectedunchanged.
Evaluator65892finished0; independentresultsverifier72327finished0. Allworkersdone.
DoNOTrestart/duplicate/automaticallyextend. Eval includesparent+botharms,
correct/blocked/shuffledcodes,allBOSandteacherqueries,dev/test/trainfit,fullLMdev,
pairedfreefacts/historical. ReconstructionaloneisNOTretrievalwin/goalcompletion.

Verified results: parent/blocked-trained/code-trained full-dev53.504594/53.407519/
53.312326%; CE2.214283787/2.221130414/2.227575978. Code-parent accuracydelta-.192268pp,
doc95CI[-.219316,-.164559];code-blocked-.095193pp CI[-.114195,-.076575].
Position-onlytestcontentcorrect728/1052/1888 of32736. Code-trained withcodeblocked
876/32736,shuffled834/32736;6.5143% on128owntrainfitblocks. Zeroexacttestblocksall.
Code-trainedteachertestcontent56.3691% vs54.8021%codeblocked;notfreefactaccuracy.
Allarms0/460positivetest,0/52missing,0/256pairs;newarms0/230dev,0/103historical,
0/32changed. Somecodecontentdependence,NOusefulfactualwin;goalACTIVE/UNMET.
Factsuite train_fit refersOLD bindingtrainexamples,NOT thispilottrainingdata;
reconstruction train_fit isthispilot's128actualtrainblocks. DoNOTconflatethem.
All124protected unchanged; nooriginalsrc/model/cache edits. Reportupdated.
Nextcapacitytestcoulduseexplicitretrieval-only raw/code matchedfacts withfresh
codes, answer-onlytargets andprogressivedifficulty. No such newrunstarted.
Historicalretrieval-onlyrecordrecheckedfromactualsavedpreds/source:512/512new
histories,464/512heldbindings,105/512newwording;8fixedkeys,16single-tokenvalues,
128distinctcodebank,2000extraearlybaselineupdates. Narrowcapacityevidenceonly.

# Active goal: decision-loss diagnostic VERIFIED; no fact-reading win (2026-10-16)

Latest goal turn made real progress: implemented, trained and independently
evaluated one new loss-only arm. Goal ACTIVE/UNMET. No production src/model changes,
growing resurrection, original checkpoints/data/cache changes or subagents.

tools/train_token_memory_64_decision.py +preflight/evaluate/verify helpers and
tests/test_token_memory_64_decision.py. Run dump/token-memory-64-decision-pilot-20261016.
Actual same completedplain64 parentstep48762, freshAdam, SAME512batches/order/LR/
labels as completedbindingordinarycontrol (reused, NOTretrained). .9languageCE+
.1readingCE+.05 fullvocab firstdivergenceCE. Additionaldiagnosticcontinuation,
NOTfresh2epochoriginalcomparison.23209216params,legacypositions,onecode/slot,
noextraKV/inferencecomputation.8192decisionterms/82960reading/1868770language labels.

4CPUtests passed incl coefficient0 exactoldordinaryAdam parity, independently
unfoldedcausal allparameter gradients and actual8stepserializedAdamresume.
Zero-updatefullsize GPUgrad maxerr3.34e-6; unchangedparent;heavy2.238GiBallocated/
2.713reservedincl2Adam+scratch.3samplebodytime.209667/.209033s no speedwinclaim.
Preflight21829finished0. Trainer31093pausedactual1;firstauditorcounted8fixedFFN
buffersasparameters incorrectly, fixed auditor ONLY. Retry62688passedstep1.
Trainerresume46500finished0at512(~4.53updates/s), verifier34130finished0:allAdam512,
allbatches/labels/LR/cursor matchcontrol. Eval72226 and independent16509finished0.
No remainingworker; doNOTduplicate/restart/continue this failedarmautomatically.

Ordinary -> decision full-devCE2.217794725->2.219261239;acc53.479209->53.466786%.
Accuracydelta-.012423pp doc95CI[-.022895,-.002067],onefixedcheckpoint/noseedproof.
Parent53.504594%, goal61%+unmet. Positiveexact3/460BOTH, paired0/256BOTH;
missing15/52->32/52, FALSEunknown120/460->298/460. Aggregate18->35exact is ONLY
abstentiongain. Firstdecision89/512->94/512;changedoutputs28->15/256 (0correctpairs).
Trainingfit5/115positive and0/64pairs;historical0/103 and0/32. Blockedunknown84/128.
Prefixcompletion near128->127/192; compressed14->12/192,0pairs. RawQA near12->28
(3/96pairs), far1->3/192(0pairs);shortcapformatlimitationstillapplies.
All124protectedhashesunchanged. Fulloutcomes in docs/token_memory_64_retrieval_research.md.

Nextinvestigation (NOTimplemented): directposition-specific code-reading/
reconstruction curriculum usingexistingraw64 duringTRAININGonly, to establish
learnabilityofcodeaccess beforemorelossweighttuning. No decoderexpandedpositions,
reconstructedrawhistoryoradditionalKV inproduction. ToyreconstructionaloneisNOT
goalcompletion; requiresbroadfreefacts/counterfactuals/languageandfreshfullbudget.

# Active goal: raw64 reading/visibility controls VERIFIED (2026-10-16)

This goal turn made new diagnostic progress, not a quality win. Prior cleanup
recheck was no retrieval progress. Goal ACTIVE/UNMET. No new training or production
model/source changes; growing families stay retired. All124 protected hashes match.

New tools diagnose_token_memory_64_raw_reading.py, _completion.py, _visibility.py
and verify_token_memory_64_reading_controls.py completed. Outputs in dump/
token-memory-64-{raw-reading,completion,visibility}-control-20261016. Sessions
77542,42116,13984,34182 all finished0; no worker remains. Do not duplicate.
96counterfactualpairs/192cases, one/two records, familiar/held firstnames,3relations.
Two-recordpairs swapvalues withsamewordbag. Far adds128filler tokens, samephase;
facts in firstold64code, secondcodefiller. Allraw64 generation stayswithinwindow.
OriginalD reconstructs2444/2444facttokens,192/192bodies; diagnostic ONLY.

Value-onlyquestion format often ignored; 8token cap truncated sentences. Added
conditionalcompletion prefix (name/relation ONLY, no value), allsame192facts;
raw33..47tokens, targets<=5 inclEOS. Parent/ordinary/interaction exactnear
103/128/126 of192 vs compressedfar0/14/13; allfarpaired0/96. Blockedvalueprefix
hits0/16/16 with0pairs and0EOS: isolatedhitsarenotretrieval. Original512 nearby
valueprefix141/192 (133exact), distant110/192 (0exact;continuedfiller),37/96pairs.
Original512 seesentirefartextraw; notmatchedraw-onlytrainedbaseline.

Same-weight visibilityoracle: loadexactplain64 tensors intoOriginal512forward,
extraattention ONLYdiagnostic. All576nearoutputs replayexactly. Farvalueprefix
parent0/192 ordinary3/192 interaction2/192, all0pairs. Untrainedlongrawdistance
remainsconfound, so DON'Tclaimcompression alone causesgap. Bothlearneddistance/
reading andcodeaccess needwork.2CPUchecks+independentallcase/output/codec/guard
auditpassed. No productionKV/params/forwardchanged, no extra original training.

Next concrete hypothesis (NOT implemented): sameparent/paireddata/order/512steps
ascompletedbindingordinarycontrol, .9languageCE+.1readingCE plus .05fullvocabulary
firstdivergenceCE instead ofrelativeinteractionloss. Reuseordinarycontrol, don't
retrainit. Needfreeanswers/changedpairs/missing/full-dev andrawcontrols, notmerely
lowerdecisionloss. Originalfresh1000answerweightfailed; thisnewinit/data differs,
stilldiagnostic only. Report docs/token_memory_64_retrieval_research.md updated.

# Active goal: paired raw64 binding diagnostic VERIFIED; no retrieval win (2026-10-16)

Evaluation and independent verification have completed successfully. Parent /
ordinary / interaction full-dev accuracy: 53.504594 / 53.479209 / 53.474423%.
Held-out positive exact: 0/460, 3/460, 4/460; paired both-correct: 0/256 ALL.
Missing exact: 0/52,15/52,17/52. Historical0/103 and0/32 ALL. Interaction
training-fit pairs2/64, ordinary0/64. No useful held-out retrieval win; goal unmet.
All124 protected hashes unchanged. Do not restart completed512-update arms.
Read-only attention diagnostic finished: layer7head8 positive fact mass
.1065/.1676/.4344, yet no held-out pair success; attention is not correctness.
Artifacts run/evaluation and run/attention-diagnosis; full outcomes in research MD.
Next: raw-window versus compressed-fact diagnostic with short value answers and
no generation rollover, to distinguish basic reading from code access. No model
or original source changes, no new training prescribed by this diagnostic.

Last goal turn (answer weighting) was progress, not success. This turn tests
explicit fact binding at identical plain64 inference cost. Goal ACTIVE/UNMET.
Original models/data/caches/encoder sources unchanged; no growing resurrection.

New tools/token_memory_64_binding.py +prepare/train/preflight/evaluate/verify
helpers operate outside src to preserve strict original checkpoint semantics.
Data dump/token-memory-64-binding-data-20261016:2048train pairs,128dev,256test;
rawprefix/coordinates/teacherprefix IDENTICAL atfirst differing answer token,
onlyOLDfacttext differs. Counterfactualvalue pairs plus20%pairs withamissing side
(10%examples). Trainrelations color/location/access;test addsuntrainedtool VALUEONLY.
Train/dev bothsentence/value formats;testformat iscorrelatedwithrelation, so
don'tclaimheldoutfullsentence tooltransfer. Splitnames/questiontemplates disjoint.
18357genuine FP32cached64blocks. Independentallcaseblock/prefix/label auditpasses.
OriginalD reconstructsall192658train/11970dev/22728testfacttokensexact;diagnosticonly.

IMPORTANT initialization differsfromearlierfreshpilots: SAMEcopyofCOMPLETEDplain64
step48762 intoeacharm;freshAdam. Thisisadditionaldiagnostic continuation,NOTfresh
twoepochfinalcomparisonandNOTcompletionoftheuser'squality/budgetgoal. Nooriginal
checkpointchanged. Botharms512updates,twoepochsNEWreadingonly,16originallanguage
segments+8readingpairs/update,samebatches/order/labels. .9languageCE+.1readingCE;
interactionarmadds .05*softplus(1-C), C=pairedfirstdivergencelogitcrossdifference.
Sharedtokenpriorcancels; fullvocabularyaccuracy/freegenerationstillrequired.
FP32/AdamW.01/clip1/warm30/lr3e-5cos3e-6. All23209216parameters; oneposition/code,
sameoriginallegacypositions/publicforward;noextraKVorinferencecalculation.

FourCPUtestspass inclactualpairedloss/Adamserializedresume,priorcancellation,
causalpacked/publiclogitsandfuture/padisolation. Fullsizezero-updateGPUpreflight
passes,exactparentweights;2.149GiBallocated/2.486reserved inclAdamscratch. Updatebody
timings .2463/.2363s only4samples andnoAdam; no speedwinclaim.
Trainer29197pausedafteractualstep1(allAdam1/cursor/labelchecksverified),resumed
session80863FINISHED0:512updatesperarm,~4.6updates/s. Run
dump/token-memory-64-binding-pilot-20261016. Botharmssaved512;doNOTduplicate/restart.
Trainingverifier+evaluation finished successfully; independent verifier passed.
Evaluationcoversparent+botharmsfulldev,dev/test/trainfitfreeanswers,blockedtest,
historicalcases,firstdivergencefullvocabaccuracy andpairedsuccess. Results above.
Sourceguardsstrict;nevereditguardedhelperswhilerun/evaluationislive.

Descriptive geometry recorded indata/representation-geometry.json:changedfactcode
medianrelativevectorchange27.46% beforeRMS/27.41%after;normchangesonly.78%median.
Normalizedcodevariance96.7%ofsecondmoment,notnearlyconstantinputs. ThisdoesNOT
supportassumingRMSnormalizationerasesallfactdifferences;notaninformationproof.
Report docs/token_memory_64_retrieval_research.md hasfullprespecifiedprotocol.

# Active goal: raw64 answer-content weighting pilot VERIFIED, no retrieval win (2026-10-16)

Session24714 trainer finished0 at1000/48762 (boundedpilot intentionallypaused);
evaluator82803 finished0; independentverifier76529 finished0. No worker from this
experiment remains. ObjectiveACTIVE/UNMET; don't rerun alreadycompletedpilot.
No existing src/model/source/checkpoint/corpus/cache changed;124protected hashes
independently rechecked identical. This turn is real research progress, notwait.

Matched ordinary-position1000 vsweighted1000 full-dev:
CE3.507397358->3.524092590;acc35.224762%->34.956340%(-.268422pp), paireddevdoc95%CI
[-.31635,-.21891]pp (fixedweights/one seed,notfinaltraininguncertainty).
Critical teacherforcedCE1.037994->.902227;accuracy1221/1966->1226/1966 only.
FIRSTcritical29/285 BOTH, despitecorrectteacherprefix. Do notcall62%pieceaccuracy
factualrecovery. All348actualdev memoryfreeanswers+64changedvalue+64changedquery+
64blocked perarm:positive0/285 both;missing9/63 ordinary->0/63weighted;changed0/64
andpaired0/64both;boundfullvalue0 evenrelaxedname/valuechecks. Historical0/103
and0/32both. Weighted repeats Amina/purplecabinet/kitchen195/285;oldmode109/285.
Changedfact outputs32/64oldvs23/64new;querychanges17/64vs11/64, no correctpairs.
Blocked64 both0EOS atcap64; memorydependencealoneisnotreading.

OriginalfrozenD diagnostic reconstructs ALL29346factbodytokens across348original+
64changedprompts,412/412exactbodies. No decoderfeatures usedbyreader. Sampledcode
hasfacts;doesnotproveuniversallylosslesscodec. Allbatch/labels/LR/cursor andAdam
1000 independentlyverified;8531criticalexposures/7416257assistantlabels.
Results under run/evaluation-step1000/report.json +independent-verification.json,
output-behavior.json,alloutputs/cases/sourcearchive. Maskindependentverification
inanswer-auditdirectory. Report docs/token_memory_64_retrieval_research.md updated.

Next: test explicit question-to-fact binding with counterfactual paired values,
missing-information supervision and languagepreservation atsameinference budget.
Do not adopt positive-onlyweight64 orcontinueit merelybecause criticalCEfell.
No broad/perfectretrieval or fulltwoepochqualitywin. Preserve original23M/onecode
per64token attention/KV advantage;anylaterqualityclaimneedsbroadheldoutchecks.

## Answer-content weighting setup and training record

Goal remains ACTIVE and unmet. Previous cleanup recheck was no retrieval progress;
this turn implemented and trained a bounded loss-only comparison. Growing/vector
families stay retired. No changes to src model implementations or original runs.

Exact corpus mask audit:214,412 critical location-word BPE labels/180,678,552
train assistant labels(.1186704%);31,411 answerable/7,859 missing memory dialogues.
Dev1,966 critical labels/285 answerable dialogues. Independent sentence-regex
derivation matches ALL train/dev selected positions without using gold record
metadata or the training helper. Audit dump/token-memory-64-answer-audit-20261016.

New tools/train_token_memory_64_answer_weight.py uses SAME token_end PositionProbe
as position pilot,23,209,216params,oneposition/code,zero extra KV. Training-only
weights64 on positive answer color/container/room BPE pieces,1 on other assistant
labels. Missing answers ordinaryweight1. Global sum(w*CE)/sum(w) before clipping;
logged CE remains unweighted. No input gold features, no extra data or labels.
Four CPU tests passed incl actual two-epoch tiny serialized Adam/cursor resume,
microbatch/allbatch allparameter gradients, mask validation and unit-weight parity.
Zero-update fullsize GPUpreflight: exactfresh47,4.266GiBreserved heavy+Adamscratch;
firsteffective32 forward/backward median.322882 ordinary/.330329 weighted(+2.31%),
not end-to-end training timing. Inference identical; no new reader/attention path.

Fresh bounded run dump/token-memory-64-answer-weight-pilot-20261016 completed
1,000updates of ORIGINAL48,762schedule,2epoch recipe/effective32/micro16/FP32/seed47.
Trainer session24714 exited0, intentionally paused1000.7,416,257original labels,
8,531 critical labels. Actual savedstep0 freshly verified against matchedCPU
singlethread/seed47 and emptyAdam; first verifier omitted matched runtime and
was corrected before comparison. No training restart. Sources/cache/data guards
remain strict; unique format token-memory-64-answer-weight-v1, answer audit SHA
inrecipe. Never resume through plain64 or position-only entrypoints.

Read-only evaluator tools/evaluate_token_memory_64_answer_weight.py completed
session82803 successfully. Scored full dev,
critical/firstcritical labels, all348dev memory freeanswers+64each changedvalue,
changedquery,blocked controls per arm,312historicalcases, codec-only fact recovery.
Same-step position1000checkpoint control retained. Results are recorded above.
Independent verifier tools/verify_token_memory_64_answer_weight.py --results
completed successfully. Report docs/token_memory_64_retrieval_research.md.

# Active goal: raw64 factual reading; position diagnostic and 1,000-step pilot VERIFIED (2026-10-16)

Current user goal is improving raw64 + compressed-memory fact retrieval and
exceeding the ~61% language reference while retaining64:1 attention-position
compression,23,209,216 trainable parameters and efficiency. Goal remains ACTIVE,
NOT achieved. Latest goal turn made progress through verified measurements and
one bounded fresh training experiment. Growing/vector families stay retired.

Reference correction:61% is original raw512 PLUS older compressed memory, not a
memory-free conventional model. A matched raw-only comparison still needs evidence.
No broad/perfect retrieval or full two-epoch architecture improvement is proven.

New tools/token_memory_64_positions.py is an explicit process-local PositionProbe;
kept src model files unchanged. token_end positions: memory64*j+63, queries
64*visible+within. Same tensors/masks/count/one slot per code; no extra KV entries.
No full unpacking/decoder/raw-history side channel. Six CPU tests pass incl exact
fresh identity, all-parameter gradients, causality/padding/blocking and actual
serialized tiny two-epoch optimizer/RNG/cursor resume. tools training adapter
reuses original trainer via scoped runtime overrides; unique checkpoint FORMAT
and recipe.position_mode guard resumes. Never resume this with original training.py.

Full7,831dev/1,859,388label inference ablation COMPLETE:
legacy plain64 CE2.214283787/acc53.504594%; all memory blocked but positions/counts
preserved CE2.720477551/47.567264%; token_end on same trained weights
CE2.457805972/49.047106%; original512 CE1.804710360/61.013194%.
Memory contributes+5.9373pp, so it is used, but useful factual reading is absent.
Block-first-token acc plain64=22.0887%, original512=56.5431%; this contributes
only0.5350pp of7.5086pp overall gap. Boundary-only repair cannot close whole gap.
Matched cached hidden-forward medians29.2981/29.4102ms(+0.38%),10 alternating
rounds; same attention shape [16,8,369,369]. Excludes head/encoder/optimizer/disk;
not end-to-end speed proof. 312 historical generation cases;624 original512/plain64
outputs replay exactly. Switched positions still0/103 notebook,0/32 changed facts.
Independent verifier rederived metrics/phase masks/scores and23protected hashes;
5k document bootstrap memory effect[5.8198,6.0595]pp,position effect[-4.6362,-4.2832].
Artifacts dump/token-memory-64-position-diagnosis-20261016. Initial generation
orchestrator return-tuple bug repaired and generation-only rerun, old evaluator
archive retained. No optimizer was used for diagnosis.

Fresh seed47 matched position pilot COMPLETED its bounded1,000 updates and is
intentionally stopped, NOT two epochs done. Run
 dump/token-memory-64-token-positions-pilot-20261016
Adapter tools/train_token_memory_64_position_pilot.py. Original48,762-step/two-epoch
schedule retained, FP32/effective32/micro16, originalcache read-only. All1,000
batches/32,000examples/7,416,257labels/LR/cursor match old baseline; all Adam1000.
Initial full state exactly same via fullsize identity test; actual run is fresh,
not fine-tune. New full-dev CE3.507397358/acc35.224762% vsold step1000
CE3.511000690/35.105476%: +0.119287pp,12/17sources better CE. Small early signal,
not final-quality or fact win. New0/103 facts,0/32changed,0/8oldcolors. No old1000
weights retained for matched-step generation comparison. Evaluation/verification
under run/evaluation-step1000. Trainer pauses before periodic eval at stop limit,
so separate read-only evaluation used. Source-archive separator fix in verifier;
no repeated training. Keep format/source/cache guards intact for future resume.

Additional read-only control original trained512 weights under unchanged64 mask:
CE2.383691481/acc50.905190%, blockfirst5.8642%,0/103facts0/32changes.
Artifact dump/token-memory-64-original-weights-control-20261016. No optimizer,
protected weights/cache/sources unchanged. Thus simple checkpoint swap is no fix;
does not rule out a trained curriculum. All workers from these probes finished.

Full report and goal requirements: docs/token_memory_64_retrieval_research.md.
Next concrete research action: independently audit fact-answer-content masks in
existing corpus, then a matched training-only critical-answer loss weighting
probe with language preservation. Same inputs/codes/parameter budget/attention/KV;
labels may set loss weights only, never model inputs. No result yet for that
proposal. Need broaden held-out factual tasks and fresh two-epoch comparisons to
meet original goal; do not mark complete for a toy retrieval win or lower loss.
No subagents, no original model/source edits, no baseline/retired-run restarts.

# Growing/vector experiments RETIRED; token_memory_64 is the focus (2026-10-16)

Latest user explicitly requested removal of all growing/vector variants, code,
models and unnecessary experiments while preserving a Markdown experiment record.
This supersedes historical no-deletion / growing-training instructions below.
No growing/vector run should be resumed or recreated without a new request.

Cleanup is complete: 316 local targets removed, plus
I:\growing-token-cache-v4. Removed 203.036 GiB gross, about203 GiB net
after retaining about87 MiB of text evidence/audits. Includes growing token, query
reader, four-readout, older latent/vector/flow/planner/constructor/grounded-codec
experiments and48 synthetic fixture directories with growing source/checkpoints.
No active corpus trainer existed at cleanup; no unrelated process was stopped.
Four-readout log reached2140, stale status2125, durable last.pt2000. Other last
checkpoints before removal: original growing35472, query reader6834.

KEEP: position_compressor and its frozen original codec/tokenizer; branch_sigmoid;
token_memory original512; token_memory_64 plain/current focus; completed64crossattn;
raw_transformer reference; all original datasets; dump/token-memory-64-cache-v4.
Preserved original512/plain64/cross64 completed checkpoint runs and stoppedattnres64
checkpoint/results. Attnres64 source was already missing; orphan test/evaluator
removed. Kept model Python, parameters, position semantics and checkpoints unchanged.
124 protected files SHA-256 verified identical.23 CPU checks passed,2 historical
fixture checks skipped; codec temp-directory ACL failure passed on unsandboxed
retry of the unchanged test. Only synthetic CPU optimizer checks; no corpus/GPU
training. No subagents.

Record: docs/retired_vector_experiments.md. Exact manifests, old README/AGENTS prose,
checkpoint identities, JSON metrics/recipes and text-only evidence.zip under
docs/experiment_records/vector_retirement_20261016. Existing research Markdown is
historical; its references to removed code/binaries are NOT runnable instructions.

Position inspection only, no architecture change: current plain64 RoPE uses code
index j for memory and visible_code_count+within for raw queries. A proposed
controlled experiment uses original-token coordinates (block end64*j+63, query
64*visible+within), preserving attention/KV dimensions. This supplies outer block
distance, not64 internal content positions, and has no quality/speed result yet.
Keep saved checkpoint semantics reproducible; do not silently switch positioning.

# Historical notes follow — superseded where inconsistent with the retirement above

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
