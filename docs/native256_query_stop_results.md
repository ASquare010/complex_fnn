# Balanced STOP pilot: verified regression

18 October2026. Both fresh-seed47 arms completed the prescribed512 updates and
all independent training, aggregate, public-forward, notebook and transfer
checks. The paired comparison also PASSED, including2,748 guarded inputs.
Do not adopt the supervised arm or automatically extend either checkpoint.

Both arms use the SAME STOP architecture and initial tensors/RNG, exact
23,209,216 reader parameters, separate frozen encoder, raw64 and one global
memory position per64 old tokens, with no extra KV entries. The comparison
isolates balanced STOP auxiliary coefficient0 versus.02; it is not an architecture
comparison against the earlier V2 step1000 model. Original query/source losses
remain. Each arm saw3,771,495 original LM labels,14,519EOS and3,756,976 other,
800 query segments/1,600 query labels and643 source segments/1,286 critical
terms. All Adam512, original orders, LR, cursors and global denominators match.

| Development measure | Ordinary | Balanced STOP |
|---|---:|---:|
| Full language accuracy |38.922592%|38.877093%|
| Language CE |3.534537|3.569743|
| Positive full fact answers |43/285|3/285|
| Missing-fact answers |0/63|0/63|
| Changed-value answers |6/64|1/64|
| Changed-query answers |16/64|0/64|
| Both original/changed value correct |4/64|0/64|
| Both original/changed query correct |1/64|0/64|
| Notebook four-way full replies |27/256|0/256|
| Notebook full all-four sets |1/64|0/64|
| Proper notebook completions |255/256|242/256|
| Completion all-four sets |63/64|53/64|
| Prefix-assisted access complete |1/224|0/224|
| Numeric reasoning probe |0/53|0/53|

Both arms score0/512 in each of seven unconstrained access groups,0 under blocked
notebook controls, and0/64 natural blocked positives. Historical present notebooks
are2/103 versus0/103; changed historical0/32 both. Correct value prefixes in the
access transfer subset do not establish complete answer correctness.

Supervised-minus-ordinary language difference is−.045499 percentage points;
paired document95% interval[−.080989,−.010329],4,852 clusters/5,000 draws. These
are fixed-checkpoint document intervals from one training seed, not estimates
of training-seed uncertainty. The positive-answer difference is−40/285.
Lower STOP training loss did not produce a factual quality gain. This result
cannot identify a universal model-capacity limit or isolate the entire STOP
architecture as the cause; it rejects this extra loss at this matched stage.

Evidence: dump/native256-query-stop-pilot-20261018/comparison-512/report.json,
both training-verification-512 receipts and independent evaluation receipts in
each arm's evaluation-512 directory. The512 steps are a bounded diagnostic,
NOT two completed original epochs or parity with the mature reference.
All existing/native runs, source files, caches and reserved TEST remain unchanged.

A separate zero-update candidate retains original outputs when EOS is available
for copying, using generator-controlled stopping otherwise. Its selected CPU
subset improves prefix-assisted access0/28 to15/28 while preserving all8 notebook
outputs; it is not adopted. A wider DEVELOPMENT-only GPU diagnostic is running
separately on the preserved V2 step1000 checkpoint. It includes full-dev teacher
scoring, all natural factual controls, fixed transfer and notebook controls. No
new training, full-budget quality or speed claim follows from this diagnostic.
See native256_generator_stop_availability_cpu_diagnostic.md.
