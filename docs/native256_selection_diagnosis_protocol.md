# Native256 selective-reading diagnosis (2026-10-17)

This is a new read-only diagnosis, not another training run or an extension of
any completed pilot. The previous goal turn made progress: balanced fixed-layout
training established closed-palette reading but not open-vocabulary copying or
entity selection. The full23M/two-original-epoch/64:1 quality objective is unmet.
Native256, all completed checkpoints, sources, corpora and caches stay unchanged.
No subagents. No decoder features or reconstructed history enter reader input.

Use the exact frozen/packing512 checkpoints from the completed balanced pilot.
Prespecify64 aligned units per condition,256 cases each. Eight known one-token
values remain balanced (each twice per16-case batch). All histories contain47
body tokens, all facts and distractors lie in the first completed64-token block,
query body starts77, correct value+EOS fits raw64, and the reader sees ONE code.
All within-unit memory swaps preserve raw query/length/position. No filtering by
model or reconstruction success. Fresh blocks are disjoint from prior cases.

Conditions: clean fixed(value24), clean shifted8(value32), clean shifted16(value40),
fixed fact plus a spare-word distractor after it, spare word before a fact shifted8,
two access-phrase records A/B and B/A (ask BOTH names), and clean fixed with eight
held answer words. Clean shifts preserve the complete token word bag. Distractor
and two-record swaps preserve both value words, changing their assignments rather
than their presence. Two-record orders also preserve the word bag and raw query.
All indices are zero-based; query position and total source length are matched.

For each trained reader, obtain full-vocabulary greedy value+EOS outputs with
correct, blocked and shuffled memory for ALL cases. Preserve code count/positions
when blocked and assert identical outputs for memory-swapped raw-identical pairs.
Save prefix-only first-decision logits (NO answer tokens), verify their argmax
against generation, and rank a condition-wide eight-word candidate set. Candidate
choice accuracy is NOT free retrieval; report full-vocabulary and choice results
separately. In particular, held-word choice vs blocked can distinguish output
palette bias from a lack of useful novel-word ranking. Chance is12.5% for choices.

Report complete answers, both changed values correct, all-four units, question/
memory sensitivity, first/second-record or distractor replay, and full/choice CE.
For two-record groups, also compute first-decision query×memory logit interaction
using fixed value-A minus value-B logits; memory-only and query-only priors cancel.
Do not infer success from attention mass, teacher EOS accuracy or output changes.

The ORIGINAL frozen reconstruction decoder separately checks every source fact
and distractor value from each arm's codes. Save full predictions and code drift;
this is evidence about encoded content, not a side channel into reader attention.
No optimizer is constructed; assert reader/packing tensors unchanged and guard
all protected old assets and both512 checkpoints before/after. Independent audit
must derive answers from actual source text, reserialize tokens, check balanced
labels/positions/swaps/word bags, rederive greedy/choice/interaction/reconstruction
metrics from saved IDs/logits, and check guards. CPU generator/causality/scorer
tests precede GPU evaluation. Do not rerun full language validation: weights and
production forward are unchanged and its verified result already exists.

This diagnosis cannot prove full-budget parity. It selects the next training or
reader hypothesis; a final improvement still needs fresh matched two-epoch runs,
identical corpus exposures/objectives/metrics,23,209,216 reader parameters, one
attention/KV slot per64tokens, broad unseen-value/binding/reasoning tests and
full language preservation. Existing61.013% reference is raw512 PLUS older codes,
not a memory-free1:1 model; a true all-raw reference remains a separate requirement.
