# Positive single-fact packing pilot (2026-10-17)

The user authorizes a new separate 512-update-per-arm comparison, preserving
64 old tokens → one 256-dimensional code → one attention position, with language
replay. Neither the native256 run nor the completed packing pilot is extended,
restarted or modified. Both arms reuse identical copies of the original partial
native step-8,750 initialization, with fresh Adam and the original packing weights.

This pilot is entirely a positive single-fact warmup. Establishing reading comes
before introducing multiple records or missing answers into the reading loss.
There are no `unknown` reading targets during these 512 updates. Ordinary language
replay remains unchanged and can contain the original corpus's missing answers.
The same question instruction about missing facts stays in the raw input.

Prepare 1,024 fresh four-example units (4,096 unique reading examples): two
different values in otherwise matching old histories, asked with two paraphrases.
Train two shuffled passes, 16 original language segments plus 16 reading examples
per update, microbatch 8. Language segments/order are exactly the previous pilot's
8,192 examples; the reading data and ordering are new and identical between arms.
Names and two-word access phrases vary; positive training prevents the special
`unknown` target from dominating first-answer decisions. It cannot prevent a
different context-free word prior; blocked and swapped-memory tests must check.

Prepare fresh single-record groups for identity, combinations, names, wording,
new answer words, composed transfer and mixed missing cases, plus actual training
fit. Retain the completed pilot's two-record held-out suites as regression tests;
they have already been inspected, so are not a fresh blind benchmark. Entire
fact-bearing blocks of fresh data are disjoint from all old cases and each other
across splits. All facts remain old; whole question and answer/EOS fit raw64.
No decoder-success filtering. New case labels are independently parsed from the
actual source text and serialized IDs.

The frozen arm updates only the 23,209,216-parameter native256 reader. The packing
arm also updates the existing 2,080,000 packing weights. Encoder prefix and
reconstruction decoder remain frozen. No new production operations, parameters
or KV entries. Loss remains .75 LM CE + .25 answer CE + .05 reconstruction CE;
four old fact blocks per update are reconstructed only for training. Reader LR
3e-5 cosine to3e-6, 30 warmup, packing LR .1 reader, FP32 AdamW .01, joint clip1.

Reuse the previous FP32 frozen-prefix feature cache read-only for the exact same
language examples, full original dev and retained regression cases. Encode only
fresh reading blocks into a separate cache. Recompute all final codes through
each arm's current packing weights, including language replay. All old assets
and native files are guarded by hashes. Unique checkpoint format prevents old
trainer resumes.

Before training: CPU objective/microbatch/resume/frozen isolation and positivity
checks, independent all-case data audit, zero-update full-size GPU checks, exact
initialization and actual first-update audit. Train exactly512 per arm; no adaptive
extension or test-based selection. Afterward evaluate complete free answers+EOS,
positive training fit, both memory swaps, blocked/shuffled codes, missing-case
hallucinations, two-record quartets, full original dev, historical312, original
decoder fact reconstruction/code drift. Independently rederive outputs/metrics
and check all protected hashes. No quality claim from lower teacher loss or from
merely replacing the `unknown` mode with a common value. Partial-parent, one-seed
and synthetic-task limits apply. Do not adopt or extend without new authorization.
