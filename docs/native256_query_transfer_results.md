# Verified transfer failures and local STOP prototype

18 October2026. This turn made experimental and implementation progress; the
full64:1/23M/two-original-epoch quality goal remains ACTIVE / UNMET. All workers
are terminal. The completed query-reader pilot remains at1,000 updates; native256,
original sources/runs/checkpoints/data/caches and reserved student TEST are intact.

## Question selection is not the whole failure

Read-only diagnosis uses all3,584 previously scored access-phrase development
cases, not the reserved final TEST or novel-literal set. The learned selector
chooses the correct entity/topic and signals the correct old record on512/512
cases in EACH of familiar, new values, new names and shifted positions. Missing
cases choose the right entity/topic512/512, with signal only on the256 present
records. All question scopes are valid. Wording changes select the entity and
correct record462/512, but the topic0/512; composed changes select the entity/
correct record256/512 and topic0/512. Correct question selection alone cannot
explain or fix the familiar group's0/512 original exact answers.

## Some values are readable, but replies do not finish correctly

The first eight sorted units/group give224 cases, both questions/both swaps,
without filtering for success. Extending original generation to32 tokens still
gives0/224 exact replies and0 EOS. Many replies start a full sentence or code
instead of the requested value-only answer.

An audited prefix, `The access phrase for NAME is`, supplies only the public
question's name and relation. It contains no value. With16 generation tokens,
the first SINGLE-TOKEN value is correct on176/208 positives; none of16 missing
cases produces the correct unknown value. Correct first values by group:

| Group | Correct value prefix | Complete exact reply |
|---|---:|---:|
| Familiar |26/32|0/32|
| New values |30/32|0/32|
| New names |20/32|0/32|
| New wording |28/32|0/32|
| Shifted positions |32/32|0/32|
| Missing group |13/16 positive;0/16 missing|0/32|
| Composed changes |27/32|3/32|

Prefix-only first-value all-four sets are3/8,6/8,1/8,5/8,8/8,0/8,3/8 respectively.
These partial prefixes are diagnostic, **not successful factual answers**. The
strict result remains3/224, with0/224 original-format replies. Blocking the first
old fact code gives0/56 correct value prefixes/complete answers. Swapping only
the fact code exactly replays the corresponding donor's enabled output.

All prefixes, source subsets/edits, token/text/EOS/strict scores and sampled
uncached public CPU outputs were verified. The final verifier checked all2,589
prior protected/input/source/output guards. The independent decision aggregation
also rederived every original literal value coordinate/head summary and confirmed
literal-only features equal actual packed-code extraction on one fixed case/group.

## The copy mixture overrides an available stop decision

This is a TEACHER-FORCED endpoint diagnosis: correct content is supplied before
asking for EOS. It is not free generation and is never used as an inference oracle.

| Endpoint | Generator chooses EOS | Final mixture chooses EOS | Correct generator overridden |
|---|---:|---:|---:|
| Original access, gold value supplied |0/224|0/224|0/224|
| Prefixed access, gold value+period supplied |212/224|10/224|202/224|
| Proper notebook completion, gold content supplied |256/256|256/256|0/256|

Mean access-prefix generator EOS probability is.310241, final EOS.147583, copy
mixture weight.486426, pointer EOS mass0, NULL.00002155. For notebook endpoints,
generator EOS is.295695 but pointer EOS averages.991615, copy weight.996299 and
final EOS.990701. The familiar notebook pathway has learned to copy EOS from old
memory; transfer memories without that usable EOS do not provide the same exit.

Every stored probability satisfies the independent mixture identity:
`Pfinal(EOS)=Pgen(EOS)*(1-Pcopy+Pcopy*Pnull)+Pcopy*Ppointer(EOS)`.
CPU public-forward endpoint samples reproduce GPU probabilities/argmax. This
establishes a measured stopping bottleneck in this checkpoint, alongside initial
format selection and residual binding errors. It does not establish a universal
cause for the old semantic encoder's quality gap.

## New separate prototype, no production training yet

`src/models/token_memory_64_query_stop` adds a final512-parameter learned local
STOP head, funded by one FFN hidden unit (31368 to31367), keeping exactly
23,209,216 reader parameters. Encoder remains separate. Two local readers,
raw64/global64:1 and the cache class/arrays remain unchanged. STOP is an output
action, not a global attention position or extra KV entry.

The head redistributes part of nonNULL lexical-copy mass to EOS, so the copy path
can finish without finding an old EOS token. NULL mass retains its existing
generator fallback; with empty/blocked memory this new head contributes zero.
Prospective auxiliary supervision uses only original assistant EOS/nonEOS labels,
globally balanced class means and detached final hidden states. No answer labels
enter inference, and no new fact text is introduced by this objective.

Six CPU checks pass: exact budget/own fresh identity; independent normalized
distribution and active gradients/NULL fallback; active cached/public/chunked
rollover with same KV bytes/attributes; causal future/padding/all gradients;
detached STOP objective/global microbatch normalization; actual mixed-objective
microbatch and exact serialized eight-step Adam resume. These are correctness
checks only. The new model is UNTRAINED and has no quality/adoption claim.

Next required work: full21-layer GPU gradients, capacity and matched timing, then
a new bounded paired training comparison if gates pass. Preserve this completed
pilot. Initial value-only formatting, missing answers, broad reasoning and final
fresh two-original-epoch parity remain unresolved.

Artifacts: `dump/native256-query-transfer-diagnosis-20261018/` and its
`decision-analysis/`, plus `dump/native256-query-stop-cpu-preflight-20261018/`.
Protocols: `native256_query_transfer_protocol.md`,
`native256_query_transfer_decision_protocol.md`, `native256_query_stop_protocol.md`.
