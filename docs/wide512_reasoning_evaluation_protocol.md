# Reasoning evaluation coverage and interpretation

Supplement to the frozen full-budget evaluator, with no changes to its source,
models or training. The original development set has406 source-labeled reasoning
documents across reasoning, reasoning_chat, reasoning_tables, reasoning_multiturn
and reasoning_tools. These mix mathematical problems, general advice, table
summaries and instruction following. Teacher-forced accuracy on these sources is
language/reference prediction, not verified reasoning correctness.

The existing first64-source-record/last-number filter yields53 cases. A list index,
year or stadium capacity is not necessarily a final mathematical answer. Preserve
that metric as a small reference-number-agreement diagnostic only. Do not upgrade
it to mathematical accuracy or broad reasoning parity in any report.

The separate CPU preparation exports ALL406 original last-assistant prompts and
references. Independently serialize every message/span and verify the complete
token stream against the immutable original token file. Prompt tokens stop before
the current assistant body. Export only model-independent original references;
no student outputs, decoder-success filtering, training examples or reserved
novel-literal TEST access. Hash code/tests/protocol, corpus inputs, cases, audit and
the original training manifest. Existing182/185 source closures remain unchanged.

Generation caps are at least original reference length+128, rounded up to256
tokens, with minimum256. This removes the original fixed256-reference-length
confound while retaining explicit cap-hit reporting. No early EOS override.
Future free evaluation is gated on BOTH48,762/361,357,104-label original exposure
audits and a free single-GPU-worker lock. The preparation alone is not evaluation.

Keep explicit final-answer scalar reference agreement separate from incidental
numbers. Parse final_answer tool calls, simple boxed scalars or wholly scalar final
text after a closed think block; reject incomplete thoughts, list/table numbers,
non-final tools, ambiguous units and unsupported symbolic expressions. Normalize
unambiguous rational/decimal/scientific numbers exactly. These teacher references
are not independently certified mathematical truth. Nonnumeric/paraphrasable
responses cannot be graded semantically by string equality. Any exact-final-text
or format metric must be labeled accordingly, with source and denominator counts.

Remaining work: source-bound independent all-case input audit, full-budget free
evaluation/replay and a separately verified arithmetic/binding diagnostic with
actual known answers and fact-reading controls. Do not substitute a narrow
conditional arithmetic test for the overall requested reasoning comparison, or
claim success because both readers fail. The full goal remains ACTIVE/UNMET.
