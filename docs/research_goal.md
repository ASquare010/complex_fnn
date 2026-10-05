# Atlas: compact, reusable computation

Main research goal and long-term vision — 2026-10-03.

This is the canonical statement of the Atlas ambition and system idea.
The active FFN research contract, experiments, evidence and success targets live
in [Step 1: a faster, more capable FFN replacement](step_1_ffn.md).
The [README](../README.md) covers setup and code layout.

## 1. The ambition

Build a language and reasoning system whose capability comes from efficient
representations, reusable computation and explicit memory, rather than relying
primarily on a large collection of dense neural weights.

The long-term ambition is **a 1 million parameter system that can rival a roughly
10 billion parameter language model**. This is an open research ambition, not a
prediction, an achieved result, or an extrapolation justified by our experiments.
The approximately 10,000-fold parameter gap is much larger than our current gains.

Two claims must remain separate:

* **Strict target:** at most 1M total learned parameters, counting the encoder,
  embeddings, reasoning machinery and generator. External storage, retrieval,
  temporary state and compute are disclosed separately and equal-access memory
  baselines are included.
* **Intermediate system target:** a 1M reasoning core attached to a larger encoder,
  generator or memory system. Useful if achieved, but never described as a 1M-total
  LLM. Report the complete system size and resource costs.

Before any final comparison, freeze the exact 8–12B dense reference checkpoint,
its revision, tokenizer, prompting, generation budget, tools and memory access.
Evaluate language modeling, factual use, multi-hop reasoning, mathematics, code,
long-context retrieval and instruction following separately. A win on a toy task,
or an average hiding major category failures, does not establish this ambition.
Define task-specific equivalence margins before those tests, use confidence
intervals, disclose contamination checks, and include both equal-compute and
equal-latency comparisons. The final benchmark contract is a future milestone.

## 2. The system idea

The central hypothesis is that a small learned system might become more capable
by organizing and reusing its computations more effectively. This does not assume
that dense models only memorize or cannot compose information. We need a specific
operation or arrangement that makes better use of a limited parameter and runtime
budget, together with evidence of where it works and where it fails.

In the long-term design, learned parameters would encode reusable
transformations; temporary states would hold the current problem; persistent
memory would retain information for retrieval. These roles may overlap in
practice. Moving knowledge into storage does not remove its cost: storage,
retrieval latency, encoder size and generation cost belong in the comparison.

The FFN study tests a concrete part of this idea. Given a contextual token
representation, can we transform its features more effectively without paying
for a large collection of independent dense connections? A successful primitive
could later support a compact reasoner. Memory and semantic compression remain
separate hypotheses that require their own evidence.

Atlas separates jobs that a conventional decoder often performs together:

    text → encoder → compact semantic states → reasoning core → language output
                              ↕                     ↕
                      persistent memory ← retrieval

The encoder would emit a variable number of latent vectors according to useful
information content. The reasoning core would receive current latents, retrieved
latents and a short raw-token window. Memory would store reusable knowledge while
the core learns to combine it. The CPU analogy means **reusable operations plus
working state and storage**, not a claim that a neural block already executes
symbolic instructions or understands causes.

Possible later components include an integrate-and-fire semantic boundary
mechanism, reconstruction-assisted encoder training, learned memory admission,
deduplication, contradiction handling and provenance. Reconstruction alone does
not establish that latents preserve relationships needed for reasoning. Exact
names, numbers, negation and order need explicit tests. Extreme compression such
as one million tokens into 80 vectors remains an unsupported aspiration.

These system components are **not implemented in the new experiment framework**.
The immediate project concerns the computational primitive inside the reasoner.

## 3. Research stages

The project advances from a useful computational primitive toward a complete
compact language and reasoning system. The long-term goal remains the same
regardless of which candidate succeeds in an individual experiment.

1. **Computational primitive:** discover a faster FFN replacement that learns
   useful feature interactions with fewer learned weights. The separate
   [Step 1 research plan](step_1_ffn.md) defines the active scope and evidence.
2. **Compact reasoning system:** establish whether a confirmed primitive can
   support a meaningful whole-model parameter advantage under fair resource
   comparisons.
3. **Semantic representations and memory:** investigate compressed inputs,
   reusable working states and persistent knowledge, with explicit tests of
   information preservation and complete resource accounting.
4. **Full-system evaluation:** assess the strict 1M-total ambition against a
   fixed large-model reference using the benchmark contract described above.

These stages express the research direction, not achieved milestones or a
promise that each component will work. A component result must justify its own
claim before it is used to motivate integration.

## 4. Research preservation and implementation

See the root README for the model packages, typed settings, dataset loaders,
trainer API and runnable checks. Raw outputs stay in ignored `dump/`; compact
records intended for review go in `records/`.

The reset preserves 34,877 old source/evidence files in
`dump/legacy/research-before-reset.zip`, verified by ZIP integrity checks and
SHA-256. Archive identity and cleanup totals are in `records/reset.json`.
Selected reference states remain in `dump/legacy/checkpoints/`, and historical
datasets in `dump/data/legacy/`. Retired weight hashes describe deleted artifacts.
Back up local artifacts separately; Git does not preserve ignored `dump/`.
Old Git history has not been rewritten. The new framework is not a bit-identical
replay of the archived training recipes.

## 5. Status and supporting evidence

The full Atlas system described here is a research proposal. The new framework
supports component experiments; no new architecture has passed the scientific
gates recorded in the Step 1 plan. The 1M-versus-10B ambition remains unachieved.

Historical FFN findings and their limitations, candidate equations, datasets,
comparison rules and retained references are maintained in
[Step 1](step_1_ffn.md). A successful FFN experiment is evidence for that component,
not proof of the full-system ambition.



