# Step 1: refine the strongest compact FFNs

## Current goal — nearby architecture exploration

Updated at the user's direction on 2026-10-04. **We have achieved the practical
milestone we were looking for: substantially smaller FFNs, lower training VRAM,
and language performance close to the full model in our development screen.**
For signed-linear context, that means 70.54% fewer FFN weights, about 26% less
training VRAM versus full native SwiGLU, approximately unchanged training speed,
and validation NLL costs of 2.58% on TinyStories / 0.94% on WikiText.
This is a useful measured tradeoff, not proof of equal/better quality on both
corpora or repeated-seed confirmation. Historical results and failures stay intact.

This completed refinement studied three measured starting points: **signed-linear
context, self-gated curves, and fixed basis readout**. They offer different
quality/size tradeoffs; they are a deliberately chosen shortlist, not three
universally dominating models. Ordinary compact SwiGLU remains a serious control.

Explore a small nearby space, compare variations directly with their parents,
then select and confirm the strongest overall recipe. Do not pursue C++, CUDA,
custom-kernel or buffer-lifetime optimization. Existing implementations may be
reused unchanged. Pending paired-readout v2 and other execution revisions are
superseded as active work by this direction; their records remain available.

The [registered refinement study](compact_refinement_hypothesis.md) defines the
first three variations, fixed comparison budget and final-selection procedure.
Run language comparisons after correctness checks; resource measurements inform
selection rather than diverting this phase into kernel engineering. Preserve at
least 70% FFN compression in the proposed variations. Track quality, total/FFN
weights, VRAM and training time, and update the dataset leaderboards after every
completed evaluation/export. Report a tradeoff honestly if no variation improves
both corpora. The original stricter thresholds below are historical targets and
remain useful stretch goals, not prerequisites for recognizing the milestone.

## Refinement progress

The first comparison is complete: all 12 language runs and frozen feature-removal
checks passed the evidence audit. Under the preregistered two-corpus ranking,
**Curve-Wide is first and Signed-Fine second**. Both improve on their parents on
both datasets in seed101. Smaller basis groups improve TinyStories but worsen
WikiText and do not advance. These are development results, not a final winner.

The [round results](compact_refinement_result.md) retain all six recipes.
[Confirmation](compact_refinement_confirmation.md) freezes the two finalists,
their parents, parameter-matched dense controls and full baselines for seeds
211/307/401. Resource measurements and the independent longer run follow before
the final report. No kernel or model changes are permitted during confirmation.

TinyStories confirmation has completed all 30 runs (10 recipes across seeds
211/307/401). Curve-Wide beats its parent and both matched compact dense controls
in all three seeds. Mean NLL is 2.592300 versus parent 2.605145, full GELU 2.589529
and full SwiGLU 2.542837. It is the strongest compact recipe in this confirmation
table, while full SwiGLU remains better on quality. Signed-Fine trails its parent
in all three TinyStories seeds. All 60 paired-seed language runs are now complete
and audited. Curve-Wide also improves over its parent in all three WikiText seeds:
mean NLL 4.272862 versus 4.282949. It ranks first among the compact recipes under
the registered combined score. The signed parent has slightly lower WikiText
mean NLL (4.271632), so this is not dominance on every dataset. Signed-Fine's means
are slightly worse than its parent's on both datasets. All 60 resource profiles
are complete. Curve-Wide passes the registered parent-relative limits on the
observed point estimates, with about 28% less allocated memory than full SwiGLU.
Full-baseline timing variability prevents a reliable speedup claim.

**Curve-Wide is the final overall research choice.** The independent seed509,
8000-update comparison and all eight held-out tests are complete and audited.
It beats its parent and compact SwiGLU h384 on both validation and test splits.
Against full SwiGLU, held-out NLL is 2.30% worse on TinyStories and 1.38% better
on WikiText. The shorter three-seed comparison favored full SwiGLU on both,
so this remains a measured quality/size tradeoff, not universal superiority.
Held-out results did not select or tune the recipe. The bounded refinement goal
is complete: three nearby variations were tested, unsuccessful variants retained,
and one winner selected without new kernel optimization. The practical milestone
is 74.9% fewer FFN weights, 42.9% fewer total weights and about 28% less allocated
training memory. Reliable speedup and larger-LLM generalization remain unproven.
[Final summary and evidence](compact_refinement_summary.md).
[Frozen selection](compact_refinement_final_selection.md) /
[resource measurements and limitations](compact_refinement_resource_analysis.md).
[Paired-seed analysis](compact_refinement_confirmation_analysis.md).
[Every confirmation run](compact_refinement_confirmation_result.md).

## Historical search and acceptance criteria

The previous goal's acceptance contract is in
[language screen v1](language_screen_v1.md). Its explicit user-defined
quality and resource thresholds supersede the older stronger aspirations below.
TinyStories and WikiText are now prepared; Maxout, the curve-only branch and
BlockShuffle completed their language screen against full and compact dense
controls. None met both language margins on both corpora; those fixed recipes
are retired. The [shared FFN input basis](shared_basis_hypothesis.md) then failed
its initial hardware screen. The [smaller bank and gate-transport study](shared_gate_transport_hypothesis.md)
is complete: plain sharing passed the short memory screen but missed both language
margins on both corpora; added gate transport failed the hardware screen and was
retired before language training. Measurements are in the
[shared-bank report](retired_models/shared_basis_transformer/result.md).
The [polynomial sketches](polynomial_sketch_result.md) combined many feature-pair
products in a compact bank but failed the slowdown limit; their quality is unknown.
The [group-product hypothesis](group_product_hypothesis.md) instead uses simple
neighbor sums and multiplications. The first three execution recipes failed their
resource screen. The [fused fourth recipe](group_product_v4_hypothesis.md) passed
with about 25% less allocated memory than full GELU and fused-activation SwiGLU.
It misses the quality targets on both corpora and trails the learned-square
control on both, so this fixed recipe is retired. The
[current model report](retired_models/group_product_transformer/result.md)
records the two-corpus screen and removal measurements; the
[hardware history](group_product_result.md) preserves the earlier failures.
The [independent-context bank](context_bank_hypothesis.md) separates
learned detector features from a smaller set of learned context predicates.
Eight detectors share one bounded context signal. It passed its
[short resource screen](context_bank_result.md) and production equivalence/resume
checks. Its [language screen](retired_models/context_bank_transformer/result.md)
completed against full/compact dense, plain and matched additive controls.
Its completed TinyStories loss 2.672406 fails the dense quality margins; zeroing
its trained coefficients changes loss only to 2.674651. WikiText loss 4.367791
also fails both margins; coefficient zero changes it to 4.370834. Retire this
fixed recipe after the completed 16-run screen. These results do not establish
useful richer inference computation. The next
[signed-context experiment](signed_context_hypothesis.md) is registered and has
passed CPU/CUDA numerical checks. Both signed recipes passed the
[short resource screen](signed_context_result.md) and production equivalence and
resume checks. The [20-run language screen](../ffn_experiments/signed_context_transformer/result.md)
completed with compact/full dense and plain/closed/open gain controls.
Both signed recipes are retired: linear gains only 0.09%/0.27% over compact
dense on TinyStories/WikiText; tanh trails compact dense on both. The next
[latent FFN hypothesis](latent_ffn_hypothesis.md) tests 1280 nonlinear detectors
inside learned rank-128 input/readout spaces, against exactly matched thinner
factorized controls. CPU/CUDA numerical checks and production equivalence/resume
passed. Its [hardware screen](latent_ffn_result.md) passed on both corpora at
about 396 MiB allocated. The [18-run language screen](retired_models/latent_transformer/result.md)
completed. Wide-recompute NLL 2.726152/4.400809 misses both dense quality margins
on TinyStories/WikiText. The fixed recipe is retired; cached/thin controls and
all removals are recorded, and conditional confirmation will not launch.
The next [coupled channel-curve hypothesis](channel_curve_hypothesis.md) keeps
full-width input/output mixing and tests learned nonlinear feature interactions.
Its [CPU/CUDA preparation](channel_curve_result.md) passed; the registered
hardware screen completed with about 26% memory savings but failed the slowdown
limit; execution version 1 is retired before language. The separately registered
[version 2](channel_curve_v2_hypothesis.md) fuses branches and coefficient
reductions while keeping the mathematical recipe fixed. Its
[hardware screen](channel_curve_v2_result.md) passed at about 401 MiB allocated,
with no material throughput regression. CPU/CUDA numerical and production
equivalence/recovery checks passed. The [18-run language screen](../src/models/channel_curve_transformer/result.md)
completed. TinyStories/WikiText NLL 2.626786/4.298236 fail the dense quality margins;
self-gate controls perform better on both. Retire this fixed recipe after all
registered controls/removals. The next [basis-readout hypothesis](basis_readout_hypothesis.md)
tests separate learned combinations of nonlinear responses. CPU/CUDA numerical
and complete-model checks passed. Its [resource screen](basis_readout_result.md)
passed for fixed templates at about 405.5 MiB with a narrow slowdown margin;
learned templates failed and are retired before language. Production equivalence
and shared-Trainer recovery passed. The [18-run language screen](../ffn_experiments/basis_readout_transformer/result.md)
completed, test loss unopened. TinyStories/WikiText 2.604161/4.283474 miss both
dense accuracy margins; the tied control is better on TinyStories. Retire the
fixed recipe with all registered controls/removals recorded. Conditional confirmation
will not launch; sustained resources are unproven. A separately registered
[activation-map diagnostic](ffn_map_result.md) completed after campaign termination.
Its 8192-pair affine fits explain 45–62%/32–53% of held-out map variation on
TinyStories/WikiText; these are sampled map fits, not semantic retention or
language-quality guarantees. The next [feature-flow hypothesis](feature_flow_hypothesis.md)
keeps P/Q full width and adds a small repeated nonlinear mixer inside the FFN.
CPU/CUDA numerical checks passed. Its [resource screen](feature_flow_result.md)
completed: about 398 MiB allocated, but roughly 50% slower than full dense.
Retire execution version 1 before language. [Revision 2](feature_flow_v2_hypothesis.md)
fuses small group operations under the same mathematical recipe. CPU/CUDA
checks passed, including the actual 2048-token gradients; its
[resource screen](feature_flow_v2_result.md) completed: 26–28% less allocated
memory, but 11–21% longer updates. Retire revision 2 before language. The next
[execution hypothesis](feature_flow_v3_hypothesis.md) fuses the whole trajectory
and reverse pass. CPU/CUDA numerical checks passed; its
[resource screen](feature_flow_v3_result.md) completed. Five comparisons pass,
but TinyStories versus GELU requires 7.3% longer updates and fails the gate.
Retire revision 3 before language. [Revision 4](feature_flow_v4_hypothesis.md)
is registered to reuse reconstructed reactions/derivatives in backward registers;
CPU/CUDA checks passed; its [resource screen](feature_flow_v4_result.md) passed
all six comparisons at 401.7 MiB allocated. Production equivalence/recovery and
variance-calibrated dense initialization checks passed. The 22-run
[language screen](retired_models/feature_flow_transformer/result.md) completed;
[conditional confirmation](feature_flow_confirmation.md) is registered.
Primary TinyStories/WikiText losses 2.606496/4.300017 fail both quality margins:
2.68%/1.47% cost against full dense and only 0.32%/0.0045% gain over the
strongest compact controls. Diagonal controls outperform it on both corpora.
All 22 registered runs and removals completed. The
[terminal audit](../records/feature-flow-language-v1-terminal-audit.json)
verifies budgets, full validation and frozen source/recipe/checkpoint hashes.
Retire the fixed recipe; conditional tuning/confirmation will not run.
Readout-reuse execution [v4](readout_reuse_v4_result.md) subsequently failed
the full-model gradient tolerance. [V5](readout_reuse_v5_result.md) repaired
numerical equivalence, passing all 192 FFN fixtures, 24 full models and three
saved-failure regressions. Its 54-profile screen completed: 419.8 MiB training
allocation (22–25% less), but 0.865–0.903x throughput against full controls.
Retire v5 before language; quality is unknown and no candidate is qualified.

The original [readout-reuse hypothesis](readout_reuse_hypothesis.md) reallocates
weights to 1152 detectors by tying input/readout projections, with small learned
grouped readouts. Exactly sized ordinary dense, fixed/diagonal tied and wider
fixed controls test the width/readout tradeoff. CPU/CUDA mathematical,
initialization and recovery checks passed; its
[resource screen](readout_reuse_result.md) completed: 26–29% less allocated
memory, but throughput .76–.80x full dense. Retire execution v1 before language.
[Revision 2](readout_reuse_v2_hypothesis.md) is registered to retain projections
and fuse reconstructed response/readout work under the same equations and gates.
Its [numerical report](readout_reuse_v2_result.md) retires v2 after a reproducible
whole-model BF16 failure; fused forward accumulation changes a few cast values
per layer. No resources/language were measured. [Revision 3](readout_reuse_v3_hypothesis.md)
is registered to preserve native grouped forward while testing fused adjoints.
Its [actual numerical report](readout_reuse_v3_result.md) retires v3: forward
matches, but whole-model BF16 embedding gradients fail. [Revision 4](readout_reuse_v4_hypothesis.md)
is registered to preserve native grouped adjoints and scalar backward ordering.
Tying and grouped projections are known mechanisms; no novelty or quality claim.
No FFN replacement has met the acceptance contract.

The next [paired-readout study](paired_readout_hypothesis.md) tests directed fixed
pairing with a wider tied feature bank and saved readout activations. CPU equations,
gradients/recovery, 120 GPU fixtures and 20 complete models passed. Its
[result report](paired_readout_result.md) records 48 completed profiles: v1 passes
against native full SwiGLU but misses the all-control gate. Retire before language.
[V2](paired_readout_v2_hypothesis.md) registers local backward-buffer reuse with
all original gates unchanged. Parameter reduction alone does not qualify the model.

Post-rewrite measurements are recorded separately in each model's report:
[dense baseline](../src/models/base_transformer/result.md),
[BlockShuffle](retired_models/blockshuffle_transformer/result.md), and
[looped BlockShuffle](retired_models/looped_blockshuffle_transformer/result.md).
These development results do not replace the confirmation requirements below.

The earlier registered mechanism was [PairFlux](pairflux_hypothesis.md), with its
equation, prior work, failure predictions and independent exchange/curve ablations.
Measurements and its research decision are recorded in the
[PairFlux report](retired_models/pairflux_transformer/result.md).
That first nine-recipe screen is complete. The combined design failed the
qualification targets. Isolated-component NLL gains are exploratory, and their
post-training removal barely changes NLL, so richer inference computation has
not been demonstrated. The next mechanism requires a separate registered study.

Active research plan and experiment guide — updated 2026-10-04.

This document owns the current FFN research scope, hypotheses, historical evidence
and evaluation contract. The broader ambition and system design remain in the
[Atlas main research goal](research_goal.md). Setup and code layout are in the
[README](../README.md).

## Research direction

**Our current goal is to discover a faster, more capable replacement for the
Transformer feed-forward network: a computation that learns useful patterns and
composes feature interactions better, using substantially fewer learned weights.**
The replacement may change the arrangement of neurons, their connections, their
activation functions, or the operations they perform. The research question is
open; BlockShuffle is a starting candidate, not the definition of success.

Throughout this document, **FFN** means the feed-forward sublayer inside a
Transformer. The repository name `complex_fnn` retains the older spelling.
We are researching this component while training and
evaluating the complete language model around it.

Three ideas organize the project. The long-term vision is a compact system that
reuses computation and separates working state from stored knowledge. The active
research goal is a better FFN primitive inside an otherwise fixed Transformer.
The baseline screen is complete. The looped candidate did not improve the local
single-pass result. Feature exchange and learned curvature subsequently failed
their language screen. Shared detector banks then missed the language targets.
Grouped products missed the language targets. The latest branch tests a small
independent context bank that guides groups of detector features.
An experiment can fail without
invalidating the vision; the vision cannot establish that an experiment succeeded.

### What better pattern learning means

We want the learned transformation to use combinations of features reliably,
including combinations not presented in the same form during training. For
example, contextual features may encode two relations, an intermediate entity
and the requested answer. A useful transformation should help combine those
features rather than merely fit familiar examples. These are capabilities to
measure in the full model, not claims that individual neurons contain explicit
human-readable rules.

| Intended capability | Observable evidence | Limit of the evidence |
| --- | --- | --- |
| Learn useful language regularities | Lower held-out next-token negative log-likelihood (NLL) on both language corpora | NLL alone does not establish reasoning ability. |
| Combine information | Lower generated-answer error on relation chains and arithmetic, especially difficulty two and above | The rest of the model also contributes; isolate the FFN change with matched controls. |
| Generalize beyond familiar examples | Results on held-out identities and the separately reported OOD suite | The current OOD suite changes both length and names, so it cannot isolate their effects. |
| Preserve simple information use | No material regression on one-hop lookup | A composition gain should not conceal a retrieval loss. |

More hidden units, a more complicated equation, or a closer fit to training data
does not itself demonstrate better pattern learning. We need improved held-out
behavior under a controlled budget, repeated across seeds and supported by a
removal experiment that explains the useful mechanism.

### What faster means

Speed is part of the objective from the beginning. Fewer stored weights may
introduce extra operations, repeated passes, data movement or large intermediate
activations. Record three different costs:

* **Training throughput:** input tokens per second at identical batch, context
  and precision, with actual supervised targets reported separately.
* **Time to quality:** elapsed training time to a validation quality target fixed
  before comparison, including failures to reach it within budget.
* **Inference cost:** prompt processing and generated-token latency measured
  separately if we later claim faster generation. Training throughput cannot
  establish a serving speed gain.

The active numerical speed target below concerns training throughput. Show
learning curves and time to quality as well; faster steps may still require
more steps. Quality comparisons at equal tokens and resource comparisons at
equal quality answer different questions. FFN-only timing can diagnose a
bottleneck, but complete-model measurements decide the practical claim.

### Scope and priority

We seek an architectural contribution: a better way to compute the feature
transformation, with an explanation of its advantage. Compression, optimizer
tuning and implementation improvements are valuable outcomes, each with its own
accurate label. A better but slower model can inform later designs without
fulfilling the combined goal of better pattern learning and faster execution.

Keep full attention and the backbone fixed while exploring the FFN's internal
structure. The first milestone is evidence for one useful computational
primitive. The future encoder, persistent memory and complete Atlas system are
later research stages, not additions to the current comparison.

## Step 1 (in progress)

**Focus: improve only the FFN.**

**Goal:** discover and explain an FFN replacement, neuron arrangement or internal
computation that learns more useful feature interactions, runs faster in measured
training, and uses fewer learned weights inside our otherwise unchanged
full-attention Transformer. Seek an original contribution and establish its
relationship to prior work before claiming novelty. Establish the improvement
against both the full dense baseline and
the best parameter-matched dense baseline before building another Atlas component.
This is the active stage. Memory systems, new attention, encoders and whole-model
recurrence are deferred. An FFN still transforms one token's current features;
attention supplies its contextual information.

Hold layers, width, attention, positional encoding, normalization, embeddings,
tokenizer, context, training examples/order, precision and optimizer recipe fixed
within each comparison. Train each complete model from scratch; do not freeze a
pretrained backbone or transplant an already trained FFN. Shared tensors start
identically for paired seeds. Only FFN connectivity, hidden width, activation or
FFN-internal computation may differ. Change one mechanism at a time, then use
removal experiments to identify which part caused a gain. The study planner
rejects variant overrides of backbone fields.

**Selected data, in order:**

1. Generated lookup, relation-chain and modular-arithmetic tasks: 12,000 training,
   600 validation, 1,200 test and 1,200 out-of-distribution (OOD) examples before
   deduplication. Lookup measures retrieval; chains and arithmetic measure
   composition. Use answer-only loss and generated-answer exact accuracy, broken
   down by task and difficulty. The present OOD suite jointly changes length and
   entity names; it cannot isolate those two causes of failure.
2. TinyStories: the pinned 50,000 / 1,000 / 1,000 story recipe is the primary
   language experiment. It is small enough for rapid experiments but does not
   establish broad language capability.
3. WikiText-2 raw: the pinned complete official article splits are the required
   second language test. Train fresh models on this corpus with identical settings
   across candidates; compare losses within a corpus, never across tokenizers.

FineWeb-Edu is deferred until these tests qualify a candidate; its unused
preparation branch was removed during cleanup. The active dataset recipes are
defined; their presence is not a claim that remote data has been downloaded.
The dataset section below defines exact splitting, deduplication and source attribution.

**8 GB GPU budget:** begin at width 192, four layers, context 256, vocabulary
4,096 and batch eight, with BF16 compute and AdamW. Run one model at a time.
Target at most **6 GiB peak PyTorch-reserved VRAM**, leaving room for the display,
driver and other allocations; also inspect device-wide usage. Reserved memory is
not the same as total device usage. The full baseline has 2,557,632 parameters;
the structured candidate has 1,728,192, about 32.4% fewer overall. These are
component experiments, not yet 1M-total models. Profile all compared configurations
before committing a training budget. If one does not fit, lower the batch for
every model in that comparison and register the revised recipe before training.

**Success targets, fixed before confirmation:**

| Property | Target and interpretation |
| --- | --- |
| FFN size | At least 70% fewer FFN parameters than the full dense control; always report total model parameters too. |
| Language quality | Strong target: at least 1% lower mean held-out NLL than the best full dense baseline on both corpora, and a gain over the best equal-size dense baseline. Lower NLL means better next-token prediction. |
| Composition | At least 20% relative error reduction on held-out multi-step tasks against the strongest equal-size dense control; include dense-loop controls when testing loops. Lose at most two percentage points on lookup. Report OOD independently. |
| Memory and speed | Strong target: at least 20% lower peak training allocated VRAM **and** at least 1.2x training throughput versus full SwiGLU at qualified quality. Report reserved VRAM, total wall time and regressions too. |
| Scaling | After a confirmed win at width 192, repeat at widths 128 and 256 with four layers and comparable FFN compression ratios. Re-profile each size; do not extrapolate to billion-parameter systems. |

These targets are ambitious hypotheses, not expected outcomes. A smaller FFN
within 1% of full-baseline NLL and beating the equally small dense baseline earns
the narrower label **compression result**. A gain on one task earns a task-specific
claim. Only meeting all strong targets supports the combined claim; fewer weights
alone do not demonstrate faster execution or lower activation memory. No finite
benchmark proves universal superiority.

**Experiment order:** first run full SwiGLU, full GELU, narrow SwiGLU, narrow GELU
and plain BlockShuffle at the same 2,000-step development budget. One seed and one
rate can catch failures but cannot select a research winner. Give surviving
candidates and all relevant dense controls equal three-seed, three-rate tuning
effort using validation only. Test two-pass FFN reuse next, with both dense loops
and a dense compute-matched control; four passes follow only if two are useful.
Select the strongest full and compact controls using validation before looking
at test results. The primary composition score is the equal-weight mean error of
relation chains and arithmetic, restricted to difficulty two and above; preserve
the task/difficulty breakdown so a single easy group cannot hide a regression.
Learned activation shapes and changed connectivity are later independent branches,
each with a separate hypothesis and ablation. The existing 99-config plan lists
these options; it is not an instruction to run them all immediately.

Freeze the candidate, rate and a 10,000-step confirmation budget (1,000 warmup;
validation every 500 steps) before fresh seeds 4001, 4003, 4007, 4013 and 4019.
At batch eight/context 256 this is 20.48M input positions; report actual supervised
targets separately for masked tasks. Compare last checkpoints at equal budgets,
not whichever checkpoint happens to look best on test. Use fresh generated test
identities for confirmation and score real-text test sets only after selection.
Report all paired seed differences and a 95% confidence interval on mean NLL
improvement. Require its lower bound above zero in addition to the effect-size
target; otherwise report inconclusive and plan more seeds. Report uncertainty for
task accuracy too. Five seeds are a minimum, not a guarantee of sufficient power.

Measure real-data training speed with the same batch, context, precision, device
and software, excluding an explicitly recorded warmup. Repeat at least three
times and alternate model order to reduce thermal/order effects. Report median
throughput, spread, peak memory and time to the fixed quality target, including
failures to reach it. The synthetic `--profile` check only checks memory fit and
rough training compute; it cannot qualify a real-data or serving speed claim.

**Deliverable:** a reproducible comparison table, complete configurations, per-seed
results and one ablation explaining the benefit. Retain useful negative evidence.
Only then choose the next block of the larger system. Each model has an explicit
`transformer.py` with its full architecture. Attention, normalization and FFNs
are reusable components, and one shared trainer keeps the comparison protocol fixed.

## Candidate mechanisms

Discover a compact learned transformation that can represent and compose useful
interactions more efficiently than a strong dense feed-forward network (FFN).
Neural, non-neural and hybrid proposals are eligible. Learned activation shapes,
connectivity, multiplicative interactions, temporary state and repeated
computation are candidate mechanisms. No family is assumed to win.

The interface remains a token-local transformation from model-width features
back to model-width features. Temporary state is internal to that transformation
in this stage, not persistent memory across examples. Each proposal must define
its forward computation, learned parameter count, intermediate state, training
method and expected runtime cost. A non-neural or hybrid proposal must explain
how it can be fitted and evaluated fairly with the same model and data budget.

### Questions a candidate must answer

1. **Representation:** Which useful interactions should become easier to learn,
   and why might a dense FFN of the same size miss them?
2. **Computation:** Which work is removed, shared or reorganized, and which
   additional operations or tensors are introduced?
3. **Learning:** Does the advantage survive training from scratch, equal tuning
   effort and fresh seeds?
4. **Attribution:** Which control distinguishes the proposed mechanism from
   width, extra compute or ordinary optimizer effects?
5. **Practical value:** Does the complete model reach better held-out behavior
   with acceptable memory and improved measured speed?

Structured connectivity asks whether wide intermediate features remain useful
without dense connections everywhere. Multiplicative interactions ask whether
selected feature combinations help beyond the existing SwiGLU gate. Learned
activations ask whether adapting response shapes helps beyond affine rescaling.
Reused computation asks whether shared state updates improve composition enough
to justify extra runtime. These are separate hypotheses; each needs appropriate
controls and a clear condition under which it would be rejected.

The desired contribution is an explanation of *why* a mechanism works, a concrete
advantage, and a clear distinction from prior work. Optimizer improvements and
kernel engineering remain useful controls, but do not satisfy architectural
novelty. Reusing BlockShuffle or adding a loop does not itself establish novelty.

**Next hypothesis:** a wide structured transformation, reused on its changing
state, may learn compositions that an equally small narrow dense FFN misses.

This is the next available hypothesis because the repository contains the needed
models and controls. It is not a commitment to recurrence or BlockShuffle as the
final design. Establish the single-pass result first, then test whether two
passes add enough value to justify their cost. If this route misses the speed or
quality target, retain the evidence and revise the mechanism.

The initial candidate is the retained BlockShuffle SwiGLU primitive. Each dense
projection is replaced by two block-diagonal factors with fixed channel shuffles:

    W = inverse(P_out) B2 P_mid B1
    f(x) = down(SiLU(up(x)) * gate(x))

For input size n, output size m and G groups, a projection has
min(n,m)·(n+m)/G weights. This preserves a wide hidden representation without
storing every dense connection. It imposes structural restrictions and can cost
more runtime and activation memory despite fewer parameters.

The new **FFN-only loop prototype** updates the state within a Transformer block:

    s0 = x + attention(RMSNorm(x))
    s(k+1) = sk + f(RMSNorm(sk)) / K       for k = 0 … K-1

The same f and normalization weights are shared across K passes; attention runs
once. K=1 recovers the ordinary block. The 1/K gain is an explicit design choice,
not a theorem about stability. Fixed K=1,2,4 comes before a learned stopping gate.
This is a fresh implementation and training protocol, not a bit-identical replay
of the historical recipe or its recomputation kernels.

A token-local FFN cannot directly read other tokens. Its loops can manipulate
information already supplied by attention; they do not replace attention.
Whole-Transformer loops, context interaction and memory reads are different
hypotheses and must receive separate experiments.

## What we already know

Historical results below are retained evidence, not fresh measurements from this
reset. Full reports, configurations and original source are in the local archive.
The compact committed record is `records/history.json`.

| Branch | Observation | Decision / interpretation |
| --- | --- | --- |
| Plain BlockShuffle SwiGLU | 70.3125% fewer FFN weights. TinyStories: 0.855% lower NLL than calibrated narrow SwiGLU, 3/3 seeds. | Strongest structured FFN starting point with language evidence; recipes differed, so not pure architecture attribution. |
| Longer BlockShuffle run | WikiText at 3,200 steps: 0.1458% lower NLL than narrow, 2/3 wins; 1.2560% worse than full SwiGLU. Slower observed training, slightly higher peak memory than full. | Useful compression, no complete quality/resource win. |
| Learned activation corrections | The controlled affine gain was only 0.101%; broader rational/curve variants did not qualify. | No established activation winner; distinguish optimizer effects from learned shape. |
| H179 narrow shared FFN loop | 53.33% greater synthetic fitting error than best narrow control; 0/9 wins. | Reject that fixed recipe, not all recurrence. This was not looped BlockShuffle. |
| Fourier / dynamic connectivity branches | Some favorable synthetic tasks improved; conventional hard controls failed. | Useful negative evidence; no general language advantage. |
| Local interpolation H220 | Numerical qualification passed; zero learning updates. | Untested learning hypothesis, not a winning model. Design survives in archive. |
| H200/H201 narrow SwiGLU engineering | Strongest historical verified language/resource recipe; approximately 70.11% fewer FFN weights with more training tokens. | Conventional reference, not an architectural discovery. |

The recovered Atlas notes' 26M benchmark claims have not been independently
verified here and are not used as evidence for this FFN project.

## A baseline worth beating

The new baseline is a small **Llama-style dense decoder**: pre-RMSNorm, rotary
positions, residual connections, bias-free projections, SwiGLU, tied input/output
embeddings and full causal multi-head attention in every layer. There is no local
window, linear-attention approximation or disabled attention. PyTorch's scaled
dot-product attention may choose an optimized exact-attention implementation.

This is an established modern design, not a claim to reproduce a released Llama
checkpoint or the latest state of the art. Tied embeddings and full multi-head
attention are deliberate small-model choices; production GQA is not required to
test the FFN hypothesis. The primary experiment holds attention, tokenizer,
context, initialization of shared tensors and training data constant.

At width 192 and four layers, the study includes:

| Control/candidate | Hidden width | FFN passes | FFN weights |
| --- | ---: | ---: | ---: |
| Full SwiGLU | 512 | 1 | 1,179,648 |
| Full GELU | 768 | 1 | 1,179,648 |
| Narrow SwiGLU / GELU | 152 / 228 | 1 | 350,208 |
| BlockShuffle, eight groups | 1,024 | 1 / 2 / 4 | 350,208 |
| Narrow SwiGLU loops | 152 | 2 / 4 | 350,208 |
| Dense SwiGLU compute controls | 304 / 608 | 1 | 700,416 / 1,400,832 |

Two structured passes cost 59.375% of the full FFN's projection arithmetic;
four cost 118.75%. These figures exclude attention, normalization, elementwise
work, output logits and hardware overhead. Actual time and memory decide a
resource claim. The runner counts all learned parameters separately from FFNs.

## Datasets that distinguish the hypotheses

All raw/prepared data stays in `dump/data/`. Recipes are small committed JSON
files in `src/config/data/`. Remote sources use pinned repository revisions; cached
prepared files have SHA-256 checks. Tokenizers are trained on training data only.
Never compare perplexities from different tokenizers as though they were equal.

| Stage | Data and split | Question answered |
| --- | --- | --- |
| Plumbing | 2,000 historical TinyStories training stories / 128 validation stories; or tiny generated tasks | Does the pipeline work? Never a scientific result. |
| Mechanism | Generated lookup, relation chains and arithmetic modulo 97; 12,000 train, 600 validation, 1,200 test, 1,200 OOD examples before deduplication | Is the gain retrieval, composition or arithmetic? Exact answers come from deterministic rules. |
| Small language | TinyStories: 50,000 official training stories, 1,000 validation, 1,000 test | Does the component learn coherent simple language? Skip the first 1,000 historical validation stories. |
| Real-text transfer | WikiText-2 raw, complete official article splits | Does the result transfer beyond simple synthetic stories? This is still a small corpus. |
| Future broader language | Proposed bounded FineWeb-Edu sample: 30,000 train documents, 500 validation, 500 test | Does it survive varied educational web text? Preparation is deferred; the unused recipe and loader branch were removed. |

The generated task suite separates one-hop lookup from two/three-hop composition.
Its OOD split uses four-to-six hops/operations and renamed relation entities.
Graphs, expressions and split seeds differ. Fact order is shuffled, irrelevant
facts are included, and duplicate graph/query pairs ignore presentation order.
Arithmetic tests exact composition, not recall of stored biographies.
Report each task and difficulty separately. Answer-only training masks prompts;
exact-match evaluation generates answers without feeding gold answer tokens.

Language preparation removes normalized exact duplicates across document splits,
giving held-out splits priority. WikiText lines are assembled into articles.
Any future FineWeb preparation must define URL grouping and deduplication;
these would not provide a near-duplicate guarantee.
near-duplicate and benchmark-contamination audits are required before a major
claim. Streaming can still read substantial source shards: character/document
limits bound selected data, not necessarily network traffic.

Normal training uses validation only. Test and OOD scoring require `--final` after
freezing the configuration. Do not repeatedly select variants on those results.
After a failure influences design, that test is development evidence; generate a
new held-out suite for the next confirmatory claim. The command is a deliberate
opt-in, not a security barrier or a substitute for this discipline.

## Decision rules and experiment order

### Immediate work package

The controlled baseline and two-pass development screens are complete and
recorded in the model reports. Two-pass BlockShuffle did not improve the
single-pass candidate in the local screen; four-pass training remains deferred.

The latest completed work package is the registered PairFlux mechanism screen, with
independent removal of feature exchange and learned curvature, fresh dense and
structured controls, and the same 2,000-update development budget. Save complete
validation, quality curves, update time and memory. These exploratory measurements
determine whether the mechanism deserves tuning; they cannot certify a winner.
The separate proposal owns its fixed recipe and stopping rules. The original
study file remains a list of available comparisons, not an instruction to spend
the GPU budget on every configuration.

Before each new mechanism, record its equation, predicted benefit, closest prior
work, expected failure mode, required controls and stopping decision. End each
study with an explicit decision to confirm, revise or retire the tested recipe,
and explain that decision in the compact research record.

### Staged evaluation

1. **Numerical checks:** independent dense projection comparison, gradients,
   causality, loss masking, split integrity, parameter counts and resume fidelity.
2. **Development screen:** `src/config/study.json` defines 11 variants × three seeds
   × three learning rates. Planning creates configs; it does not launch 99 runs.
   Select each variant's rate by mean validation score with equal search effort.
3. **Mechanism confirmation:** freeze selected recipes, then use five fresh seeds
   (4001, 4003, 4007, 4013, 4019) and fresh test identities. Include parameter-matched
   dense controls, dense loops and projection-compute-matched controls. Also
   measure wall time; matched projection counts do not imply matched total cost.
4. **Language confirmation:** repeat on TinyStories and WikiText at equal sampled
   tokens, then extend duration and move to the bounded FineWeb subset. Longer
   training receives a new study ID. Compare fixed-duration endpoints and show
   learning curves; no cherry-picked seeds or unreported extra training.
5. **Scale and Atlas integration:** only after a reproducible component advantage,
   test an order-of-magnitude whole-model parameter advantage, then compressed
   semantic inputs and memory. The 1M-versus-10B ambition remains the final stage.

The Step 1 contract above sets the active strong targets. The following minimum
qualification gates describe narrower useful outcomes, not the combined claim:

* **Compression:** at least 70% fewer FFN weights; report total model reduction.
* **Language:** at most 1% relative NLL cost against full SwiGLU and full GELU,
  plus a consistent gain over the strongest equally sized dense control on both
  language corpora. Report paired per-seed differences and confidence intervals.
* **Compositional mechanism:** at least 20% relative error reduction on held-out
  multi-step tasks against both compact dense and looped dense controls, with no
  more than two percentage points lost on lookup. Report OOD separately. Near a
  score ceiling, freeze an absolute-margin alternative before seeing test scores.
* **Resource benefit:** target at least 20% lower measured training peak memory or
  1.2× higher measured throughput at the qualified quality; report both metrics
  and all regressions. This qualifies a scoped resource result, not every goal.
* **Novelty:** identify closest prior work and the mechanism-specific ablation.
  Parameter savings, looping, learned activations and external memory are all
  established ideas. A new combination earns a contribution only with evidence.

These are early qualification gates, not the ambition itself. Failure closes a
fixed recipe; it does not prove an entire architecture family impossible. A
gain that only occurs on its own synthetic family does not earn a broad claim.
Changing a gate after seeing results creates a new, explicitly exploratory study.

## Prior work and provenance

These references are retained from the supplied charter. This documentation
revision is not a fresh literature review or verification of the papers. Before
claiming an original mechanism, compare the closest methods, identify inherited
ideas and state exactly what differs in the proposal.

* [Structured FFNs / BlockShuffle](https://arxiv.org/abs/2406.16450): source of the
  retained structured parameterization; our implementation is not its invention.
* [Monarch](https://arxiv.org/abs/2204.00595): related structured matrix family.
* [Llama 3](https://arxiv.org/abs/2407.21783): established decoder design reference;
  this small baseline has deliberate differences described above.
* [Universal Transformers](https://arxiv.org/abs/1807.03819) and
  [Ouro](https://arxiv.org/abs/2510.25741): recurrence and shared Transformer
  computation are prior art. Ouro's controlled storage/composition distinction
  motivates separate tests here; it does not validate FFN-only looping.
* [Learnable activations](https://arxiv.org/abs/1412.6830) and
  [KAN](https://arxiv.org/abs/2404.19756): activation learning is an established
  research direction, so any new curve needs a specific contribution.
* Dataset sources: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
  [WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
  [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
  Preserve source dataset attribution and licenses when redistributing data.

Status at reset: historical findings retained; the new framework is a research
starting point. No new architecture has passed the scientific gates above, and
no production-sized training campaign was started as part of cleanup.
