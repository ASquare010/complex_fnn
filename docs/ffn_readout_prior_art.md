# Readout reuse: research notes before choosing the next FFN

These are literature and algebra notes during the frozen feature-flow language
campaign. No new recipe is selected, implemented, tuned or measured here.
They do not alter the ongoing experiment, its budget or its acceptance gates.

[Energy Transformer](https://arxiv.org/html/2302.07253v2), Section 2's Hopfield
module and equation 9, already uses the same matrix for projecting into hidden
features and its transpose for projecting responses back. Its architecture
also changes attention and uses global energy dynamics. Those backbone changes
are outside this FFN-only goal. Transpose tying and associative-memory language
cannot serve as novelty claims for a future proposal.

[SLlama](https://aclanthology.org/2025.emnlp-main.1198.pdf), Section 3.3, names
Shared Projection MLP and ties expansion/reduction weights by transpose. Its
complete architecture also changes attention and layer sharing. Reported whole-
model results do not establish an isolated FFN improvement under our backbone,
data, compression threshold or controlled budgets. A future reused-readout
proposal needs a plain shared-projection control, not just an ordinary dense
comparison.

[Causal Energy Minimization](https://arxiv.org/html/2605.07588v1), Section 2.2,
equations 10–11, also ties a gated MLP's nonlinear projection and transpose
readout while retaining a separate multiplicative projection. Section 2.3
adds learned diagonal-plus-low-rank preconditioners and recursive updates.
Thus neither a two-projection tied SwiGLU nor a lightweight output
preconditioner would be a new primitive. Its energy differentiates the update
variable while conditioning on the input that supplies the gate; that is not
the same derivative as the complete input-to-output map. A future design needs
to identify any actual difference from these equations and test it directly.
Their moderate-scale results do not establish our 70% compression and two-corpus
quality/resource contract.

For a smooth scalar response phi, the normalized-input FFN map
f(x)=P.T*phi(P*x) has Jacobian P.T*diag(phi'(P*x))*P, which is symmetric.
This describes the FFN map itself, not the complete Transformer including its
normalization and attention. It exposes a representational constraint from
tying detector and value directions, rather than proving a capacity benefit.
Reusing weights saves storage, but a projection used twice still incurs both
forward products and both parameter-gradient contributions.

A possible direction to investigate after the current campaign is complete
is a small directed hidden readout C before P.T, yielding P.T*C*phi(P*x).
Its Jacobian P.T*C*diag(phi')*P is generally not symmetric. Block-diagonal C
would give a response a learned value direction in the span of its group's
input vectors. It would retain a constraint on each group's output directions;
extra detector width would not automatically compensate for that restriction.
This is an algebraic option, not evidence of novelty, speed, accuracy or retained
information. Grouped matrices and asymmetric associations also have prior art.

The reason to consider different weight allocation is local evidence: the
current feature-flow primary loses to diagonal/no-update controls on TinyStories,
and its frozen one-step intervention barely changes loss. Repeating a compact
512-feature bank did not establish useful extra depth in that allocation. A
future hypothesis should test independent feature breadth against readout
freedom explicitly, with matched ordinary dense and known tied controls, rather
than assuming recurrence or weight reuse supplies better information.

The older repository's archived paired-reciprocal SwiGLU implementation was
inspected with `git show HEAD:research/archive/retired/src/paired_feature_ffn/__init__.py`.
It splits one projected vector into a/b, concatenates SiLU(a)*b and SiLU(b)*a,
then uses an independent learned down projection. It does not reuse input
weights for the readout. Its equations therefore do not test transpose tying;
its existence is also a reason not to claim swapped gate/value responses as
a new idea. No removed folders were restored.
