# Feature flow: closer prior-work comparison

Written during the frozen language campaign. These notes do not change its
recipe, gates, controls or selected architecture. The literature search is
not exhaustive and does not establish originality.

[ODE Transformer for Sequence Generation](https://arxiv.org/pdf/2203.09176),
Section 3, reuses a normalized attention or FFN residual function inside
Runge–Kutta-style intermediate evaluations, with shared parameters and optionally
learned combination coefficients. Parameter reuse and iterative Transformer
computation therefore precede this experiment. Its evaluation concerns different
tasks/settings; its gains do not validate our compression or quality targets.

Our implemented comparison: P and Q each execute once per FFN call, while the
reused operation is a 32-coordinate block matrix acting on a bounded scalar
response. Three fixed Euler-style steps evolve only that token's internal
features. The backbone normalization/attention is unchanged, there are no learned
solver coefficients or adaptive termination, and differentiation follows the
fixed discrete computation. This is a concrete implementation distinction,
not proof of a new theoretical family or a first use of this construction.

[Neural ODE Transformers / DiffEqFormer](https://arxiv.org/html/2503.01329v2),
Sections 4.1–4.2, generates attention and FFN weights from continuous-depth
embeddings and a hypernetwork. It concerns depth-dependent Transformer dynamics.
Our FFN has layer-specific P/Q/M, with M fixed across its three internal steps,
and generates no weights from time. Sharing internal steps must be judged by
our retrained one-step control, rather than assumed beneficial.

[FFNs as key-value memories](https://arxiv.org/abs/2012.14913) relates learned
activation patterns to output token distributions. That interpretation does
not establish that more basis responses or internal interaction steps preserve
more useful information in our model. Validation comparisons and removal tests
are needed; matrix rank and frozen ablation damage are insufficient substitutes.

The registered hypothesis already acknowledges scalar gates, block-diagonal
matrices, recurrence and ODE interpretations as prior art. The currently
defensible claim is that this particular FFN mechanism is being tested under
fixed resource/quality comparisons. No novelty or language superiority claim
is established. Inspect closer author implementations before any surviving
candidate receives an originality claim.

The independent [local expansion audit](feature_flow_expansion.md) passed:
one-step and three-step first matrix derivatives at zero mixing differ by at
most 1.11e-11 in central differences. Halving epsilon gives quadratic state/output
differences and cubic residuals after subtracting the derived second-order term.
Evidence is `records/feature-flow-expansion.json`; no model, checkpoint or language
data entered this small CPU FP64 check. It explains why recurrence alone is not
evidence of added useful computation at initialization, without predicting the
trained model's loss or changing the ongoing controls.
