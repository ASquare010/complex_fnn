# Nonlinear feature flow: execution history and language evidence

[Registered hypothesis](feature_flow_hypothesis.md). CPU/CUDA mathematical,
gradient, finite-difference and complete-model checks passed. The registered
48-profile whole-Transformer resource screen completed. Allocated peak about
397.7 MiB, but update time about 50% longer than full dense; the slowdown gate
fails on both corpora. Retire version 1 before language; its quality is unknown.
The separately registered [execution revision 2](feature_flow_v2_hypothesis.md)
tests fused group updates/adjoints without changing the mathematics or thresholds.
Its [completed screen](feature_flow_v2_result.md) also fails the slowdown limit;
[revision 3](feature_flow_v3_hypothesis.md) is registered to fuse the entire
trajectory and reverse pass. Both retired executions have unknown language quality.
Revision 3's [completed screen](feature_flow_v3_result.md) passes five resource
comparisons but fails TinyStories versus GELU. It is also retired before language;
[revision 4](feature_flow_v4_hypothesis.md) is registered to reuse the reconstructed
reactions and derivatives inside backward registers.
Revision 4's [resource screen](feature_flow_v4_result.md) passed all six
comparisons, followed by exact production equivalence and checkpoint recovery.
Its [language screen](retired_models/feature_flow_transformer/result.md) completed:
all 22 runs and removals, with a
[verified terminal audit](../records/feature-flow-language-v1-terminal-audit.json).
Primary TinyStories/WikiText NLL 2.606496/4.300017 costs 2.68%/1.47% against
full dense and gains only 0.32%/0.0045% against the strongest compact controls.
Both quality margins fail on both corpora; diagonal controls outperform the
primary on both. Retire this fixed mathematical recipe after its resource-passing
execution revision. No conditional tuning/confirmation will run.

P/Q stay full width and run once. A small learned block matrix evolves hidden
features through bounded nonlinear responses at three internal steps; the final
scalar response follows those mixes. This adds nonlinear interaction depth,
unlike adjacent linear B/Q readouts in the retired basis experiment. Groups
still limit internal interaction and have no known semantic locality. No token
state, attention changes or external memory. M starts at zero, and all controls
start at the same initialized function.

2,162,688 FFN weights, 74.42% fewer than full SwiGLU. Forward projection work
4,587,520 FLOPs/token; training projection work including reconstructed states
14,155,776 plus scalar/reconstruction/reduction work. FP32 hidden evolution and
grouped GEMMs, BF16 P/Q. The custom operation saves z/M references and reconstructs
only one layer's states in backward; its real memory/time remains to be measured.

CPU FP64 independent full-matrix/scalar equations and z/M adjoints passed at
zero/one/three steps with/without internal feature removal. Finite differences
passed, quadrature agrees at 128/256 nodes, mean matches the analytic Gaussian
variance/2. All five complete models passed initialized-function equality,
causality, token locality, counts, save/load and exact CPU AdamW recovery.
CUDA FP32/BF16 independent adjoints passed at widths 512/37/1 and complete/
incomplete shapes; complete models use nonzero learned matrices for those checks.
Frozen interventions agree with ordinary autograd. BF16 casting uses Torch;
tested ties, zeros, subnormals, infinities and NaN classification agree. Raw
kernels read/write FP32 only. All actual errors/tolerances are retained.

Evidence: `records/feature-flow-v1-checks.json`; frozen profile protocol/source
and completed `result.md` under `dump/feature-flow-v1/`. This is exact discrete
backpropagation, not a continuous adjoint. No resource/quality claim from equations.
