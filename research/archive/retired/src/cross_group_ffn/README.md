# Cross-group coupled SwiGLU

Hypothesis: the grouped sandwich loses cross-group nonlinear interactions.
Mix the value input with its neighboring group before the existing gate.

F(x) = D Q [SiLU(Ux) * V Mx],
M = (I + 0.5 P) / sqrt(1.25).
U,V,D are the same grouped matrices as grouped_ffn, Q shuffles hidden channels,
and P cyclically shifts input groups. No new learned weights or attention changes.
At d=192,h=512,G=4,L=4: 294912 unique FFN weights, 75% FFN reduction,
1672896 total model weights. Matrix FLOPs match the narrow SwiGLU control;
roll, add and normalization cost extra execution and memory.

The uncoupled FFN is additive across input groups. The candidate has nonzero
mixed derivatives across neighboring groups. The fixed mixer has singular
values in [0.4472,1.3417]. These are structural statements about the FFN input,
not proof of improved learning or whole-network gradient stability.
See [derivation and frozen experiment](../../../../cross_group_hypothesis.md).

This is related to existing cross-gating and structured-matrix research. No
novelty or superiority is claimed. Risks include still restricted interactions,
conditioning of the learned matrices, and additional eager-operation overhead.
Verdict: REJECTED for LM promotion at 200 steps, seed 17: NLL 4.63555,
versus narrow GELU 4.24112 and narrow SwiGLU 4.41907. The expressivity
separation remains valid; a constructed-function diagnostic tests optimization.
