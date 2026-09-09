# H013: match decay of the represented projection - frozen ablation

Factor-aware learning rates change AdamW's effective shrinkage of a factored
matrix. This is a second optimization variable that needs an explicit control.
For W=B2 P B1, with factor LR multipliers k1,k2 and the same decay q, a
zero-gradient step yields W_new=(1-eta*k2*q)*(1-eta*k1*q)*W. The first-order
shrinkage is eta*q*(k1+k2), rather than eta*q for the dense reference.
At h=832, G=8, the up/gate multiplier sum is 8; the down sum is 21.3333. This is a
large regularization difference over 800 steps, even though each scalar step
is small. It may help or hurt quality; the algebra alone does not decide.

Test q_i=q/(K*k_i) for a K-factor projection. Then pure decay is
(1-eta*q/K)^K*W = (1-eta*q+O((eta*q)^2))*W. For K=2 the absolute multiplicative
difference from the dense step is exactly (eta*q)^2/4. This matches decay
through first order, not the complete optimizer or gradient updates. Decay of
ordinary dense FFN maps and non-FFN parameters stays unchanged. For a standalone
grouped map K=1 the match is exact. Learned nonlinear coefficients retain no decay.

This is an elementary optimizer-accounting correction, not verified novel prior
art. Matrix-product regularization is an established topic; see the existing
structured-layer sources in literature.md before any priority claim.

Freeze one ablation: blockshuffle_swiglu h=832, G=8, fan_in LR, product decay,
800 steps, seed 17. Compare with the in-progress parameter-decay h832 run and
existing full/narrow references. Same data order and initialization; only decay
coefficients change. If it improves quality, require seeds 29, 43 before acceptance.
The CLI records --ffn-decay-mode product and actual per-group decay/LR values.
Do not present this as an equal-decay-coefficient run; it matches represented-map
shrinkage through first order. No additional parameters or inference operations.


## Recorded outcome
At 800 steps, seed 17, product decay reaches NLL 3.118254 compared with
3.115322 for parameter decay. This ablation does not improve the selected
quality result, so the width and replication runs retain parameter decay.
One seed does not establish statistical superiority of either decay rule.
The optimizer-accounting derivation and implementation remain useful controls.
