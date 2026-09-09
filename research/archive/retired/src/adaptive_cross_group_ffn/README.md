# Adaptive coupling of projected values

Hypothesis: optional neighbor interactions fix the forced-coupling failure of
cross_group_ffn while retaining every original grouped FFN at zero initialization.
The existing value features are mixed with a cyclic neighbor using one bounded
coefficient per hidden channel. There are three grouped projections, as before.
See the [equations, proofs, budget and frozen tests](../../../../adaptive_coupling_plan.md).

At the LM screen dimensions this adds 2048 parameters and retains 74.8264% FFN
reduction. Parameter savings do not guarantee latency or activation-memory gains.
The fixed mixer's singular-value bound does not prove Transformer stability.
Existing cross-feature gating and structured projections are prior art; novelty
remains unverified. Verdict: REJECTED for LM promotion: 200-step NLL 4.63076 versus narrow
GELU 4.24112. Three-seed function tests repair the fixed mixer's additive failure
but have inconsistent pure-product optimization. See the
[completed report](../../../../interaction_report.md).
