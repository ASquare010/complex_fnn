# H011: optional coupling of projected values - frozen before training

## Evidence motivating this change
The fixed input mixer in H008 solves its constructed polynomial but damages
the additive negative control. It also fails the 200-step LM gate. Forcing all
features to mix does not preserve the uncoupled function class. Retain the
original group-local value and make each cross-group contribution optional.

## Architecture and exact parameter accounting
Let u=Ux, v=Vx, p=Pv, lambda=.75*tanh(theta), with one theta per hidden channel.
F(x)=D Q [SiLU(u) * (v+lambda*p)/sqrt(1+lambda^2)].
U,V,D and Q match the grouped baseline; P shifts whole projected-value groups.
All theta initialize to zero. The initialized forward equals the uncoupled
FFN exactly, including its existing matrix initialization. There are still
three grouped GEMMs. Additional roll and elementwise operations must be timed.
This mixes after V; H008 mixed before V, so it is not the identical operator.

At d=192,h=512,G=4,L=4: 296960 unique FFN weights (74.8264% reduction),
1674944 model weights. The additional 2048 weights cost .174 percentage points
of FFN reduction. The normal dense FFN reference has 1179648 weights.

## Structural results, with scope limits
1. Setting theta=0 recovers every original grouped FFN at the same U,V,D.
2. Pure cross products a*d are representable. In group 0 set two up features
   to a,-a, its own two value features to zero, neighboring group 3 values to
   d,d and its up features to zero. Set the two active coefficients to .5.
   Using SiLU(a)-SiLU(-a)=a and compensating output weights gives a*d exactly.
   Thus the family strictly contains the original additive family for the
   demonstrated widths. Adding original own values also gives a*(a+.5*d).
3. In value space the mixer is A=R(I+diag(lambda)P), where
   R=diag(1/sqrt(1+lambda^2)). Since max|lambda|<=.75,
   sigma_min(A)>=(1-.75)/sqrt(1+.75^2)=.2 and sigma_max(A)<=1+.75=1.75.
   This elementary bound prevents this value mixer alone becoming singular;
   it does not control U,V,D, SwiGLU derivatives, or whole-network gradients.

This is a proposed combination of existing structured projections and feature
gating, not certified novelty. The proof is a representational result, not a
sample-complexity theorem or evidence of a better Transformer.

## Frozen experiments
First verify the exact inclusion, explicit polynomial/product constructions,
input/parameter gradients, parameter counts and bounds independently. Then
run the unchanged 200-step seed-17 LM screen at G=4. Controls are grouped
SwiGLU (exact base matrices), fixed H008, narrow GELU/SwiGLU and full references.
Promote only if >=1% better NLL than the better narrow control and no catastrophic
speed/memory regression. Otherwise reject LM promotion in this setting.

Function diagnostic: same three targets, data, optimizer, 1000 steps and seeds
17,29,43 as H008. Adaptive variant has 65 parameters versus grouped/narrow 49;
add a narrow SwiGLU h=5 control with 61 parameters (closest smaller width).
Report the remaining four-parameter advantage. This extra-budget control is
required even if the exact-budget 49-parameter baseline loses. Save all trials.


## Completed outcome
The 200-step LM NLL is 4.630757; the candidate fails the frozen promotion gate.
The three-seed function diagnostic and the extra-budget control are complete.
See [all results and limitations](interaction_report.md). No further training
of this LM branch is currently planned. This outcome does not negate the
exact function-class inclusion or constructed separation proved above.
