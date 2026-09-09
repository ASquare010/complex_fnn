# What the 70% budget forces in a multi-head FFN

Dense input/output mixers consume most of this small budget. This explains the
narrow heads in H045/H046 and identifies a simpler, exactly matched control.
The analysis changes no frozen architecture or running experiment.

## Exact budget bound

For width d divisible by three, full SwiGLU has 8*d^2 FFN weights per layer.
The parallel multi-head architecture has

    P = 2*d^2 + 3*d*E*de + d*E,

including its router. Keeping at most30% of full weights requires

    E*(3*de + 1) <= 0.4*d.

If each private subnetwork keeps expansion de >= (8/3)*dh, then necessarily

    dh <= (0.4*d/E - 1)/8.

At d=384 and E=2 this gives dh <=9.475. Because heads divide384, the largest
feasible head width is8, with48 heads. Thus the very small head width follows
from the parameter constraint and expansion requirement; it was not selected
by observing validation NLL. Rounded hidden widths can only tighten the bound.
The [published architecture](https://arxiv.org/html/2512.06989v1) evaluates much
larger heads; our small adaptation cannot establish a published-scale conclusion.

## A simpler control with exactly the BlockShuffle budget

Use one headwise SwiGLU per head, dense input/output mixers, and no router:
d=384, H=24, dh=16, de=48. Its parameter count is

    P_single = 2*d^2 + 3*d*de = 350208 per layer.

Across eight layers this is **2,801,664 FFN weights**, exactly the current
BlockShuffle and calibrated narrow budget:70.3125% fewer than full SwiGLU.
It has1,152 hidden features per token, versus2,304 for the current E=2 comparator
and2,048 for BlockShuffle. All private matrix dimensions are multiples of16.
Fewer intermediate features and removed routing suggest lower execution cost;
measured quality, memory and timing are still required.

This is the headwise architecture without parallel routing, a prior-art-inspired
control rather than a claimed new primitive. Removing an E=1 epsilon-normalized
router is an explicit architecture change: that router is not exactly identity
for finite logits. No unmeasured speed or expressivity ordering is claimed.
The larger per-head input subspace and fewer hidden features trade different
forms of capacity. Neither architecture is shown to contain the other.

## Why calibrated RMS is slightly above the dense reference

For a Gaussian vector q with dh coordinates, E[||q||^4]=dh*(dh+2). For independent
random private gate/value weights of variance H*sigma^2, the leading quadratic
term of SiLU(u)*v has second moment proportional to

    d^2 * sigma^4 * (1 + 2/dh).

The corresponding full dense expression is proportional to

    d^2 * sigma^4 * (1 + 2/d).

After the H045 down/mixing calibration, the leading-order RMS ratio is therefore
approximately sqrt((1+2/dh)/(1+2/d)). At dh=8,d=384 this is about1.115; H045 measured
1.11925. Higher SiLU terms and finite sampled weights explain why this is only
an approximation, not an identity or a fitted validation result. The calculation
concerns expectations over Gaussian inputs and independent random weights.

This suggests a possible analytic refinement of initialization, but none is
applied to H046. Exact norm preservation of orthogonal mixers, this moment
calculation, training behavior and whole-network gradient stability remain
separate claims.
