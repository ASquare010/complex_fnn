# H009: depth-specific scalar functions over shared projections

Frozen before candidate training, 2026-09-06. State: IMPLEMENTING.

## Question and mechanism
Can very small per-layer changes to a reusable feature bank recover specialization
lost by strict FFN sharing? The matrices stay shared across each four-layer group;
each layer gets three hidden-width vectors controlling input scale, shift and
output scale of its scalar gates. This is related to FiLM and relaxed sharing,
not a claim that affine modulation or parameter sharing is new.

For u=Ux, v=Vx, shared U,V,D, layer l:
a_l=.5*tanh(theta_a,l), b_l=tanh(theta_b,l), c_l=.5*tanh(theta_c,l).
F_l(x)=D[(1+c_l) * SiLU((1+a_l)*u+b_l) * v].
Initialize all theta=0 to obtain the strict shared SwiGLU exactly, with live
gradients. Evaluate using residual arithmetic (u+a*u+b, y+c*y) to preserve
identity initialization under BF16. No changed attention, extra norm or router.

## Counts and limited guarantees
Three vectors per layer, 3h parameters. With sharing span 4:
unique FFN parameters=ceil(L/4)*3dh + L*3h.
At d=192,L=4,h=512: 301056 unique FFN weights, 74.4792% fewer than full
SwiGLU's 1179648; total 1679040 versus 2557632. Matrix FLOPs are unchanged
relative to full SwiGLU. Elementwise work and activation storage increase.

Scale factors stay between .5 and 1.5 and shifts between -1 and 1.
At the scalar gate with its value input held at 1, the derivative magnitude is
bounded by 2.25 times the SiLU derivative bound. This is not a nonvanishing or
full-network Jacobian guarantee. Learned matrix directions remain shared.

## Protocol and comparisons
First verify exact strict-sharing initialization, unique counts, finite/live
control gradients and serialization. Then run 800 steps, seed 17, same existing
micro_v1 cohort: compare strict shared SwiGLU, full SwiGLU, narrow SwiGLU and
GELU controls. This budget is justified by the observed reversal of activation
rankings between the 200- and 800-step controls.

A meaningful gain over strict sharing earns multiple seeds and component
ablations (scale only, shift only, output weighting only). If it approaches the
gold target, compare layer-wise rank-4 LoRA as an established relaxed-sharing
control. Do not claim mechanism novelty from a FiLM-like modification.

If full-quality matching occurs, also test larger depth/width and a broader
corpus before accepting it. No additional width, tokens or tuning are hidden.


## Completed seed-17 result
NLL 3.13918 versus strict shared SwiGLU 3.14017: only .00099 absolute change,
well below what one seed can establish. Training throughput 20,840 tokens/s;
full-sequence forward throughput 68,597 tokens/s, versus strict sharing 85,060.
Peak live allocation 274.7 MB versus strict sharing 255.2 MB. Decision: do not
promote this scalar adaptation; preserve the implementation and learned curves.
It does not close the gap to full SwiGLU 3.06505.
