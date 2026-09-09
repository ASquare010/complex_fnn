# Structural results and their limits

These are elementary project derivations, not claims of novel theorems or better LM quality.

## Static cubic banks collapse
Let B_k(t)=sum_j b_j(t)p_kj with cubic Bernstein basis b_j.
For input-independent mixing a_k:
sum_k a_k B_k(t)=sum_j b_j(t)[sum_k a_k p_kj].
This is exactly one cubic, with controls q_j=sum_k a_k p_kj.
Normalized mixing preserves common fixed endpoints; only two control degrees of
freedom remain for endpoints (0,1), however large K is. Unnormalized mixing still
lies in the same four-dimensional polynomial span. A common fixed base plus
residual curves has the same limitation. Optimization trajectories may differ.

Consequence: a static same-coordinate K=16 cubic bank cannot create 16 independent
nonlinear directions. Different coordinates, knot intervals, degrees or dynamic
gates can escape this result, at additional cost.

## Bounded residual curve
C(t)=3(1-t)^2*t*delta_1+3(1-t)*t^2*delta_2, |delta_j|<=c.
Bernstein positivity gives |C|<=3ct(1-t)<=3c/4.
With controls p=(0,delta_1,delta_2,0),
C'(t)=3 sum_j Bernstein_2,j(t)(p_(j+1)-p_j).
The Bernstein weights form a partition of unity, so |C'|<=6c.
For t=sigmoid(x/tau), |dt/dx|<=1/(4tau):
|dC/dx|<=3c/(2tau), which is .75 for c=.5,tau=1.
At theta=0, delta=c*tanh(theta)=0 but ddelta/dtheta=c: the residual starts
exactly zero with live gradients. A zero curve times a zero gate would not.
Tanh itself may saturate; monitor control parameters.

This bounds only the deformation. GELU derivatives can still vanish; matrix
spectra, normalization and residual paths determine whole-network stability.

## Nonlinearity cannot recover discarded input directions
For F(x)=B phi(Ax), A in R^(r x d), every v in ker(A) has F(x+v)=F(x).
Where differentiable, J_F=B J_phi A has rank <=r.
For any target T differing at x,x+v, the triangle inequality implies at least one
of the two errors is >=||T(x+v)-T(x)||/2. A richer phi cannot restore information
removed by A. The outer residual x+F(x) preserves identity but its learned update
still obeys this restriction. Structured full-rank maps merit investigation.

The present 75%-reduced dense FFNs have h=d, so they need not discard input
directions. The kernel theorem applies to actual rank-deficient projections,
not to every reduction in expansion width.
