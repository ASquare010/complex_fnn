# Internal learnable curves: local proofs and limits

H082 tests a different location for learned curvature: between the two factors
of each BlockShuffle projection. The scalar shapes use familiar nonlinearities;
this note does not establish a novel activation family or a better FFN.

## Definition and parameter cost

For group controls theta_a,theta_b, let

    a = (1/2) tanh(theta_a),  s = exp(log(4) tanh(theta_b)),
    phi_q(z) = z + a s q(z/s)^2,  q = tanh or sin.

Thus |a|<=1/2 and 1/4<=s<=4, using closed bounds that include parameter limits.
Replace the linear projection P_out^-1 B2 P_mid B1 x by
P_out^-1 B2 P_mid phi_q(B1 x), elementwise with group-shared controls.
The middle dimension is384, not the outer hidden width2048. Three projections
therefore evaluate1,152 additional scalar curves per token and spend48 weights
per standalone FFN. This adds pointwise work; it does not replace the existing
outer SwiGLU or prove a memory/speed reduction.

At theta_a=0, phi is the identity for every theta_b. All plain projection tensors
and functions are contained exactly in real arithmetic. Consequently theta_b is
initially unidentifiable and its gradient is zero; amplitude must learn before
scale can receive a shape-dependent gradient. This is not a convergence result.

## Scalar derivative bounds

For tanh, write u=z/s and t=tanh(u). Differentiation gives

    phi_tanh'(z) = 1 + 2 a t (1-t^2).

On [-1,1], the largest absolute value of t(1-t^2) is2/(3 sqrt(3)), attained at
t=+/-1/sqrt(3). Hence

    1 - 2/(3 sqrt(3)) <= phi_tanh'(z) <= 1 + 2/(3 sqrt(3)),

approximately[0.61510,1.38490]. This bound is independent of s.
For sine, differentiation gives

    phi_sine'(z) = 1 + a sin(2z/s),

so its derivative lies in[0.5,1.5]. The equal-count affine control
psi(z)=z+0.5 tanh(theta_a)z+0.5 tanh(theta_b) has the same[0.5,1.5] bounds.

By the mean-value theorem, each curved scalar function is bi-Lipschitz with
these positive lower/upper constants. Since |phi(z)-z|<=|a|s, its limits at the
two ends of the real line are respectively minus/plus infinity. It is therefore
a bijection R->R with inverse Lipschitz constant at most the reciprocal lower
bound. This is a statement about exact real arithmetic. Rounded floats can
collide, and very large sine arguments can be inaccurate or overflow after
division. Numerical tests cover declared finite ranges, not every floating value.

Positive input slopes protect only this scalar operation. Learned matrix factors
may have nullspaces or poor conditioning; products, the outer SiLU gate, depth
and optimizer can still lose useful gradient directions. Tanh control coordinates
can saturate. Do not turn this lemma into a whole-model nonvanishing-gradient or
fast-learning guarantee.

## Connection to the user's quadratic Bezier proposal

Let t=(1+q(z/s))/2. With Bezier controls P0=P2=a s and P1=-a s,

    B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
         = a s (2t-1)^2 = a s q(z/s)^2.

Thus phi=z+B(t). This is a constrained quadratic curve with an identity path,
not a claim that all Bezier controls or all scalar functions can be learned.
Both q choices satisfy q'(0)=1 and q(0)=0, so phi''(0)=2a/s. For a!=0 the
scalar function is non-affine, while phi(z)+phi(-z)=2a s q(z/s)^2 is nonconstant.
An affine map cannot represent that scalar function exactly. The pre-existing
full SwiGLU FFN is already nonlinear; this argument separates only projection
maps, not complete FFN function families or finite-budget approximation error.

## A preserved structural limitation

The projection Jacobian is P_out^-1 B2 P_mid D(x) B1, where D is diagonal.
In the fixed d384/G8 canonical partition, each block still factors through the
same six routed intermediate coordinates. Its rank remains at most6. Inserting
these diagonal scalar derivatives does not remove the local linear-rank bound
or defeat H079's Hadamard matrix witness. The Jacobian now depends on input;
whether that helps learn nonlinear targets is an empirical question.

## Prior art and scope

[Ziyin, Hartwig and Ueda (2020)](https://arxiv.org/abs/2006.08195) introduce an
identity-plus-squared-sine activation for periodic inductive bias. Our sine
control changes its parameter bounds, identity initialization and placement;
it is explicitly Snake-related, not a new periodic principle. The tanh choice
is a smooth nonperiodic counterpart. A keyword search is not a novelty audit.

[Baker et al. (2024)](https://arxiv.org/abs/2402.06751) analyze how architecture
and activation linearity influence gradient-rank bounds. Their results do not
prove these particular curves increase useful rank or improve language quality.
[Ganea et al. (2019)](https://arxiv.org/abs/1902.08077) study learned monotonic
functions on logits to address the softmax bottleneck. That placement and rank
notion differ from the internal Jacobian restriction above.

All counts, derivative calculations, identity claims, finite differences and
initial/checkpoint equality must be checked by the H082 implementation. Even a
successful fitting result remains synthetic evidence before a separate resource
and language comparison. The gold research objective is unchanged.
