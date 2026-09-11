# H151: fixed tail map is a capacity constraint

Previous turn: progress. H150 verified candidate resource/stability gains but
failed two baseline stability checks. Before allocating more training, test a
necessary representational property. This does not reopen a failed timing gate.

For h_j=Q_j*h_(j-1)+b_j+r_j and coordinatewise
r_j=a_j*z_j/(1+abs(z_j)), ||r_j||<=||a_j||. Unrolling gives
F(x)=A*x+beta+R(x), A=Q_L...Q_1, beta the composed affine offset,
||R(x)||<=B=sum_j ||Q_L...Q_(j+1)||_op*||a_j||.
For orthogonal Q and |a_ji|<=c/L, B<=c*sqrt(d), independently of all biases
and learned shape values. Thus lim_(t->infinity) F(tv)/t=A*v for every v:
training shapes/biases cannot change the asymptotic linear map.

For mean-zero isotropic X with Cov(X)=sigma^2 I and linear target Mx,
center F(X)-MX. Reverse triangle in L2 and Var(R)<=E||R||^2<=B^2 give
E||F(X)-MX||^2 >= max(sigma*||A-M||_F-B,0)^2.
No Gaussian assumption is needed. Nonzero means are handled by centering X.
This holds for every allowed theta,bias, not just saved checkpoints. For a
finite sample replace sigma*||A-M||_F by RMS((X-meanX)*(A-M)^T).
A fixed translation cannot evade the centered bound. These are lower bounds,
not claims of attainability. Arbitrary learned outer maps or trainable Q change
the conclusion and need new cost/stability analysis. Compact-domain approximation
and scalar scaling layers are not ruled out by this statement.

Exact witness: d16,L8,c2,Q_j=I,target M=-I, inputs uniform on +/-4*sqrt(d)*e_i.
Cov(X)=16I, ||A-M||_F=8, B<=8. Thus total MSE>=576, target energy256:
normalized error>=2.25, exceeding the zero predictor's1. This is an exact
integer/rational certificate of one family-wide obstruction, not a Monte Carlo
proof. Same result applies to any exactly orthogonal Q with target -A.

CPU-only audit of all six H150 learned:fused repeat0 saved models, no fitting.
For each use a fresh256x384 Gaussian sample (seed281+source seed+source batch),
scales1/4/8, targets original independent orthogonal map and -A. Eighteen forward
passes /36 comparisons. Save input/output/matrices and test decomposition and
pointwise residual bound, finite-sample universal risk bounds, and ray convergence
bound using the existing samples: ||F(x)-Ax||<=(||beta||+B).
For stored approximately orthogonal matrices use conservative norm estimates
sqrt(1+||Q^T Q-I||_F) with1e-12 numerical padding and products of those bounds.
These floating estimates are checks, not formal interval certificates.

Freeze sources/prior receipts first. Independent audit recomputes all saved-data
risk inequalities and the exact integer witness without invoking model.forward.
No GPU, optimizer updates or backwards. Document relation to extrapolation prior
art, H147 contraction bound, and which architecture changes could evade this
constraint. The broad VRAM-and-quality research goal remains open.
