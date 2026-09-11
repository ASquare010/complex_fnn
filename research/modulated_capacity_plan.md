# H144: capacity preflight for output modulation

Previous turn: progress. H143 removed a rank obstruction but failed learning and
resources. Consider a distinct single-pass family F(x)=b(Ux)+x*g(Ux), where * is
coordinatewise multiplication and rank(U)<=h. A concrete candidate uses one
GELU feature bank and two affine readouts (3dh+h+2d parameters). FiLM-like
modulation and diagonal-plus-low-rank structure have substantial prior art.

Before fitting, prove the following for x~N(0,I) and a linear target Mx, allowing
arbitrary measurable b,g with finite risk. The best population squared error is
inf_{D diagonal, rank(L)<=h} ||M-D-L||_F^2. A nonlinear gate cannot improve this
relaxed family over its best diagonal-plus-low-rank affine predictor on this
particular distribution/target. This does not apply to arbitrary non-Gaussian
inputs, nonlinear targets, independently conditioned extra inputs or deep stacks.

For even d and J=blockdiag([[0,-1],[1,0]],...), derive exact optimal risk
max(d-2h,0), or max(1-2h/d,0) normalized by zero-predictor risk d. Establish the
lower bound by skew projection plus Eckart-Young; give an explicit diagonal and
rank-h factorization attaining it in the relaxed family. Do not claim the finite
GELU candidate attains the relaxed optimum. At d384/h128 the lower bound is1/3.

CPU-only preflight: exact integer witness/factorization and squared errors for
(d,h)=(6,0),(6,1),(6,2),(6,3),(384,64),(384,128),(384,192),(384,256);
independent SVD/skew-projection verification; conditional-risk identities for
random projections and matrices in three seeds; pointwise diagonal optimality;
streamed Gaussian Monte Carlo checks against six estimated standard errors.
Implement the small candidate and verify input/parameter FP64 gradcheck, exact
parameter accounting and a full-output-rank constant-gate witness. No training,
no GPU work, no latency or memory-saving claim. Numerical checks validate the
implementation; the written proof establishes the population statement.

Freeze code/proof before running checks. Preserve H143 and maintained sources.
Record every witness, tolerance and random seed. Failure stops publication and
must be diagnosed without silently weakening the claim. A passed capacity
obstruction blocks allocating h128 as a general rank-repair solution; it does not
eliminate all output modulation on real FFN inputs. Separate any later hypothesis
from this shared-latent bottleneck and keep the broad research goal open.
