# Mathematical scope of the multi-head architecture reference

See [architecture and budget](README.md). These are component bounds in exact
arithmetic, not claims of a new architecture or a trained-network guarantee.

## Normalized sigmoid routing

Let s_i = sigmoid(p_i), D = sum_i s_i + epsilon, r_i = s_i / D, with
epsilon > 0. Then 0 < sum_i r_i < 1 for finite logits. In particular,
weighted head aggregation cannot amplify the maximum subnetwork output norm:

    norm(sum_i r_i f_i) <= sum_i r_i norm(f_i) <= max_i norm(f_i).

The derivative of the routing weights is

    J_ij = r_i * (delta_ij - r_j) * (1 - s_j).

Write R = sum_i r_i <= 1. For column j,

    sum_i abs(J_ij) = r_j * (1 + R - 2*r_j) * (1 - s_j)
                    <= 2*r_j*(1-r_j) <= 1/2.

For row i, bounding each (1-s_j) by one gives

    sum_j abs(J_ij) <= r_i * (1 + R - 2*r_i)
                    <= 2*r_i*(1-r_i) <= 1/2.

Thus both induced 1-norm and infinity-norm are at most 1/2, and the induced
2-norm is at most sqrt(norm_1(J)*norm_inf(J)) <= 1/2. This is independent of
the number of subnetworks and retains the epsilon term. It gives an upper
bound on routing sensitivity, not a positive lower bound: saturated routing
can have arbitrarily small derivatives.

For input-dependent subnetwork outputs f_i(q), the full derivative also
contains sum_i r_i Df_i(q) and the output-weighted routing derivative. The
bound above therefore does not bound the entire FFN without additional
bounds on projections, subnetwork outputs and derivatives. Dense input/output
projections, attention, normalization and depth remain outside this claim.

## Initialization calibration at a small head width

Let sigma=0.02, d=H*dh, h_full=floor(8*d/3), and residual scale
c=1/sqrt(2*layers). The calibrated input mixer Q is orthogonal, so
||Qx||=||x|| exactly in real arithmetic. For isotropic input, each query
coordinate has unit variance. Gate/value entries have variance H*sigma^2,
so a private preactivation has expected variance

    dh * H * sigma^2 = d * sigma^2,

the full dense preactivation variance under the same assumptions. A zero router
starts every weight at 1/(E+2*epsilon), retaining the normalization term.

To motivate down initialization, suppose gated features have a common second
moment nu and centered subnetwork outputs are independent and isotropic.
A full SwiGLU output coordinate has approximate variance

    h_full * nu * sigma^2 * c^2.

Private down entries use variance E*h_full*sigma^2/de. Summing de features,
then uniformly mixing E subnetworks and applying the orthogonal output mixer
scaled by c, gives approximate variance

    (E/(E+2*epsilon))^2 * h_full * nu * sigma^2 * c^2.

Thus the extra mixing depth and average over subnetworks do not force the very
small initial output scale of the reference initialization. The epsilon factor
is close to one but is not dropped from the forward equation.

These are variance calculations under explicit distributional assumptions.
Finite head dimensions, within-head dependencies, learned routing and shared
inputs violate exact independence. Orthogonality gives an exact norm identity;
it does not make the nonlinear FFN an isometry. The measured Gaussian RMS ratio
1.11925 is consistent with approximate calibration, not exact moment equality.
No full-network Jacobian lower bound, convergence or trained-quality guarantee
follows. Uniform AdamW learning rates are an explicit local choice; changed
parameter scales also change optimization geometry.

The [H045 plan](../../../../multihead_integration_plan.md) freezes both
initializations before measurements. Existing standalone FP64 forward/derivative
checks remain, and integrated diagnostics report routing mass, normalized
routing entropy and saturation outside timing. Native execution materializes
intermediates and does not claim the paper's fused memory or speed behavior.

The [budget geometry](../../../../multihead_budget_geometry.md) proves that the
70% target forces the current two-subnetwork adaptation to heads of width8
or less at d=384. It also derives a one-subnetwork, no-router control with
exactly the BlockShuffle parameter budget and explains the finite-head
Gaussian RMS correction. These are analytical observations; the running
H046 architectures and initialization remain unchanged.


H046 now has a [completed validation result](../../../../multihead_screen_results.md):
both local initializations fail quality and memory gates. The separately frozen
[router-free control](headwise.md) reuses this folder and has its own equation,
initialization and exact count. No H046 configuration was changed.
