# Multi-head FFNs: parallel and router-free

Two registered variants implement the same published architecture with explicit
initialization controls: `multihead_swiglu` and `multihead_swiglu_calibrated`.
CPU/GPU qualifications pass. Both local parallel variants fail the completed
validation screen. The simpler `headwise_swiglu` control passes memory and fails
quality. These are comparators, not new primitives or reproduced paper results.

The architecture follows [FlashMHF equations 7 and 11-14](https://arxiv.org/html/2512.06989v1):
a dense input mixer forms heads, private SwiGLU subnetworks operate within each
head, normalized sigmoid routing combines them, and a dense output mixer joins
heads. The reference materializes intermediates; it does not implement the
paper's SRAM fusion algorithm.

    r_e = sigmoid(p_e) / (sum_j sigmoid(p_j) + epsilon)
    s_h = sum_e r_e * ((SiLU(q_h K_he) * (q_h U_he)) V_he)
    y = concat_h(s_h) W_out

The forward retains epsilon = 1e-6. Logarithmic normalization avoids premature
sigmoid underflow, accumulating routing in FP32 for lower-precision inputs and
preserving FP64 for reference checks. This choice is documented rather than
silently replacing the equation by softmax.

## Configuration and budget

`groups` is the FFN head count, independent of attention `heads`. `hidden` is
width per private subnetwork. Registered parallel variants use exactly two subnetworks;
the standalone class supports general positive counts. Default hidden is
`8 * ceil(width / (3 * groups))`; explicit hidden overrides it.

For model width d, E subnetworks and private hidden de:

    P = 2*d*d + 3*d*E*de + d*E.

At d=384, groups=48, head width=8, E=2 and de=24, there are 350,976 FFN weights
per layer, 2,807,808 across eight layers and 9,105,792 total. FFN weights fall
70.2474% versus the full control and are 0.2193% above BlockShuffle. Matrix work
is twice the weight count; routing, activations and aggregation are additional.
This small adaptation is much narrower than the paper's head dimensions.

## Initialization controls

The reference variant uses independent name-local normal std 0.02, with its
output mixer divided by sqrt(2*layers). The calibrated variant uses orthogonal
input/output mixers, the same output residual scale, larger private-weight
scales and a zero router that starts at uniform weights. The
[model notes](model.md) derive the variance approximation and its assumptions.
Standalone construction still uses the original normal initialization unless
`initialize(...)` is called explicitly.

On the frozen 4096-row Gaussian check, first-layer output RMS is:

| Full SwiGLU | Reference multi-head | Calibrated multi-head |
|---:|---:|---:|
| 0.01292124 | 0.00000183 | 0.01446211 |

The calibrated/full ratio is 1.11925, passing the frozen 0.5-to-2.0 range.
This is an initialization diagnostic, not evidence of better trained NLL.
All 28 previous registered variants retain exact tiny CPU parameters, outputs,
losses, gradients and matrix-work counts. Twelve comparator tests pass.

See the [frozen integration plan](../../../../multihead_integration_plan.md),
[CPU preflight](../../../../../results/verification/multihead_initial_v1.json) and
[comparator audit](../../../../current_comparator_audit.md). No validation or
test data was used to choose the initialization scales. The completed LM screens use separately frozen rates, budgets and acceptance
rules after GPU qualification.

The [completed GPU qualification](../../../../multihead_integration_results.md)
passes for both variants. Initial FP32/BF16 logit relative L2 is 0.00626/0.00808;
peak allocation is 832.10 MiB for each. All router gradients are finite and
nonzero over ten repeated-training-batch updates. The complete suite passes
213 tests. No validation/test target was scored, and no language-model quality
or flash-kernel performance advantage follows from these pipeline checks.


The [H046 validation screen](../../../../multihead_screen_results.md) fails
quality and memory promotion for both local initializations. The
[router-free headwise control](headwise.md), variant `headwise_swiglu`, is now
evaluated under [H047](../../../../headwise_screen_plan.md):
NLL 6.021328 at 638.37 MiB. It saves 23.3% of parallel training allocation but
fails quality. Read the [complete result](../../../../headwise_screen_results.md).
