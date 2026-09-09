# H045: integrate and qualify the multi-head architecture comparator

Frozen after the corrected activation study and 207-test artifact audit, before
framework integration or GPU measurements for this comparator. H033 supplies an
equation-level CPU reference. This step qualifies that reference for future
controlled training; it is not a claim of reproducing published performance.

Source: [FlashMHF v1](https://arxiv.org/html/2512.06989v1), equations 7 and 11-14
and Appendix D. The architecture uses dense input/output mixers, private
head/subnetwork SwiGLU weights and normalized sigmoid routing. The appendix
reports initializer_range 0.02. Our much smaller dimensions, residual scaling,
data, schedules and native materialized execution are local adaptations.

## Two explicit initialization controls

Register `multihead_swiglu` and `multihead_swiglu_calibrated` in the existing
folder. Both share the unchanged reference forward. Use config.groups for FFN
head count, config.hidden for width per subnetwork, and exactly two subnetworks
in registered variants. Attention heads remain independent. Default hidden is
8*ceil(width/(3*groups)); explicit hidden overrides it. At width384, groups48,
hidden24 and eight layers, both have 2,807,808 FFN weights and 9,105,792 total,
70.2474% below the full FFN reference. Matrix work is twice the FFN weight count;
pointwise routing and aggregation costs are additional.

Reference initialization: independent name-local normal std0.02 for all FFN
weights, with output mixer divided by sqrt(2*layers), matching the repository's
residual-branch scaling convention. This is not claimed to be the paper's exact
unreleased implementation. Preserve existing standalone constructor behavior.

Calibrated initialization: orthogonal input mixer; orthogonal output mixer divided
by sqrt(2*layers); private gate/value std0.02*sqrt(H); private down std
0.02*sqrt(E*h_full/de), with h_full=floor(8*width/3). Start router at zero, giving
uniform weights 1/(E+2*epsilon) for E=2 and sigmoid(0)=1/2. Keep epsilon1e-6.
Use independent name-local generators. This approximates full SwiGLU's initial
output variance under isotropic inputs and independent centered subnetwork
outputs: the mixture divides variance by E, offset by the down-weight factor E.
Correlations, finite dimensions and nonlinear input moments limit exact equality.
Orthogonal mixing exactly preserves Euclidean norm before residual scaling.
This is an initialization control, not a new activation or expressivity primitive.

## Verification before any training screen

Preserve all 28 registered model state/logit/loss/gradient/FLOP snapshots exactly.
Check configuration/counts, standalone FP64 loop/derivative agreement, deterministic
name-local initialization, orthogonal mixer norms, uniform routing, finite router
learning signals and diagnostic restoration. Diagnostics must handle the new
module without assuming every FFN has a dense `up` module. Record routing mass,
normalized routing entropy, saturation and per-layer gradients outside timing.

Check initial output RMS on a fixed training-independent isotropic Gaussian
input: 4096 rows, width384, seed73, first FFN at layer0/seed17. Report all three
full SwiGLU/reference/calibrated values. The calibrated/full RMS ratio should be
between0.5 and2.0; a failure rejects this proposed calibration pending diagnosis.
Do not tune this ratio using validation data.

Then run one bounded GPU preflight, sequential fresh workers for each of the two
new variants. Use width384/layers8, batch16/context128, native BF16, ten AdamW
updates on one fixed frozen WikiText TRAINING batch, LR0.0006, uniform parameter
learning rates, decay0.1, betas(0.9,0.95), eps1e-8, clip1. This is a pipeline and
finite-gradient check, not an LM quality comparison. Compare initial FP32/BF16
sampled logits with relative L2<=0.05; require all losses/gradients finite and
record preclip norms, clipping, peak allocation, initial/final layer diagnostics.
Fresh worker timeout600s; no compiler, no validation/test selection, no retries
without explicit failure diagnosis. Preserve source/config/data/hash records.

No 200/800-step promotion follows from this integration alone. If checks pass,
freeze equal new learning-rate screen budgets and stronger controls separately.
Do not compare native comparator timing against a fused BlockShuffle as a fair
kernel-effort comparison. Published-scale conclusions and the broader research
goal remain open.
