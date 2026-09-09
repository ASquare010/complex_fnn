# Efficient-FFN comparator: FlashMHF architecture reference

The BlockShuffle measurements do not establish superiority over all efficient
FFNs. [FlashMHF, sections 3.1-3.2](https://arxiv.org/html/2512.06989v1) describes
dense input/output mixing, private head-specific SwiGLU subnetworks, sigmoid
routing with epsilon normalization, and an SRAM algorithm. Its hardware and
training settings differ substantially from this repository's experiments.

## Small-budget adaptation

The bias-free architecture count is 2*d*d + 3*d*E*de + d*E, where E is
subnetworks per head and de their hidden width. With d=384, H=48, head width 8,
E=2 and de=24, the count is 350,976 per layer, or 2,807,808 over eight layers.
That is 70.2474% below the 9,437,184-weight full FFN reference and 0.2193% above
BlockShuffle's 2,801,664 weights. Total model weights would be 9,105,792 with
the same non-FFN components. There are 2,304 intermediate features per token.
These analytic counts are now verified against all six trained H046 checkpoints.

## Completed qualification (now archived)

Two variants implemented the same reference forward with normal and
variance-calibrated initialization. They were subsequently screened and archived;
neither remains registered in the active factory. The latter uses orthogonal mixers and
head-aware private scales. Both retain epsilon normalization and the
[component routing bound](archive/retired/src/multihead_ffn/model.md). No flash kernel or
claim of the paper's trained performance is made; local validation is now measured below.

The [frozen H045 integration](multihead_integration_plan.md) qualifies the
small-budget adaptation before any new training screen. All 28 previous model
snapshots remain exact. Twelve comparator tests pass, including FP64 loop and
derivative agreement, counts, deterministic initialization, orthogonal norms,
router learning signals and diagnostic restoration. Gaussian first-FFN RMS is
0.012921 for full SwiGLU, 0.00000183 for the reference and 0.014462 for calibrated
multi-head. The calibrated/full ratio 1.11925 passes the frozen interval.
Both bounded GPU qualifications now pass, at 832.10 MiB peak each; no
validation/test data informed these scales. The [result](multihead_integration_results.md)
verifies finite gradients and source/checkpoint records, with no LM quality claim.

The paper reports initializer_range 0.02 in Appendix D. Its dimensions, training
budgets and hardware differ from this small-model adaptation. Our explicit
residual scaling and calibration must not be described as its exact recipe.
H046 below records the completed local training comparison and its optimizer
treatment. Any paper-scale reproduction would need its own matched tuning budget. Native timing cannot establish superiority
or inferiority to the paper's flash implementation.

Learnable-activation prior art is covered separately in the
[domain guide](learnable_activation_domain.md). The corrected local activation
study finds no material same-rate promotion. The completed small-budget screens
below do not replace paper-scale reproduction, convergence or novelty requirements.

The [budget geometry](multihead_budget_geometry.md) proves that the
70% target forces the current two-subnetwork adaptation to heads of width8
or less at d=384. It also derives a one-subnetwork, no-router control with
exactly the BlockShuffle parameter budget and explains the finite-head
Gaussian RMS correction. These analytical observations did not alter the frozen H046 architectures
or initialization; its completed outcome is reported below.


## Completed local screen

[H046](multihead_screen_results.md) gives best NLL 6.331841 (normal) and
6.027381 (calibrated), versus 5.912434 for calibrated narrow. Both fail quality
and memory gates at 832.58 MiB. All three rates per initialization, failures,
routing diagnostics and control differences are retained. These are
200-step, seed-17, tiny-head results; neither optimum is bracketed.
[H047](headwise_screen_results.md) completed the separately specified router-free
headwise simplification at exactly the BlockShuffle budget. Its memory gates
pass but quality fails. A [structured-mixer budget proposal](structured_headwise_budget.md)
proposed larger head-local subnetworks by restricting the mixer family. Later
[overcomplete structured-headwise screens](overcomplete_screen_results.md) tested
a distinct qualified allocation and failed quality. Historical budget proposals
do not automatically authorize reviving the retired family.
