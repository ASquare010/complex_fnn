# H049: Close the lower boundary of the 800-step rate grid

Frozen after the completed H048 result and before any H049 training. H048's
eight higher-rate trials leave all four recipes selected at 0.0012, the lower
boundary of its primary grid. Its quality/memory pass does not establish an
interior rate optimum. Existing independent seeds for unchanged recipes are
retained; they are not rerun.

## Fixed comparison

Complete the four-rate grid 0.0006, 0.0012, 0.0024, 0.0048 for full SwiGLU,
full GELU, calibrated narrow and plain BlockShuffle. Reuse all twelve H048 cells
and the original H039 plain BlockShuffle 0.0006 cell. Run only the three missing
0.0006 cells, in order full SwiGLU, calibrated narrow, full GELU. Each has 800
steps, 1,638,400 sampled tokens and all 322,688 WikiText validation targets;
new budget is 4,915,200 sampled tokens. One fresh GPU worker at a time, at most
2400 seconds per cell. Official test stays unscored.

Change only global peak LR within each original 800-step recipe. Preserve
width 384, layers 8, attention heads 6, context 128, vocabulary 4096, batch 16,
seed 17, BF16, AdamW, clipping, data order, initialization, decay and group LR
calibration, recomputation and 80-step warmup/cosine schedule. The configurations
and optimizer caveats in H048 apply unchanged. No activation, decay, architecture,
schedule or post-result extra-rate search is introduced.

## Verification

Require H048 complete and all current computation unchanged from its 223-test
snapshot. Verify its source/plan, all twelve artifact/config/data/count/optimizer
contracts and final sampling states. Verify the original plain 0.0006 metrics
and checkpoint hashes against H039, with its history, sampling, archive and
settings. It was secondary context in H048 and is explicitly included in H049's
four-rate selection. Do not silently revise H048's three-rate result.

H048's four fresh initial validation reproductions remain applicable only while
NN, data and initialization sources remain unchanged. Each new training run's
initial full-validation NLL must independently match its same-recipe reference
within 1e-7. Reconstruct all validation targets, including the final nine-window
batch; check all eight immutable data hashes. Verify complete histories at
1/200/400/600/800, exact schedule, finite gradients/diagnostics, model and optimizer
metadata, checkpoints, source archives and final sampling RNG.

Preserve every failure. A recorded nonfinite clip-norm failure is an ineligible
numerical cell; an unexplained process/import failure stops for diagnosis.
No automatic retries or overwritten/duplicated completed trials. Any unchanged
continuation requires a written diagnosis and verified source/plan identity;
partial run directories require separate qualification.

## Selection and decisions

Select the lowest final NLL independently per recipe over all four rates, ties
toward lower rate. Report every same-rate comparison plus selected-recipe deltas,
all failures, allocation, clipping, descriptive timing and late learning curves.
Report interior versus lower/upper grid-boundary winners; this discrete bracket
is not proof of a global continuous optimum or of convergence.

Apply the same local gold gates as H048: >=70% fewer FFN weights, <=1% relative
NLL cost against BOTH selected full controls, strictly beating selected calibrated
narrow, allocated training peak <=1.1 times EACH selected full control, and finite
diagnostics. Also report the separate >=0.2% narrow margin. No significance claim.

If the lower controls remove the quality pass, qualify the prior selected result
and investigate before promoting longer runs. If all pass and selected recipes
are unchanged, retain the already completed three-seed evidence and freeze a
longer-budget test; do not repeat identical seeds. Changed selected rates require
independent-seed checks before replication claims. Remaining boundary winners
require further explicit tuning qualification. This is optimizer validation,
not a new nonlinear primitive or a breakthrough.
