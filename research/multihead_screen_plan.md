# H046: frozen multi-head validation screen

Written after H045 CPU/GPU qualification passes, before any multi-head validation
score or 200-step training run. This closes the missing trained-comparator screen;
it is not a reproduction of published-scale FlashMHF performance.

## Hypothesis, controls and fixed cost

Test whether the small multi-head architecture can retain language-model quality
at a 70.2474% FFN parameter reduction. The calibrated initialization is the primary
recipe, chosen from training-independent variance analysis. The normal
initialization is a separately reported control. Do not pool the two recipes'
best losses and pretend they received one recipe's tuning budget.

Each receives exactly peak rates 0.0003, 0.0006, 0.0012: six fresh 200-step runs,
409,600 sampled training tokens each, total 2,457,600 new tokens. Use the frozen
WikiText cache and train-only tokenizer, seed17, width384/layers8, attention
heads6, FFN heads48, two private subnetworks with hidden24, batch16/context128,
native BF16. Use the original full-SwiGLU screen training configuration with
only the FFN model fields changed: uniform global AdamW, decay0.1, betas(.9,.95),
eps1e-8, clip1, 10% warmup and cosine decay to0.1 peak. No compiler, gate
checkpointing or extra optimizer scaling. Evaluate all322,688 validation targets
at initialization and steps1/50/100/150/200. Official test remains unscored.

Run ascending rates, alternating recipe order by rate. Fresh GPU worker per
trial, 1200-second deadline. Finish all six unless an infrastructure or numerical
failure requires diagnosis. Preserve every result, diagnostic, checkpoint and
failure. No adaptive rate extension or change of initialization/optimizer.

Reuse ALL twelve original H032 control trials: full SwiGLU, full GELU, calibrated
narrow and plain BlockShuffle at the same three rates and 200-step schedule.
This gives three rate trials per independently reported recipe. Historical
architecture/init search effort is unequal and must be disclosed. The stronger
800-step plain recipe is not compared to a 200-step candidate as though token
budgets matched. H043/H044 already showed rate rankings can change with duration.

## Preflight and verification

Verify H045 completion and the 213-test source snapshot for computation files.
Check archived control metrics hashes, source archives, model/training settings,
optimizer metadata, data hashes and final sampling RNG. Preserve exact original
base FFN/data/optimizer code and training-loop AST; new factory/config/diagnostic
branches are documented, with H045's 28-model exact CPU snapshots as evidence.
In one fresh read-only GPU worker, reproduce each of the four original full-model
initial validation NLLs within1e-7 under current code. This is an initial-function
check, not a bitwise training-trajectory claim.

Reconstruct the exact validation target stream, including the last nine-window
batch. New workers must share all non-FFN initial parameters and the same final
sampling RNG as the archived200-step controls. Verify finite per-layer/route
statistics, checkpoint counts, histories and optimizer groups. Record trained
routing entropy/mass, saturation, clipping, allocation and descriptive throughput.
No paired native-versus-fused or cross-date speed claim.

## Selection and decisions

Select lowest final NLL separately for each initialization; exact ties use the
lower peak rate. Mark boundary winners and unbracketed optima. Report all same-rate
candidate/control NLL differences as well as the selected-recipe comparison.
A selected-rate advantage is not an isolated initialization or architecture cause.

Quality-investigation gate: at least70% fewer FFN weights, within1% relative NLL
of BOTH selected full controls, and at least0.2% below selected calibrated narrow.
Full candidate promotion additionally requires <=1.1 times the measured peak
allocation of EACH selected full control and finite diagnostics throughout.
These thresholds are engineering gates, not statistical significance tests.

Report the normal-versus-calibrated contrast at each rate. If quality passes but
memory fails, retain a quality lead and require a separately frozen execution
investigation; do not promote800-step candidate training automatically. If both
quality and memory pass, freeze a longer comparison with all required controls
and matched rates before further training. If quality fails, keep the adaptation
as negative/inconclusive local evidence; it does not refute the published
architecture at larger heads, different optimization or longer budgets.

One seed and200 steps cannot establish convergence, broad superiority, a new
primitive or full-network gradient stability. No outcome silently changes H041,
H043/H044, or any other earlier frozen decision.
