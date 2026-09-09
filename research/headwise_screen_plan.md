# H047: Router-free headwise SwiGLU at the exact BlockShuffle budget

Frozen before implementation or any H047 validation score. H046 failed both
quality and memory gates. Test one simpler, prior-art-inspired control, not a
new activation family or a claimed new primitive.

## Hypothesis and construction

Dense input/output mixers surround a single private SwiGLU per head:
q = A x; f(x) = B concat_h[D_h(SiLU(U_h q_h) * V_h q_h)].
Use width 384, eight layers, 24 FFN heads, head width 16, private hidden width 48,
attention heads 6, context 128 and vocabulary 4096. No router, biases, extra
activation coefficients, checkpointing or compiler. FFN weights are exactly
2,801,664; total weights 9,099,648, matching BlockShuffle and calibrated narrow.
There are 1,152 private hidden features, half H046's 2,304. Removing routing,
doubling the head width and changing hidden count happen together: this is a
matched-budget recipe comparison, not an isolated router ablation.

Keep H045's calibrated rule with one subnetwork: A orthogonal; B orthogonal
scaled by 1/sqrt(2*layers); private gate/value normal std 0.02*sqrt(heads);
private down normal std 0.02*sqrt(full_hidden/private_hidden). Each parameter
uses the existing name-local seed convention. Do not apply the optional
finite-head moment correction from the budget note. Orthogonal mixer norm
preservation is exact at initialization; full-network gradient guarantees are
not claimed. No initialization or learning-rate redesign after scores.

## Qualification before training the screen

Save exact CPU state, logits, loss, all gradients and FLOP counts of all 30
registered variants before integration. Verify every old snapshot after edits.
Only add new branches to config, factory/initialization and matrix-work counting;
keep trainer, data, optimizer, existing architectures and diagnostics unchanged.
Test FP64 independent-loop forward/backward agreement, deterministic/common
initial tensors, parameter counts, mixer norms and diagnostic restoration.
Gaussian output RMS relative to full SwiGLU must lie in [0.5, 2], using the same
4096 width-384 rows from CPU generator seed 73 as H045, layer 0 / model seed 17.

One fresh GPU worker runs ten constant-LR 0.0006 AdamW updates on H045's exact
fixed training batch (2048 tokens, sampling seed 10017). Require initial BF16
versus FP32 logit relative L2 <=0.05, finite activations, every parameter gradient
finite and nonzero, and final pre-update loss below initial. Record all gradient
norms and clipping, allocation, batch hash, source/config and checkpoint.
These 20,480 repeated token exposures are not LM validation. Qualification failure
stops the screen for diagnosis. Existing import failures do not authorize blind
retries or overwriting any result.

## Frozen validation screen

Only after qualification and targeted tests pass, run the single calibrated
headwise recipe at peak rates 0.0003, 0.0006, 0.0012 in ascending order: three
200-step fresh GPU workers, each at most 1200 seconds. Use exactly H046's training
configuration and native BF16: batch 16, uniform AdamW, decay 0.1, beta (0.9,0.95),
eps 1e-8, clip 1, 10% warmup, cosine to 0.1 peak, seed 17. Evaluate all 322,688
WikiText-2 validation targets at initialization and steps 1/50/100/150/200.
New screen training tokens: 1,228,800. Official test stays unscored.

Reuse all twelve H032 controls and report all same-rate cells; H046's six cells
are secondary context with their separate initialization tuning effort disclosed.
Keep the 800-step stronger plain model separate from this 200-step comparison.
Verify all frozen reference metric/checkpoint/archive hashes, immutable data,
exact validation target sequence and final sampling state. Reproduce the four
original control initial validation NLLs in a fresh GPU worker within 1e-7.
Archive all new source, diagnostics, optimizer metadata, histories and failures.
Changed registry/factory/counting branches must be explicitly qualified;
computation shared with old trials must remain unchanged.

## Gates and decisions

Choose lowest final NLL among the three rates, breaking ties toward lower rate.
Mark a boundary winner as unbracketed. Quality-investigation gates are >=70%
fewer FFN weights, <=1% relative NLL cost against each selected full control,
and >=0.2% improvement over selected calibrated narrow. Full promotion also
requires training peak allocation <=1.1 times each selected full control and
finite diagnostics. These are engineering gates, not significance tests.

Quality failure eliminates this local recipe from longer training; preserve its
simplification and analysis. Quality pass with memory failure earns a separate
execution investigation. Both passing earn a separately frozen longer/multiseed
comparison; no automatic extension, novel activation additions or claim of
convergence. Training throughput is descriptive until synchronized contemporary
controls are measured. No outcome revises earlier frozen results.
