# Isolate name-supervision gradients: fresh512 diagnostic

Prospective 18 October 2026. The preceding name-anchor pilot improves familiar
binding but costs9.608699 language-accuracy points against its matched control.
Disabling copying restores only.013176 points. Test whether the direct NAME-loss
gradient into the shared LM backbone contributes to that failure.

One NEW `isolated_name` arm starts from the exact same fresh seed47 initial
reader tensors as the completed name-anchor arms, with fresh Adam. Do not resume,
extend, retrain or alter either completed comparator. The encoder, packing and
trained word decoder stay frozen.23,209,216 stored reader parameters, encoder
separate; one attention/KV position per64 old tokens; raw64 and inference unchanged.
No word-bit representation/integration or new production source is used here.

Use the EXACT immutable previous data, all512 LM/QA orders, labels, weights, LR,
FP32 AdamW.01, clipping1, effective16LM+16QA and micro8. Updates1..128:
.75LM+1.0NAME. Updates129..512:.75LM+.25answer+.25value-address+.25NAME.
The ONLY intervention is `query = h.detach()` for NAME scoring. Controller query
and gate gradients remain enabled; source-code gradients are not changed, though
the encoder is frozen. Answer/value/LM gradients remain ordinary. Do not detach
the NAME scalar, disable controller training or freeze backbone learning overall.
Indirect changes through the shared controller and LM copy mixture remain
possible: this test isolates direct NAME gradients, not every interaction.

Data are4096 QA examples/two new passes, not two original LM-corpus epochs.
Each arm/control exposes1872347LM labels,16896QA labels and8192NAME queries;
12660 answer/value labels are trained after warmup,8192NAME queries trained in
the new arm.128QA-held single-token values were known to the frozen encoder and
ordinary language corpus. The previous held-name/position/composed failures and
historical transfer remain necessary checks, even if familiar accuracy improves.

Before training: immutable parent data/initial/order/hash audit; CPU independent
joint-gradient derivation, direct NAME-backbone isolation/controller gradients,
microbatch parity, exact serialized8-step Adam resume across warm128->129,
future-code masking. Full-size zero-update GPU gradients and unchanged initial
weights/empty Adam; record paired body timings with Adam/I/O/eval/save excluded.
Pause after the first actual update, independently audit, resume that exact arm
to512, audit every order/label/LR/cursor/Adam step/frozen decoder and old hashes.
Never use the original/native/name-anchor training entrypoint to resume this run.
All new helpers/tests/protocol and old protected files stay immutable while live.

Evaluate all previous same-template held groups, positive/missing training fit,
free value+EOS, both questions/both swapped memories/all-four, source orders,
name/value selection, blocked/shuffled controls, full original7831-segment dev
(1859388labels), generator-only ablation, historical312. Compare with both saved
old arms without training them. Independently rederive all metrics, replay CPU
public generation, aggregate language and paired fixed-checkpoint document CIs.
No automatic extension, adoption or full-budget quality claim. Full goal still
requires fresh two-original-epoch training, broad facts/reasoning, matched true
1:1 reference and measured complete efficiency.
