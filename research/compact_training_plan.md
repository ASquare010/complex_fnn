# H125: complete-update validation of compact normalization and gradient staging

Previous turn: progress. H124's combined candidate passed its diagnostic gate.
Test full updates before longer training: WikiText/TinyStories seed101 source
step800, chunked FP32 classifier, two arms, 30 updates each. Both arms offload
checkpoint inputs. Baseline uses ordinary RMSNorm/resident gradients; candidate
uses compact RMSNorm/staged gradients, restoring all gradients before the
unchanged global clipping and default AdamW update. Use ordinary RMSNorm during
no-grad/evaluation, including BF16 validation. Preserve parameter/state names.

Reuse H120 run_case, numerical Adam audit and native scoring/batch audit without
editing them. A scoped adapter intercepts the model factory, training loss and
Tensor.backward; only the current training loss triggers restoration. Restore
all patched methods and release model references before allocator boundaries.
One GPU process, existing UV runtime, FP32/TF32-off, four CPU threads.

Qualification: two tiny Transformer backwards, scaled loss to activate clipping,
plus four no-grad eval forwards (FP32/BF16 across both arms). Require gradient
agreement <=1e-5 global/1e-4 max tensor, independent FP64 clip error <=1e-6,
exact staged restore, and identical eval logits. Stop training if this fails.
No toy optimizer steps. Then 120 training updates/backwards =491,520 targets.
Total122 backwards, eight full study scores, four independent native scores,
120 audit batch reconstructions and12 tensor artifacts. Audit has zero backwards.

Require each corpus: complete-job allocated peak <=90% baseline, median
summed-event and synchronized wall <=1.15x, timing split-half ratio <=1.15,
pinned allocated peak <=128 MiB, final NLL <=1.01x baseline; all values finite.
Include initialization, warmup, evaluation, all transfers, clipping, optimizer,
serialization and checks before any peak reset. Use last20 of30 timings.

Audit first raw gradients, clipping, actual step801 parameters/moments against
NumPy AdamW equations (parameter global <=1e-6 / max absolute <=2e-7, moments
<=1e-6). Verify all source/first/final states, steps800/801/830, batches/samplers
and full native-vs-streamed score agreement <=1e-6. All GPU boundaries zero.
Do not change optimizer settings or accept a mean that hides a failing corpus.

This is short continuation evidence, not fresh-run or multiseed quality proof.
A pass earns a separately frozen longer training study; H117/H119 failures,
parameter-efficiency and broad breakthrough requirements remain unresolved.
No completed cases repeated; failures preserved. No maintained default change.
