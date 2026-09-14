# Explicit training-memory helpers

```python
from src.core.training_memory import buffer_model_loss, offload_checkpoint_inputs

model.set_recompute_scope("block")
model.train()
with offload_checkpoint_inputs(model, last_n_blocks=4):
    optimizer.zero_grad(set_to_none=True)
    loss = buffer_model_loss(model, tokens, targets)
    loss.backward()
    # Apply your unchanged clipping/optimizer policy here.
    optimizer.step()
```

This API does not alter ordinary model/trainer defaults or register new parameters.
Use the context around forward and backward. It restores instance forward methods,
even on exceptions. Do not concurrently use the same model; existing forward
overrides and nested wrapping of the same blocks are rejected before mutation.
The selected blocks must already use whole-block recomputation. CPU, evaluation
and no-grad forwards bypass CPU transfer hooks. The final four of eight blocks (H156) and four of twelve blocks (H163) were tested; other counts/scales need their own measurements.

`buffer_cross_entropy(hidden, weight, targets)` is the standalone tied/untied
classifier primitive. It owns its logits and backward temporary, reusing native
log-softmax buffers without overwriting caller tensors. It accepts matching
FP32/FP64 tensors with autocast disabled, 2D hidden or contiguous3D hidden,
contiguous2D classifier weights, and int64 targets matching the hidden leading
shape on the same device. Classes must be valid native CE indices or-100.
Reduction is mean; ignore_index=-100. All-ignored labels keep native NaN-loss
semantics. No bias, class weights, label smoothing, higher derivatives,
compilation/torch.func or graph-capture support is claimed. It uses private ATen
operators; rerun numerical and resource qualification after PyTorch upgrades.

`buffer_model_loss` supports the ordinary repository Transformer with its tied
embedding classifier. It bypasses the top-level Transformer.forward hooks;
embedding, block and norm hooks still execute. FP64 scalar tests do not establish
FP64 Transformer training equivalence (the model's RMSNorm computes in FP32).

[H140](interleaved_training_results.md) measured18.7–19.0% lower allocated GPU peak
with1.3–3.1% complete-update overhead in balanced short continuations across two
corpora and three saved seeds. Parameters were unchanged. This result is scoped
to a9.1M-parameter FP32 model, batch8/context512, ordinary attention and eight
checkpointed blocks. Offload consumes pinned host RAM. It is an established memory
technique, not a new activation or parameter-reduction result.

[H138](partial_offload_long_results.md) remains failed on runtime/stability despite
passing quality. That failure remains unchanged. Later studies establish only their specific workloads, with no breakthrough or unrelated-domain claim. [H141 integration verification](training_memory_integration_results.md)
records maintained-versus-frozen checks and their exact scope.

[H156](batch_scale_long_results.md) passed fresh 800-update training at batch16
on the 9.1M model across both corpora and three seeds: about24% lower allocated
GPU peak, with 2.9–12.6% slower complete updates.

[H163](exact_offload_long_scale_results.md) passed six fresh 800-update runs on a
22.2M-parameter model (width512, hidden608, twelve blocks) on WikiText2. Batch16,
context512, FP32/TF32off, ordinary attention and unchanged model parameters.
Allocated GPU peak fell925.58→758.08MiB (18.10%); complete updates were1.2–5.9%
slower; the largest validation NLL increase was0.24%. Peak resident host memory
increased63.82–66.11MiB. These are three-seed results under the fixed documented
optimizer and hardware;800 updates do not establish terminal convergence.