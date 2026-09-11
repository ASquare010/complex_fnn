# H123: completed-gradient host staging

**The fixed 10% memory gate fails on both corpora.** Savings of 9.4605% and
9.9778% must not be rounded into passes. Gradient and transfer-cost gates pass.

| Corpus | Resident peak MiB | Staged peak MiB | Saved | Event-time ratio | Wall-time ratio |
|---|---:|---:|---:|---:|---:|
| wikitext2 | 281.150 | 254.552 | 9.4605% | 1.1134x | 1.1019x |
| tinystories | 275.901 | 248.373 | 9.9778% | 1.1305x | 1.1194x |

Both arms already offload checkpoint inputs and use chunked FP32 classifier
loss. The staged arm additionally moves each completed parameter gradient to
pinned CPU storage and clears its CUDA .grad field. It restores **every**
gradient to CUDA after backward. Restoration is included in both memory and
timing, so this is not a one-way transfer comparison. No optimizer step runs.

Peak CUDA allocation falls by 26.60 MiB on WikiText and 27.53 MiB on TinyStories.
Additional pinned gradient payload is 34.71 MiB. Combined pinned-host allocator
peak reaches 112.034 MiB, including the checkpoint-input buffers and rounded
cached allocations. Active and allocated host statistics are recorded separately.
These are process allocator figures, not system-wide VRAM or total RAM use.

The post-accumulation hook runs exactly once for each of 50 parameters per
backward, including tied embedding weights. Staged CUDA .grad payload is zero
at the synchronized backward boundary; all gradients are subsequently restored.
Every final restored gradient is bitwise equal to its staged CPU buffer.
Against H121's independently audited reference, maximum global symmetric
relative L2 is 1.083e-07, maximum tensor relative L2 2.239e-07, and maximum
relative loss difference 0.000e+00. The shared-weight FP64 toy passes 1e-12
checks. All saved gradients are finite; source weights, Adam moments/counters
and sampler state remain unchanged.

Budget: two toy backwards plus four cases with ten backwards each = **42
backwards, zero updates, 163,840 diagnostic target evaluations**. The last seven
repetitions are timed after three warmups. Mean, median, sample variance and
all repetitions are saved. This is a short transfer-cost screen, not sustained
training throughput or language-quality evidence. No case was repeated.
All six GPU boundaries are zero; all 61 maintained file hashes are unchanged.

The prototype only handles one backward per batch. It intentionally rejects
multiple post-accumulation callbacks; gradient accumulation, distributed
training and concurrent streams are not qualified. The original AdamW/global
clipping recipe would still need complete-training validation. That follow-up
was not earned under this study's fixed gate. H121/H122 failures remain intact.

**Next hypothesis:** reduce a remaining temporary allocation instead of making
this transfer path more complex. For example, a memory-efficient normalization
backward could be tested against the maintained RMSNorm under explicit gradient
and end-to-end memory gates. This is a new hypothesis, not a rescue of H123.

[Plan](gradient_staging_plan.md), [source](../results/gradient_staging_v1/staging.py),
[evidence receipt](../results/gradient_staging_v1/receipt.json).
The existing [PyTorch post-accumulation hook](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.register_post_accumulate_grad_hook.html)
supports changing a leaf parameter's .grad field. Related gradient-lifetime
work appears in the official [optimizer-in-backward tutorial](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html).
No algorithmic novelty, parameter reduction or breakthrough is claimed.
