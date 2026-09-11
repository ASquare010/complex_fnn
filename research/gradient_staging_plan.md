# H123: stage completed gradients on the host

Previous turn: progress. H122 located a late requested-payload peak after
checkpoint-input offload, but failed exact allocated-peak attribution.
Test a concrete storage change using actual allocator counters.

Use H121's two seed-101, update-800 chunked FP32 fixtures. Both arms offload
checkpoint inputs. Resident arm retains normal CUDA parameter gradients.
Staged arm uses post-accumulation leaf hooks to copy each completed gradient
into a preallocated pinned CPU buffer and set parameter.grad=None. Require
exactly one hook per parameter per backward, including the tied embedding.
After backward synchronize, restore every gradient to CUDA before proceeding.
Include the restoration in latency and peak memory; no optimizer change.
This prototype supports one backward per batch, not gradient accumulation.

Budget: two FP64 toy backwards (shared weights used twice, resident/staged)
then four cases x ten backwards (first three warmup, last seven timed):
42 backwards total, zero updates, 163,840 full-model diagnostic targets.
Alternate mode order across corpora. One GPU process, UV-managed runtime,
four CPU threads, TF32 off. No training or quality extension in this study.

Require all: toy agreement at atol/rtol 1e-12; restored gradients bitwise equal
to staged buffers; raw gradients match audited H121 reference at global
symmetric L2 <=1e-5, max tensor <=1e-4, relative loss <=1e-6; unchanged source
weights/moments/sampler; finite saved gradients; zero GPU boundaries.

For each corpus require complete-case allocated peak <=90% of resident,
median forward/backward/restore event-sum and synchronized wall <=1.15x,
and peak pinned-host allocated bytes <=128 MiB. Include construction,
warmup, all copies, restoration, serialization and state verification before
any peak reset. Record pinned active and cached allocation separately.

A pass earns a separately budgeted complete-training test with the original
AdamW/global clipping, not a breakthrough claim. Preserve failed gates and
logs. No repeating completed cases or changing thresholds after observation.
This is established gradient-offload machinery, not a novel activation:
https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.register_post_accumulate_grad_hook.html
