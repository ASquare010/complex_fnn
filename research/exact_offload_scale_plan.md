# H161: larger-scale exact memory-helper probe

Prospective scope: one WikiText2 cache, three fresh seeds (401,409,419), six
30-update runs, ordinary versus unchanged buffered loss plus last-four-block
checkpoint-input offload. Order: ordinary/helper, helper/ordinary, ordinary/helper.
No retries to replace completed outcomes. Budget: 180 optimizer updates and six
initial backwards; stop on execution failure. No new data downloads.

Model: GELU, width512, hidden608, 12 layers, 8 heads, context512, vocab4096;
batch16. This scales the H156 narrow model's dimensions; both arms have exactly
the same parameters. FP32, TF32 off, four CPU threads, ordinary attention;
whole-block checkpointing in both arms. AdamW lr0.0006, beta(0.9,0.95), eps1e-8,
decay0.1 for matrix parameters and zero for vectors; clip norm1. No LR schedule.

Count construction, dataset GPU storage, initial gradient probe, initial/final
full validation and complete training updates in peak allocated/reserved GPU
memory. Validation uses the existing classifier-chunk loss with explicit FP32
and no autocast. Save first gradients and final model weights for native replay;
no optimizer checkpoint copies, so this is not a resumable long-training study.
Report pinned-host allocated and active peak, and process RSS (sampled telemetry
is approximate). Keep per-update CUDA forward/backward/optimizer and wall time,
loss and gradient norm. Passive nvidia-smi monitoring records device conditions.

Freeze code, plan and data hashes before any GPU execution. Each worker checks
source hashes and starts with zero allocated/reserved GPU memory. The first
probe uses the first training batch, then resets the sampler. Batch hashes and
initial model hashes must match per seed. Native final-state audit uses full
classifier CE instead of chunked evaluation, with relative score tolerance1e-6.
Compare saved initial full gradients with NumPy FP64: global relative L2<=1e-5,
max tensor relative L2<=1e-4, relative loss<=1e-6. No approximate-gradient limits.

Every seed must meet: whole-job allocated GPU ratio<=0.90; median complete CUDA
and wall ratios<=1.15; host allocated peak<=128MiB; final validation NLL ratio
<=1.01. After10 warmups, each run's two ten-update wall means must have max/min
<=1.15 and max wall/median<=5. All finite checks and all audits must pass. Report
all seeds, mean/median/sample variance. The short final-NLL test is only a smoke
gate, not a learned-quality claim; no architecture or parameter-reduction claim.
Do not change gates after seeing results. A pass warrants a later sustained
qualification, not automatic promotion. A failure limits this exact recipe at
this scale and must remain visible.
