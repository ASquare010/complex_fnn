# H132: intermediate external cuBLAS workspace

Previous goal turn: progress. H131 established exact workspace accounting and
native gradient replay, but its 128-KiB setting failed runtime badly. Preserve
that result. Test explicit 8-MiB versus 32-MiB external workspaces while keeping
CUBLAS_WORKSPACE_CONFIG=:4096:8 unchanged in both fresh processes.

Use torch.backends.cuda.cublas_workspace_size(bytes), documented locally in
.venv/Lib/site-packages/torch/backends/cuda/__init__.py, before any model work.
Set the size explicitly in both arms; query and assert the returned value.
Keep strict deterministic algorithms, default SDPA, cuDNN deterministic on,
benchmark off, FP32 training, TF32 off and four CPU threads. No backend fallback.
This changes external workspace capacity, unlike H131's environment change.
CUDA allocator measurements do not include untracked library/driver allocations.

Reuse H131 worker, native replay, one-batch BF16 evaluation warmup, H123 probe
loop and thresholds unchanged. Two workspace policies x2 corpora x2 classifiers
(native or H128 buffer reuse, both with checkpoint-input offload) x10 backwards,
three warmup =80 probe backwards. Eight independent native replays give 88 total;
zero optimizer updates. Eight gradient artifacts; 327,680 probe and 32,768 replay
diagnostic targets, plus sixteen 4,096-target evaluation batches. Same original
step-800 states, batches, optimizer states and arm-order reversal across corpora.

Bitwise replay of every gradient/loss/evaluation result, finite and unchanged
states, data provenance and zero boundaries are mandatory. Low-workspace buffer
candidate must use <=0.9x peak and <=1.15x median event/wall time versus BOTH
8-MiB native and 32-MiB buffer controls, per corpus; host peak <=128 MiB.
Record all timings/mean/median/variance, no completed-case retries. Measure
workspace release after each case; test 2*(32-8)=48 MiB prediction separately.
Do not infer performance or quality in complete training from these probes.

A pass earns complete-update testing with ordinary execution controls. A failure
blocks that expansion. No threshold changes, H131 rescue, maintained-default
change, parameter reduction or novelty claim. Full research goal remains open.

[PyTorch backend API](https://docs.pytorch.org/docs/stable/backends),
[NVIDIA cuBLAS reproducibility](https://docs.nvidia.com/cuda/cublas/index.html).
