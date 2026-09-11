# H128: layout-aware native-buffer classifier

Previous H127 failed its exactness gate in five noncontiguous-input cases.
Preserve that failure. Change only the input-gradient product layout to match
PyTorch's mm_mat1_backward: column-major inputs use (W^T G^T)^T, otherwise
G W. Native NLL, aliased log-softmax and weight-gradient product remain fixed.
Scope: contiguous weights; 2D hidden or contiguous 3D hidden; FP32/FP64;
first derivatives, no autocast. No general backend/version guarantee.

This is a new candidate comparison, not a retry of the rejected H127 candidate.
Reuse its eight paired qualification cases and six directional derivatives
unchanged. Add one full-size contiguous FP32 classifier pair, hidden [8,512,384],
weight [4096,384], standard unit upstream; compare native loss and both
gradients bitwise and check input preservation. Totals: 18 qualification
backwards, 12 finite-difference forwards. The additional case directly covers
the target contiguous layout and vocabulary, not merely toy inputs.

If all qualify, reuse exactly H127/H123's planned four full-model cases and
four independent native replays: two corpora, same step-800 states and batches,
10 repeats with 3 warmup, reverse arm order, CPU checkpoint-input offload in
both arms, ordinary RMSNorm/resident gradients, original Adam state, 0 updates.
Maximum 62 backwards; 163,840 probe and 16,384 replay diagnostic targets;
qualifier targets separate. Four saved gradient artifacts; no trained states.
Model gradients and loss must match bitwise, states/batches must match prior
reference, all outputs finite, boundaries zero. Per corpus diagnostic peak
<=0.9x control; median event and synchronized wall <=1.15x; pinned peak <=128 MiB.
All phases, means, medians and variances retained. No training quality claim.

Stop on qualification failure. A resource/exactness pass earns a separate
complete-update comparison; it does not rescue H126 or satisfy the broad goal.
No parameters change. No maintained defaults change. No novelty claim:
[prior in-place CE](https://github.com/mgmalek/efficient_cross_entropy),
[CCE](https://arxiv.org/abs/2411.09009),
[PyTorch layout-aware backward](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/csrc/autograd/FunctionsManual.cpp).
