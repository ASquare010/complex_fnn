# H127: native-buffer classifier qualification fails

**FAIL exactness; no full-model resource or training claim.** Sixteen operator
backwards and twelve finite-difference forwards ran. The strict gate stopped
all forty proposed model probes and all four proposed model replays. No
optimizer updates, training targets, model gradient artifacts or run retries.

| Precision | (tokens, vocabulary, width) | Exact loss | Exact input gradient | Exact weight gradient | Input max absolute difference |
|---|---|---|---|---|---:|
| torch.float32 | (3, 7, 5) | True | True | True | 0.000e+00 |
| torch.float32 | (9, 1024, 32) | True | False | True | 1.490e-08 |
| torch.float32 | (11, 4096, 48) | True | False | True | 9.313e-09 |
| torch.float32 | (5, 4097, 16) | True | False | True | 4.768e-07 |
| torch.float64 | (3, 7, 5) | True | True | True | 0.000e+00 |
| torch.float64 | (9, 1024, 32) | True | False | True | 1.180e-16 |
| torch.float64 | (11, 4096, 48) | True | False | True | 2.082e-17 |
| torch.float64 | (5, 4097, 16) | True | True | True | 0.000e+00 |

All outputs/gradients were finite, all caller inputs/upstream gradients were
unchanged, and all six directional derivative checks passed (maximum absolute
error 4.410e-10). The numerical derivative appears correct within the tested
tolerances, but five of eight cases fail the deliberately stronger bitwise
requirement. Every hidden input has column-major strides; this study does
not establish whether row-major/full-model cases differ.

The candidate owns its logits buffer and uses native log-softmax/NLL kernels
with aliased output storage. It then computes dH=G W directly. PyTorch's
[mm_mat1_backward implementation](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/csrc/autograd/FunctionsManual.cpp)
selects (W^T G^T)^T for column-major inputs. These are algebraically identical,
but matrix-product dispatch can change floating-point results. This provides
a specific layout hypothesis; it is not a demonstrated causal attribution yet.
A separate prospectively frozen layout-aware variant is the next experiment.
The failed original implementation and thresholds remain unchanged.

In-place cross-entropy already has [prior art](https://github.com/mgmalek/efficient_cross_entropy).
[Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) avoids materializing the full
logit matrix, unlike this native-buffer prototype. No novelty, parameter
reduction, VRAM improvement or long-run quality result is established here.

Source/input hashes, all 61 maintained files and two zero GPU allocator
boundaries verify. [Protocol](native_buffer_loss_plan.md),
[raw qualification](../results/native_buffer_loss_v1/qualification.json),
[receipt](../results/native_buffer_loss_v1/receipt.json).
The broader research goal remains open.
