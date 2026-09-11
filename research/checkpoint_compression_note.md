# Follow-up lead: compressed checkpoint storage

This is a literature/design note, not an executed experiment or a change to H156.
H156 remains frozen and must finish its full native audit before a new GPU test.

## Why inspect this route

H142's classifier-buffer reuse plus offload of four checkpoint inputs saved about
24% allocated VRAM at batch16. CPU transfer is a possible runtime cost, not an
established explanation for every measured slowdown. Checkpoint input compression
on GPU could trade controlled numerical error for less transfer traffic.

For eight equal d-wide FP32 checkpoint inputs, keeping four on GPU and offloading
four retains 4*4=16 bytes per feature on GPU. Keeping all eight as FP16 also costs
8*2=16 bytes per feature, before temporaries and allocator effects. This is a
simple storage comparison, not a prediction of equal peak VRAM or faster training.
FP32 unpacking, casts and tensor lifetimes must be measured. Classifier buffering
would remain identical in both memory candidates.

## Relevant prior work

- [ActNN, ICML 2021](https://arxiv.org/abs/2104.14129) studies stochastic low-bit
  activation storage and gradient variance. Its reported savings do not predict
  this checkpointed Transformer workload.
- [GACT, ICML 2022](https://proceedings.mlr.press/v162/liu22v/liu22v.pdf) applies
  compression through saved-tensor hooks and allocates precision according to
  estimated gradient sensitivity. It is a relevant comparator and precedent;
  generic saved-tensor compression is not novel.

## A bounded test to design after H156

Start with FP16 storage of whole-block checkpoint inputs only, reconstructing
FP32 for the existing backward recomputation. Exclude weights, optimizer state,
classifier/log-softmax tensors and integer tensors. Compare ordinary checkpointing,
qualified four-block offload, and compressed storage with the same buffer loss.
Use an explicit native identity-pack control to catch hook/lifetime mistakes.

First inspect actual checkpoint input range and quantization error on existing
saved states. Compare complete parameter gradients against the native path,
including per-layer errors and directions. Check small and large activation
scales; stop on overflow or nonfinite values. Then measure whole-job peak memory,
complete-update timing and host memory in temporally paired runs, with conversion
and recomputation included. Only a viable numerical/resource result earns a fresh
multi-seed quality run. Retain all failures.

Unbiased rounding of inputs does not imply unbiased gradients through arbitrary
nonlinear recomputation. This is approximate training, so the exact-path gradient
caps from H156 cannot simply be reinterpreted as passing. Establish prospective
approximation and quality criteria, include fixed-rate/seed controls, and document
what is lost. No guarantee of stable gradients or preserved long-run quality is
claimed. A successful test would improve execution memory, not prove an efficient
new activation or satisfy the separate architectural parameter-reduction target.
