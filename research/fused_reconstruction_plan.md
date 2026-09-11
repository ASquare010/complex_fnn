# H149: fuse reversible inverse and local derivatives

Previous turn: progress. H148 found memory savings but failed one timing limit
and a control stability check. Preserve that rejection. New execution hypothesis:
one tiled kernel can replace the inverse's many full-sized temporaries and eager
launches. Forward, matrices, shapes and learning recipe remain unchanged.

Kernel computes z=phi_inverse(y), pre=z-b, dz=g*phi'(z), and partial reductions
for theta/bias gradients. Fixed tiles64rows x32channels, four warps, no autotune,
no floating operation fusion. Reduction partials use PyTorch sum; two matrix
products remain native. Skip pre output at the earliest layer, where only the
input gradient is needed. First-order FP32 contiguous CUDA tensors only. The
squared-discriminant formula is qualified on abs(z)<=1e12, not all finite FP32.
No higher derivatives, mixed precision, compiler or cross-hardware guarantee.

Freeze before testing. GPU kernel checks at shapes(1,1),(7,33),(128,384),(2048,384),
seed217, signed logspace1e-12..1e12 plus zero, theta in[-1.5,1.5], rho.25.
Compare pre,dz,theta/bias gradients with independent CPU FP64 explicit formulas:
relative error<=5e-5 and max abs error/(1+abs(reference))<=1e-4, all finite.
Then full d384/depth8 native-versus-fused output/input/parameter VJP comparisons
at batches128/2048, seed217, relative<=1e-4 per family. Four autograd backwards
(two native,two fused), no optimizer updates. Stop on failed correctness checks.
Compile/setup wall time is recorded and excluded from subsequent update timing.

Resource screen:10 arms, batches128/2048, seeds223/239/251,12updates =>60runs,
720 optimizer/backward calls,61 allocator boundaries. Arms: full GELU/SwiGLU
eager/checkpoint, budget GELU checkpoint, learned checkpoint/native reconstruction/
fused reconstruction, fixed-shape native reconstruction, affine native reconstruction.
H148 measurement loop unchanged apart from import/root. Four warmups/eight timed
complete updates, FP32/TF32off, four CPU threads, default workspace; input gradients,
AdamW LR.003 betas(.9,.95), decay0,clip1. Rotate arm order. Count all fixed buffers.
No tuning/retries. The same batch/target construction; diagnostic losses only.

Primary fused candidate must satisfy all H148 gates versus BOTH checkpointed full
controls in EVERY fixture: parameters<=.8, allocated peak<=.9, CUDA/wall<=1.15,
and candidate/control half-window stability<=1.15. Show same-model native/checkpoint
and affine/fixed controls. Resource success only earns a separate quality study.

Independent CPU saved-state scorer, dataset/fixed-buffer regeneration, timing,
optimizer and finite-gradient audit adapted from H148; score tolerance1e-5.
Compare final execution states descriptively; no bitwise trajectory claim.
Keep kernel/check sources, compiler hash, library version, code receipts and logs.
Exclude generated compilation caches from evidence hashes, retain their directory.
No maintained module or default changes. Fusion is known prior art; no novelty claim.
