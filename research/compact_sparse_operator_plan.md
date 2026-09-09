# H086 - Compact fixed 2:4 projection qualification

Freeze this plan before numerical execution. H085's learned offsets improve
synthetic fitting but lose to narrow GELU. Internal elementwise curves also leave
BlockShuffle's canonical block-Jacobian rank cap unchanged. Test a conventional
sparse comparator with wider hidden features and independently learned edges.
This is prior-art sparsity, not a novel architecture or a breakthrough claim.

The previous interrupted status turn made no scientific progress. H085 remains
complete with its separate administrative recovery receipt. No training worker
is live at the start of this allocation; the visible Python processes are the
editor formatter. Preserve all prior sources, plans, results and failure records.

## Operator and exact budget

For each output row and consecutive group of four inputs, choose two distinct
positions uniformly from the six unordered pairs, using a fixed CPU generator.
Only the selected values are Parameters; store positions as uint8 buffers.
There is no trainable dense shadow weight or topology update. Gaussian values
have variance 2/input_width; mask and value generators are independent. Seed 17
is the qualification seed. An input/output bias is not added.

Native execution scatters values into a temporary dense matrix and uses linear.
Its gradients are ordinary autograd. This implementation does dense matrix
multiplication; reduced learned counts do not establish sparse training FLOPs,
peak memory or speed. Report the temporary int64 indices and dense matrix as
well as persistent values, masks and prospective two Adam moment tensors.

At model width384, SwiGLU hidden608 and GELU hidden912 each contain350,208
learned values per FFN: respectively 3*384*608/2 and 2*384*912/2. Eight FFNs
contain2,801,664, a70.3125% reduction from9,437,184. Holding other model weights
fixed gives9,099,648 total versus15,735,168. Masks cost350,208 bytes per FFN;
they are not learned parameters but must be counted as storage. No model is
registered in the active tree during this stage.

## Scoped mathematical claims

The fixed-support insertion map S obeys ||S(v)-S(w)||_F=||v-w||_2, since every
value occupies a distinct matrix entry. Its adjoint gathers those entries from
the dense weight gradient. There is no multiplicative factor rescaling gauge
in this coordinate map. This is not a network conditioning guarantee.

For zero-mean independent weights of variance2/n, each row has n/2 values, so
E||row||^2=1 and E_W[(Wx)_i^2 | mask]=2/n times the sum of selected x_j^2.
Averaging over uniform masks gives ||x||^2/n. No claim about trained gradients
or exact finite-sample variance follows.

For one admissible support, H8 tensor I48 is a384-square matrix in the family:
choose each quartet's row-residue position plus its cyclic successor. The
second allowed weight may be zero. Its 64 canonical48-square blocks have
rank48 and its global rank is384. In this fixed partition, any matrix with
rank<=6 in each block has squared relative Frobenius error>=7/8. This witness
is NOT asserted for every random mask. A384-square sparse projection has
73,728 learned slots versus36,864 for square BlockShuffle, so this is not an
equal-count projection containment result. Equal FFN budgets change hidden
widths and do not prove nonlinear family inclusion or a task-loss lower bound.

## Frozen 28 checks, no training

1. One shape/count/storage check for both FFNs, and one invalid-shape/support
   determinism check. Masks must have two distinct indices in[0,3].
2. Four CPU FP64 projection cases (output,input)=(12,8),(64,48),(384,384),
   (608,384). Independently reconstruct the matrix with scalar indexing;
   verify insertion isometry and analytic input/value gradients at atol1e-11,
   rtol1e-10. Record initial row energy descriptively, not as a statistical gate.
3. Eight whole-FFN checks: GELU/SwiGLU crossed with CPU FP64, CPU FP32, CUDA
   FP32, CUDA BF16 autocast. Use noncontiguous inputs of shape(2,3,384).
   Compare outputs and input/selected-value gradients to independent dense
   masked references at (atol,rtol)=(1e-11,1e-10),(2e-6,2e-5),
   (2e-6,2e-5),(0.02,0.02), respectively. Ordinary execution and non-reentrant
   activation checkpointing must match exactly. State-dict reload must preserve
   values and masks exactly. Save every output/gradient pair for independent audit.
4. One central finite-difference check on a small FP64 projection: three value
   and two input coordinates, epsilon1e-6, atol1e-7, rtol1e-5, ten forwards.
   One exact rank witness check as above, using integer Gram identities.
5. Twelve inference capability cases: two FFNs, two explicit PyTorch backends
   (CUTLASS and cuSPARSELt), batches16/256/2048, BF16. Conservatively zero-pad
   each projection's input/output dimensions to multiples64, preserving actual
   learned counts; slice back to logical outputs. Record padded and packed
   storage. Do not retain a trainable padded copy. Backend conversion and first
   call errors explicitly reporting unavailable/not-compiled/unsupported
   backend operations are recorded UNAVAILABLE, without fallback or retry.
   Shape bugs, numerical mismatch, illegal accesses and unknown errors fail.
   Compare available sparse outputs with both padded and logical dense BF16
   references, atol0.03,rtol0.03, then time10 warmups and5 blocks of20 calls per
   mode (sparse, cached logical dense, native compact materialization), with CUDA
   synchronization around each block. Report median block time per call. These
   are standalone inference timings, not training or autoregressive serving.
   Keep all versions resident and report that combined peak explicitly; it is
   not a fair per-model peak-memory comparison. Measure compression once and
   separately. Do not tune algorithm IDs or use compile/custom kernels.

Use UV's existing environment, one hidden GPU qualification process, four CPU
threads, TF32 off. Collection must find exactly28 checks before execution.
There are zero optimizer updates, corpus accesses or training examples. Stop
after a failure, preserve all artifacts and do not automatically retry or relax
tolerances. Missing accelerated backends do not invalidate the native operator,
but prohibit a hardware speed claim for that backend.

## Evidence and next decision

Before execution pin H085's final recovery receipt, source and recovery support
archives, original valid metadata archive,135 scientific sources and74 frozen
plans. Add this plan, theory and three isolated source files. Write fsynced JSON
and tensor artifacts with immediate readback; freeze a source ZIP and software/
hardware/git provenance. Independently check every saved tensor pair, reconstruct
the sparse matrices and gradients, derive counts from saved values/buffers, and
verify the rank witness and source hashes. No old study is rerun.

A native pass earns only a separately frozen learning comparison with fresh
full/narrow/BlockShuffle controls, equal tuning effort and data. GPU capability
and timing inform its feasibility. Qualification cannot earn active model
promotion, language/scale/convergence evidence or completion of the research
goal. Update the readable report, current state, overview and candidate ledger.

## Primary prior work

- [RigL](https://arxiv.org/abs/1911.11134) updates sparse topology during training;
  this comparator keeps its sampled support fixed and is not a RigL replication.
- [PyTorch 2:4 training](https://pytorch.org/blog/accelerating-neural-network-training/)
  motivates hardware checks; published GPU speedups do not establish laptop results.
- [PyTorch sparse documentation](https://docs.pytorch.org/docs/stable/sparse)
  describes compressed tensor support. Installed backend constraints can be
  less restrictive than generic documentation; this study uses common64 padding.
- [Venom sparse pretraining](https://arxiv.org/abs/2602.06183) combines sparse and
  later dense steps. Its result does not establish a permanently compact
  parameter-only training method. No novelty is inferred from these sources.
