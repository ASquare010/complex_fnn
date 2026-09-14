# H164: balanced BTT shape learning screen

H163 qualifies the exact memory helper only for specific workloads. Architecture
compression is still open. This screen tests an existing BTT shape-choice lead,
not a novel activation. The previous goal turn made progress by sealing H163.

Use a64->152->64 GELU function, with bias after each projection. Rank1 BTT has
right cores(m1,m0,n0), left cores(n0,m1,n1), learned scalar gains and the official
RMS ceiling, Gaussian muP core scale sqrt(min(fanin,fanout))/fanin and core LR
multiplier original_input/(2*core_fanin). No dense matrix is formed in training.
Batched GEMMs are plain PyTorch, not the paper's optimized kernels. Dense uses
fan-in Gaussian initialization; local BlockShuffle uses its unchanged orthogonal
initialization and the analogous per-core LR multipliers. All biases start zero.
Gains/biases use base LR. This is a scoped baseline adaptation, not replication of
the paper's whole training recipe or performance claims.

For152, greedy factors are4*38 and closest factors8*19;64 is8*8.
Greedy up has intermediate32, so rank<=32; balanced up has intermediate64 and can
be full column rank. Balanced down intermediate152 versus greedy304. Count:
wide19672, balanced4380, greedy5340, BlockShuffle3672 parameters including biases
and BTT gains. Matched dense widths33/40/27 have4321/5224/3547 parameters.
Balanced up/down must be checked against independently materialized matrices,
and normalized-core/input/gain gradients checked with CPU float64 gradcheck.

Seven arms: wide, narrow33, narrow40, narrow27, BlockShuffle8groups, greedy BTT,
balanced BTT. Two tasks: independent Gaussian dense GELU teacher64->152->64,
and coordinate cyclic products x_i*x_(i+1). Three independently generated data,
teacher and optimization seeds(431,443,457).4096train/1024validation/1024report
examples per task; disjoint iidGaussian inputs. Center/scale outputs with training
mean and one global training RMS only. These are synthetic screens, not real-data
or universal-complex-pattern claims. Do not use report data for selection.

Each arm gets identical two-rate search(.001,.003),600AdamW updates, beta(.9,.999),
zero decay, clip norm1, batch256 sampled using a common per-seed CPU generator.
FP32, TF32off,4CPUthreads, one GPU model at a time.84fits/50,400learning updates.
Select rate and step among200/400/600 by validation only; retain all final/selected
states and metrics. Equal search size, not equal total FLOPs. Stop on execution
failure; no replacement of completed results. Original failures remain visible.

Before training, freeze code/plan/check/data-generation hashes and check gradients.
Save data and CPU selected states, under100MiB expected. Independent CPU float64
native-GELU evaluation verifies each selected state's validation/report losses.
No model training in the audit. Require reportwide normalizedMSE<=.25 teacher and
<=.50 product on every seed, otherwise that task is inconclusive for promotion.

For each task/seed, balanced must have reportMSE<=1.05wide and<=.95narrow33,
>=25%fewer parameters, plus measured local training allocation<=.90wide and
median complete CUDA and wall update<=1.15wide. Resource profiles: same selected model restored
for each of20 completeAdamW updates on one fixed8192example batch (5warmups),
resident batch/model/gradients/moments included, no dataset on GPU. Resource
updates are separate from learned checkpoints and cannot affect report scores.
Seven selected models per task/seed; total42selected profiles,840extra updates;51,240updates
including profiling. Report absolute GPU allocation/reservation and times.
This is local FNN job memory, not a Transformer or inference claim. Record all
profiling outcomes; a slow plain-PyTorch implementation does not disprove a faster
kernel exists, but earns no performance promotion here.

Shape hypothesis separate from promotion: balanced/greedy mean reportMSE<=.95 on
both tasks and no seed >1.05, evaluated only on qualified tasks. Preserve every
seed, mean/median/sample variance, learning curves, gradients and failed gates.
A shape improvement alone earns no claim of beating dense. No gate changes.

[BTT prior art](https://arxiv.org/abs/2406.06248),
[retained official-code comparison](btt_official_comparison.md).
