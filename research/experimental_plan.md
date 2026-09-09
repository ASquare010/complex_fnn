# Frozen initial protocol — 2026-09-06

## Target
At least 70% fewer FFN weights and <=1% relative validation NLL degradation
against full GELU and SwiGLU. Report total-model reduction separately.
A new primitive must beat a parameter-matched narrow conventional control.
Final acceptance requires >=3 paired seeds, equal tuning budgets, compute-matched
comparisons, ablations, a second corpus and larger models. Define catastrophic
efficiency regression as >20% lower inference throughput or >10% higher allocated
peak VRAM than the full reference. Screening cannot establish a breakthrough.

## Fixed Transformer
Decoder-only: 4 layers, d=192, 6 heads, context 128, RoPE, causal PyTorch SDPA,
pre-RMSNorm and final RMSNorm (eps=1e-5), tied embedding, no bias/dropout.
Initialize parameters using name-derived seeds so common tensors match exactly
across candidates. Linear weights N(0,.02^2), residual outputs divided by sqrt(2L).

TinyStories: immutable revision, up to 12,000 train and 1,000 validation stories,
exact normalized deduplication across splits. Train-only byte-level BPE up to
4096 vocabulary. Freeze token files and tokenizer hashes. Pack with EOS; allow
cross-story attention uniformly. Validation uses fixed nonoverlapping windows.
This small prefix subset is not representative evidence for broad language.

AdamW lr=6e-4, betas=(.9,.95), eps=1e-8, decay=.1 on matrices only; warmup 10%,
cosine decay to .1 peak. Batch=16. Gradient clip=1 with pre-clip norms and clip
frequency recorded. BF16 autocast; FP32 master weights, RMS statistics and loss.
Eager for every model, one GPU run at a time, independent model and sampling RNGs.

Screening: 200 steps = 409,600 tokens/model, seed 17; final checkpoint reporting.
This establishes operation, not convergence. If evidence warrants, compare
survivors and controls at 800 steps, seeds 17,29,43. Budgets form separate cohorts.
Report repeated exposure when training tokens exceed cache size.

## Analytic counts
Bias-free FFN: dense 2dh, SwiGLU 3dh, residual cubic 2dh+2G.
Total = Vd + L(4d² + FFN + 2d) + d.

| Model | h | G | FFN/block | Total at V=4096 | FFN reduction |
|---|---:|---:|---:|---:|---:|
| ReLU/GELU/SiLU full | 768 | — | 294912 | 2557632 | 0% |
| SwiGLU full | 512 | — | 294912 | 2557632 | 0% |
| GELU narrow | 192 | — | 73728 | 1672896 | 75% |
| Shared cubic | 192 | 1 | 73730 | 1672904 | 74.9993% |
| Grouped cubic | 192 | 8 | 73744 | 1672960 | 74.9946% |

Eight groups differ from the narrow control by .022% FFN parameters. A 75% FFN
reduction is only 34.59% total. Forward FFN matrix FLOPs/token: 4dh or 6dh;
report nonlinear work separately. FLOPs are not measured latency.

## First curve
Fix Bézier X controls at 1/3,2/3: X(t)=t, no inversion.
delta_j=.5*tanh(theta_gj), initially theta=0, t=sigmoid(x).
f_g(x)=GELU(x)+3t(1-t)[(1-t)delta_1+t*delta_2].
Starts exactly at GELU. Added derivative bounded by .75, not a whole-network
nonvanishing-gradient guarantee. See theory.md.

## Order and records
Validate CPU mathematics/counts/causality, CUDA backward and tiny-batch overfit,
then dense GELU/SwiGLU LM baselines before implementing candidates.
Next screen shared/grouped curves and seven function families. Inspect curves,
saturation, layer outputs/gradients, synchronized timings and allocated VRAM.
Promote useful signals only. No huge architecture search or custom kernels.

Each run: full config/seed, source-tree hash including uncommitted code, git
commit/dirty state, data/tokenizer hashes, software/GPU, token count, checkpoint,
loss curve, diagnostics, parameter/optimizer bytes, allocated peak VRAM, timings.
Inference timing means full-sequence forward throughput, not cached generation.
Time after warmup and CUDA synchronization. Exclude evaluation/checkpoint writes
from training timing; report wall time separately. Compare Pareto frontiers only
within matching data/config/budget/precision cohorts.
