# H124: compact RMSNorm backward plus gradient staging

**The combined storage candidate passes the two-corpus diagnostic gate.**
It saves 13.73–14.33% peak allocated VRAM relative to ordinary RMSNorm with
resident gradients, at 4.36–6.41% measured event-time overhead. Both arms
already offload checkpoint inputs and use chunked FP32 classifier loss.
This qualifies a separate complete-training test; it does not establish
language quality, parameter reduction, production throughput or a breakthrough.

| Corpus | RMSNorm | Gradients | Peak MiB | Saved | Event ratio | Wall ratio | Gate |
|---|---|---|---:|---:|---:|---:|---|
| wikitext2 | ordinary | resident | 281.150 | 0.00% | 1.0000x | 1.0000x | baseline |
| wikitext2 | compact | resident | 269.150 | 4.27% | 0.9482x | 0.9519x | FAIL: memory |
| wikitext2 | ordinary | staged | 254.552 | 9.46% | 1.0875x | 1.0808x | FAIL: memory |
| wikitext2 | compact | staged | 242.552 | 13.73% | 1.0641x | 1.0610x | PASS |
| tinystories | compact | staged | 236.373 | 14.33% | 1.0436x | 1.0486x | PASS |
| tinystories | ordinary | staged | 248.373 | 9.98% | 1.0903x | 1.0857x | FAIL: memory |
| tinystories | compact | resident | 263.901 | 4.35% | 0.9597x | 0.9649x | FAIL: memory |
| tinystories | ordinary | resident | 275.901 | 0.00% | 1.0000x | 1.0000x | baseline |

Compact RMSNorm removes exactly 12 MiB from the peak in each matched gradient
policy on each corpus. Alone it saves only 4.27–4.35%, failing the fixed 10%
requirement. Gradient staging alone still fails. Their combination passes
all memory, time and host-memory gates. Old H123 failures remain failed.

## Equation and implementation

For r=(mean(x²)+epsilon)^(-1/2), y=w*x*r, incoming gradient g:

    dw = sum_over_tokens(g*x*r)
    dx = r*(g*w) - x*r^3*mean(g*w*x)

The custom autograd function saves x, w and r. Its backward computes dw first,
then uses an owned dx buffer and addcmul_ for the correction. Saved inputs and
upstream gradients are never overwritten. This is the standard RMSNorm
function with an analytical first derivative; changed floating-point operation
order can still affect training. FP32 is the execution scope, FP64 the
mathematical qualification. BF16/FP16 and higher-order derivatives are not
supported by this prototype.

In exact arithmetic, unweighted RMS normalization has tangential Jacobian
eigenvalue r and radial eigenvalue epsilon*r³. Thus its spectral norm is at
most 1/sqrt(epsilon), and multiplication by w gives the bound
max(abs(w))/sqrt(epsilon). This local bound does not prevent exploding or
vanishing gradients through a whole network. No new expressivity is claimed.

## Verification and scope

Three FP64 shapes, including zero inputs, pass reference-autograd comparisons
at 1e-10 tolerances. Six joint finite-difference directions have maximum
absolute error 6.399e-07. Input/upstream hashes are unchanged by backward.
Full-model gradients match H121's independently audited references with maximum
global symmetric relative L2 2.663e-07, maximum tensor relative L2 5.258e-07,
and relative loss difference 0.000e+00. Saved gradients are finite. Staged
round trips are bitwise exact and every parameter hook runs once per backward.

All source weights, optimizer moments/counters and sampler states remain
unchanged. All ten GPU allocator boundaries are zero. Complete diagnostic
peaks include construction, warmup, forward/backward, gradient restoration,
serialization and state checks. Combined pinned-host allocation remains about
112.034 MiB. CUDA allocated/reserved and pinned active/cached statistics are
recorded separately. Driver/context memory is excluded from allocator figures.

The unchanged H123 run_case is reused through a temporary constructor adapter
and separate output roots; only normalization construction differs. The
maintained repository code and all 61 protected file hashes remain unchanged.

Budget: **86 backwards** (six qualification plus 80 full-model), twelve extra
finite-difference loss forwards, zero optimizer updates, 327,680 diagnostic
target evaluations and eight gradient artifacts. Three warmups and seven timed
repetitions per case; mean, median, sample variance and every repetition are
saved. Arm order reverses across corpora. No scientific case was repeated.
These are two fixed trained states, not independent training seeds.

## Next required evidence

Run a separately frozen complete-training comparison including the unchanged
AdamW update and global gradient clipping, all round-trip transfers, validation
and complete-job VRAM. Verify first updates and endpoint quality before
extending seeds or scale. The FP32-only wrapper needs an explicit ordinary
RMSNorm path for existing BF16 evaluation. Earlier H117 quality and H119
numerical failures remain unresolved; this short result does not erase them.

[Plan](compact_rmsnorm_plan.md), [implementation](../results/compact_rmsnorm_v1/norm.py),
[receipt](../results/compact_rmsnorm_v1/receipt.json).
Prior art includes [RMSNorm](https://arxiv.org/abs/1910.07467) and established
[normalization kernel optimization](https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/).
No algorithmic novelty or parameter reduction is claimed.
