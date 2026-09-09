# H012: two-factor BlockShuffle - frozen first screen

The earlier grouped sandwich is not a full BlockShuffle projection. Use the
established two-factor map from
[Wei et al., section 2.1](https://arxiv.org/html/2406.16450v1#S2.SS1):
W = P_out^-1 B2 P_mid B1, intermediate width k=min(N,M), G blocks.
Each projection has k*(N+M)/G weights. In a gated FFN with d=192,h=832,G=8,
three projections use 73728 weights per layer: 294912 over four layers,
75% below the full reference, exactly matching narrow SwiGLU h128 in matrix
weights and matrix FLOPs. It has more hidden activations and six grouped GEMMs
per FFN. Permutations and activations cost time and memory.

Orthogonal factor initialization is motivated by section B.1 of the same paper.
Use semi-orthogonal blocks with product gain s=.02*sqrt(max(N,M)); divide s
by sqrt(2L) for the residual-output projection. Split gain equally across the
two factors. Because one factor is square orthogonal and the other is
semi-orthogonal, all nonzero singular values of the complete projection equal
s initially. Its mean squared row norm matches N*.02^2 (with residual scaling).
This is an initialization statement, not a guarantee after training.

Keep the same d,L,attention,data,tokenizer,seed 17,200-step budget and base AdamW
settings as micro_v1. First test this initialization with the uniform existing
learning rate. An optional second, predeclared optimizer treatment uses the
[Qiu et al. factor fan-in rule](https://arxiv.org/html/2406.06248v1#S3.SS2):
for each projection's factor, multiply LR by N/(2*factor_input_width).
This is an architecture-dependent optimizer treatment, explicitly reported;
it is not an identical-optimizer comparison. Dense components retain multiplier 1.
The dimensional rule is prior art and approximate here, not a tuning-free proof.

Promote only >=1% lower NLL than the better narrow control and noncatastrophic
measured speed/memory. If quality is promising but eager kernels fail the
speed gate, record that limitation and investigate execution independently.
No paper-level reproduction or novel architecture claim: the cited work uses
other scales, a dense first FFN, and optional self-guided training. Here every
FFN is compressed, and all non-FFN choices remain fixed.


## First measured decision and serving audit
Uniform LR reaches NLL 5.017039; factor-aware LR reaches 4.177205 at 200 steps,
a 1.51% improvement over narrow GELU 4.241118. Eager factorized speed is only
53511 tokens/s, so the original serving-speed gate is not met. Training peak
allocated memory is 286403584 bytes. This is quality evidence with a runtime
limitation, not accepted efficiency.

A separately saved post-training audit tests the cited pre-merge idea. Retain
the trained factors and cache each dense equivalent as FP32 or BF16 buffers.
Record cache bytes and construction time, new validation NLL, and steady-state
full-sequence latency and peak inference memory. Compare factorized and cached
modes on the exact same checkpoint. Original training metrics stay unchanged.
If the serving audit recovers usable throughput with negligible NLL change,
fund one 800-step run using the same factors and optimizer recipe. This longer
run examines quality; training still uses the factorized implementation.
