# H145: resource-first test of separate base/gate observations

Previous turn: progress. H144 proved a capacity obstruction for a shared latent
observation; H143 showed that half as many parameters need not materially lower
VRAM. Test resource feasibility before allocating new learning experiments.

Candidate shared: z=GELU(Ux+a), F=Bz+c+x*(Cz+e), h192.
Candidate split: z=GELU(Ux+a) with288 features; base reads the first192, gate reads
the final96. One concatenated projection implements the independent banks.
The split observation has rank at most288; H144 still applies to this combined
observation. Both chosen widths remove its particular skew-target lower bound;
that is not a general approximation or learning guarantee.

Controls: full GELU h384; full SwiGLU h256; budget GELU h288; budget GELU h288
plus a learned diagonal input path. Biases are included. Split and static-diagonal
controls have exactly equal parameters and matrix MACs; shared differs by96 biases.
Candidate matrix MACs are25% below both wide controls. Gate readouts and static
diagonal initialize at zero. Shared/split base tensors and initial functions match
exactly through name-local initialization; other projections use consistent fan-in
scaling, no labels. This initial zero gate is deliberate, not a claim of permanent
zero gradients or an optimization improvement.

Use FP64 input/parameter gradcheck after perturbing gate weights away from zero;
verify explicit independent formulas and actual parameter/MAC counts before CUDA.
No architecture/default changes. Modulation/FiLM, low-rank maps and feature splitting
have prior art; novelty is unestablished. See H144's cited primary-source survey.

Resource preflight: d384, batches128 and2048, seeds53/67/79, all six arms. Each
starts fresh and runs12 AdamW updates on one fixed Gaussian input batch and its
independent orthogonal linear target. LR.003, betas(.9,.95),zero decay,clip1,FP32,
TF32off, four CPU threads, ordinary eager execution and default workspace. Data
stay CPU until the measured run; count model, buffers, resident batch, gradients,
optimizer and temporaries in job peak. No validation is run and no learning-quality
claim follows. Alternate each six-arm order by deterministic cyclic rotation.

Budget36 runs x12=432 updates/backwards; 37 zero CUDA allocator boundaries.
Four warmups and eight complete CUDA/wall update timings per run; gradient reset
is outside timing, forward/backward/clip/Adam inside. Report all peaks and timings,
finite flags, input/target hashes, raw parameter/fixed-buffer/optimizer byte counts,
initial/final loss (diagnostic only), and first/final gradient norms by module.
No tuning, clocks/power changes, selected repeats, or hidden retries.

For each dynamic candidate to earn fitting, every batch/seed must use at least20%
fewer trainable parameters than both wide controls, GPU peak<=0.90 of both,
complete median CUDA/wall<=1.15 of both, and half-window median stability<=1.15.
These are resource-allocation gates, not language-quality or throughput proofs.
The matched-budget/static controls must accompany all comparisons. Rejection only
closes this eager resource recipe at these settings; do not silently add kernels,
recomputation, longer training or change thresholds to rescue it.

Freeze all sources/conditions before execution, save states and verify them after.
If resources pass, freeze a separate learning comparison with strong positive
controls and mandatory activation baselines. If not, document the allocation
tradeoff and keep the broader research goal open. Previous failures remain intact.

Inputs require gradients to model an internal FFN's upstream backward cost. Clear
input gradients alongside optimizer gradients outside each update timer. Record
post-clipping parameter-gradient norms and input-gradient norms after updates1
and12. A final no-grad score is included in job peak but excluded from timing;
its saved model state will be checked by an independently written scorer.
