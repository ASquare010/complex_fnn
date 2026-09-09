# H072 - Qualify the existing ungated structured control

Freeze before outcomes. H071's fixed-budget rotations remain rejected. Test the
existing BlockShuffleFFN(gated=False) path at two hidden widths, without changing
any active source, registry, recipe or previous evidence. Read the
[scoped theory](ungated_blockshuffle_theory.md). No optimizer update or corpus
score is allocated. This is a conventional control, not a new architecture.

## Fixed 18 checks

1. Twelve full-size local cases: h2048/h3264 x CPU FP32/CUDA BF16 x seeds17/29/43.
   CPU inputs [2,7,384], CUDA [16,128,384]. Existing name-derived initialization
   uses blocks.0.ffn.up/down and residual down scale 0.25. Synthetic inputs and
   incoming gradients use CPU generator 95000+seed. Compare eager to standard
   non-reentrant whole-FFN checkpointing (preserve_rng_state=True): output,
   input gradient and all four factor gradients must be bitwise identical and
   finite, with each vector's norm nonzero. Save every paired raw tensor and hash.
   This is not a whole-Transformer training-trajectory qualification.
2. One count/topology check: both GELU widths, plain SwiGLU h2048, dense GELU
   h1536, dense SwiGLU h1024 and narrow GELU h456. Match the theory's counts;
   full-size intermediate routing has all 64 path counts equal to six.
3. One optimizer/initialization check: each GELU width, seed17. Every parameter
   appears once. Existing fan-in multipliers are 4/4 for up and 4/(h/96) for
   down, with zero decay at base LR0.001. Verify each represented projection's
   mean row squared norm equals input_width*0.02^2*(residual_scale)^2 to rtol1e-5
   and atol1e-8 after independent matrix construction. No second calibration.
4. Two GELU parity/Jacobian checks, one per width, CPU FP64 with seed17. Inputs
   [14,384] and tangent [1,384] from generator96017. Require the odd identity
   G(x)-G(-x)=D U x and origin JVP=DUv/2 within rtol1e-11, atol1e-12.
5. One SwiGLU parity/separation check, h2048 CPU FP64 seed17: verify the even
   identity S(x)+S(-x)=D[(Ux)*(Vx)] within the same tolerance and origin JVP
   identically zero. Store full-size single-coordinate GELU/SwiGLU witnesses
   at input coordinates1 and2, verifying exact-family obstructions from their
   even/t^2 and odd/t ratios (gaps >0.01). Preserve outputs and factor states.
6. One population-bound quadrature check: d4 cyclic multiplicative target;
   all 3^4=81 nodes, per-coordinate nodes(-3/sqrt5,0,3/sqrt5) and weights
   (5/18,4/9,5/18). Verify covariance matrices (34/5)I and (24/5)I, orthogonality
   of residual to linear inputs, and relative floor12/17 within atol/rtol1e-12.
   Use an independent QR output rotation from seed96018 and fixed scales
   (0.8,1.2,0.9,1.6) to check the rotated/scaled ratio. This integrates moments;
   it neither fits parameters nor validates a finite-sample error floor.

Four CPU threads, TF32 disabled, UV with compile/data extras. One GPU process
under a hidden coordinator. Freeze all previous H071 sources, the three new
source files, this plan and theory before tests. Save process/PID/UTC/return
code, raw tensors, source archive and individual verdicts. No overwrite, automatic
retry, source mutation, tolerance change or additional width after outcomes.

## Decision

All 18 checks must pass to earn a separately frozen learning comparison. It must
include matched dense/narrow controls and distinguish activation change from
width reallocation; report positive-control generalization and absolute error.
Passing does not earn language training, claim a learned advantage, or establish
full-network gradients, novelty or the gold goal. No active model is added.
Failure closes this fixed qualification until a separately justified correction.
Keep H068-H071 and all earlier failed recipes closed under their own protocols.
