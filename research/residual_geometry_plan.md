# H057 - Residual normalization and factor geometry diagnosis

Status: frozen before new checkpoint measurements. H056 is rejected for promotion;
this analysis is motivated by its recorded scale growth, not an independent
confirmatory study. It spends zero optimizer updates and makes no quality gate.

## Fixed inputs and scope

Use H056's six selected seed-17 200-step endpoints: full SwiGLU, full GELU,
calibrated narrow, plain BlockShuffle, square headwise and overcomplete headwise.
Also inspect the two lower-rate overcomplete endpoints. For each of the six
selected recipes reconstruct its exact pretraining initialization, including
narrow width calibration. Load trained states without recalibrating them.

The fixed probe uses train.npy[0:256] reshaped (2,128) as inputs and
train.npy[1:257] as targets. It is a tiny training-prefix diagnostic, never a
validation-quality or convergence estimate. Hash the tokens and all source,
checkpoint, config, tokenizer/cache artifacts. Do not call the training sampler.

CPU FP32: six initial plus eight final probes. Native CUDA BF16: six selected
final probes only, to check whether the ordering survives the training precision.
Use four CPU threads, one device job at a time. Record diagnostic cross entropy,
not a new benchmark ranking. No training, optimizer updates, full validation,
new test data, saved modified checkpoints, model registration or parameter changes.

## Measurements fixed before inspection

For all 17 RMSNorm invocations, capture input x, learned gamma and the loss
adjoint at the norm output. Compute the isolated local input VJP through the
actual RMSNorm module and independently through the derived closed form. Compare
relative L2 <=5e-5 and finite values; this is not the total residual-path gradient.
Report tokenwise p10/median/p90 of input RMS, 1/s, epsilon/s^3, and the actual
local VJP gain ||J^T g||/||g||. Record gamma norm and FFN output/loss-adjoint RMS.
Report candidate/control ratios of geometric means of median local gains over
FFN norms in layers 1-7, and whether native BF16 retains their ordering. No
threshold converts these descriptive observations into a causal claim.

For initial/three trained overcomplete states, inspect every layer's actual
composed rectangular A/B maps on CPU FP64. Record extreme singular values,
condition number, Frobenius norm and private tensor norms. Initialization
isometry is not a maintained training constraint. These spectra are not the
whole nonlinear layer or Transformer Jacobian.

In a disposable copy of the selected trained candidate, apply value *= 8 and
down /= 8 in every headwise FFN. This exact real-arithmetic function symmetry
must preserve probe logits/loss within relative 5e-5, while the associated value
parameter gradients scale by 1/8 and down gradients by 8; other gradients stay
unchanged within relative 5e-5. Restore and verify every original weight. Run
this on CPU FP32, plus a separate FP64 unit comparison with independent inputs.
No optimizer step or changed checkpoint is written. This checks the limits of
interpreting raw parameter-gradient magnitudes across parameterizations.

## Interpretation and stopping boundary

A local inverse-scale gradient effect is established prior art. Derive the exact
finite-epsilon formula for this implementation and verify it; do not claim a new
normalization technique or treat the zero-epsilon invariance as exact here.
Residual identity paths, attention, data, learned gamma and parameter Jacobians
remain part of the full derivative. A small local VJP cannot prove whole-network
vanishing gradients or that normalization caused H056's quality loss.

Complete the bounded diagnosis, retain contradictory observations and all process
failures, and write a report/ledger decision. No architecture or optimizer repair
is automatically earned. A new training hypothesis must state a distinct causal
test or move to a different mechanism. H056's failed quality gates remain binding
for that tested recipe. Existing computations/configurations/plans stay unchanged.
