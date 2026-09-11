# H153: direct even-shape tangents add little in this local test

**NO PROMOTION from even-direction diagnostic.** At the H152 initialization, the proposed control adds mathematically
even tangent directions, but the held-out ridge/replay criteria are not met across
all nine probes. This is a local diagnostic, not a trained-network benchmark or
a global impossibility result. It does not justify another full training sweep.

## Mechanism and exact local parity

The original activation x+rho*tanh(theta)*x/(1+abs(x)) is odd. With zero core
biases and zero readout bias, its stack and readout are odd for every theta.
Differentiating with respect to theta or readout weights preserves odd parity.
Core-bias derivatives are even: the derivative of the original odd scalar map is
even, and the backward factors remain even under x -> -x.

Thus an even target cannot correlate with theta/readout-weight tangent features
under a symmetric distribution. Those parameters can still reduce the odd output
component of the loss. Existing biases already supply even directions and can
break symmetry during training; this is not a claim that H152 can never learn
even functions or that its entire loss gradient is zero.

Consider the two-sided extension

    phi(x)=x+rho*tanh(theta+sign(x)*eta)*x/(1+abs(x)).

At eta=0 it is the original activation, and

    dphi/deta = rho*(1-tanh(theta)^2)*abs(x)/(1+abs(x)).

This is an even direction. Each half-axis slope lies in[1-rho,1+rho], so for
rho<1 the continuous map is strictly increasing and globally invertible. Nonzero
eta generally creates a derivative kink at zero; smoothness is not claimed.
Use sign(y) to select the half-axis coefficient and the existing quadratic inverse.
The FP64 signed-range inverse check passes after the fixture correction below.
At d32/L4 this adds128 parameters:1312 ->1440, without duplicating feature outputs.
Actual GPU storage/latency of this extension has not been implemented or measured.

## Diagnostic controls

Fresh model/data seeds307/317/331, scalar readout probes0/11/23, targets
x_i*x_(i+1). Each probe has its own fitted coefficients, making this a relaxation
of a shared32-output model. Train/validation/reporting counts are1024/512/1024;
antithetic inputs isolate even feature components. Feature mean/RMS and target
mean use training samples only. Analytic theta/bias/eta tangents pass directional
central finite differences; unwanted parity components are below1e-12 relative norm.

Three ridge designs: existing bias directions; duplicated bias directions;
bias+eta directions. Duplicate blocks are scaled by1/sqrt(2), preserving the
feature Gram kernel and ridge function class. All duplicate predictions match
bias-only within1e-9. This prevents attributing a gain to column duplication or
changed ridge strength. Training-normalized bias+eta blocks use the same scaling.
Ridge penalties1e-6/.001/1 are chosen by validation only, using stable SVD solves.
An unrestricted tangent solution can require parameter changes outside its local
validity. Therefore selected coefficients are replayed in the actual nonlinear
model at fixed fractions.01/.1/1; no fraction is selected after seeing results.

## Held-out tangent and actual-update results

Novel eta energy is the fraction outside the numerical training bias-feature span
(rank threshold1e-9 relative to the largest singular value). Tangent ratio is
augmented/bias-only MSE. Replay ratio uses the actual even output at fraction.1,
divided by the zero-even predictor's target energy. Both ratios must be<=.95 in
every probe to pass. Eta norm reports the actual.1-step parameter displacement.

| Seed | Probe | Bias rank | Novel eta energy | Bias tangent MSE | Augmented MSE | Tangent ratio | Replay ratio | Eta step norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 307 | 0 | 128 | 0.0160 | 1.0012 | 0.9797 | 0.9786 | 0.9894 | 3.72 |
| 307 | 11 | 128 | 0.0150 | 0.8786 | 0.8615 | 0.9805 | 0.9893 | 4.35 |
| 307 | 23 | 128 | 0.0156 | 0.8724 | 0.8479 | 0.9718 | 0.9864 | 4.73 |
| 317 | 0 | 128 | 0.0159 | 1.0066 | 0.9868 | 0.9803 | 0.9856 | 4.39 |
| 317 | 11 | 128 | 0.0162 | 0.9277 | 0.9170 | 0.9885 | 0.9859 | 3.35 |
| 317 | 23 | 128 | 0.0161 | 0.9605 | 1.0049 | 1.0462 | 1.0399 | 64.50 |
| 331 | 0 | 128 | 0.0160 | 1.0305 | 1.0042 | 0.9745 | 0.9858 | 3.67 |
| 331 | 11 | 128 | 0.0161 | 0.8643 | 0.8508 | 0.9845 | 0.9861 | 4.40 |
| 331 | 23 | 128 | 0.0151 | 0.8799 | 0.8490 | 0.9649 | 0.9852 | 4.87 |

Distinct feature directions alone did not provide consistent useful improvement.
The local fitted model is not the nonlinear model after a finite parameter update.
Replay evaluates only its even output component; the remaining odd output error
is excluded, so these numbers must not be presented as full prediction MSE.

| Fixed step fraction | Direction set | Mean normalized replay MSE | Median | Sample variance |
|---:|---|---:|---:|---:|
| 0.01 | bias | 0.9989 | 0.9989 | 5.332e-08 |
| 0.01 | augmented | 0.9982 | 0.9985 | 1.389e-06 |
| 0.1 | bias | 0.9901 | 0.9896 | 5.305e-06 |
| 0.1 | augmented | 0.9926 | 0.9861 | 3.169e-04 |
| 1.0 | bias | 1.0400 | 0.9899 | 1.095e-02 |
| 1.0 | augmented | 1.2095 | 1.0144 | 1.856e-01 |

These descriptive aggregates combine correlated probes and three seeds; they are
not significance tests or independent neural training runs. All81 ridge solves,
selected coefficients, features, targets and54 nonlinear replay outputs are saved.
Independent NumPy auditing verifies normal equations, validation/reporting MSE,
selection and duplicate predictions. It re-scores saved nonlinear replay outputs;
it does not independently regenerate those outputs. Finite differences provide
an independent check of the analytic tangent implementation.

## Recovery, prior work and decision

The first stage stopped before feature fits: Python scalar operands in torch.where
created FP32 coefficients inside an intended FP64 inverse fixture. The1+a inverse
intermediate then rounded in FP32. A single bounded recovery explicitly used
FP64 theta/eta tensors, preserving the equation, tolerance and all seeds. Original
source, protocol and failed log remain. The recovery and its distinct protocol
are documented in recovery.md; no model training or GPU work was repeated.

[Even activations](https://arxiv.org/abs/2011.11713) and the repository's H092/H099
parity mixtures establish prior work. This two-sided invertible parameterization
is a distinct test within the current reconstructed stack, not a novelty claim.
The earlier non-invertible parity-mixture recipes remain rejected.

Do not equate an even-direction derivative with a useful nonlinear learner.
This test weakens the case for adding shape coefficients alone to the fixed
mixing recipe. A next hypothesis should address the projected feature directions
or finite-update optimization, with comparable low-parameter controls. It must
be independently tested; the current evidence does not establish a solution.

Zero optimizer updates, zero autograd backwards, zero GPU initialization.
The ridge solves and finite differences ran on CPU. Maintained modules/defaults
and all prior receipt hashes verify. The full research goal remains open.

[Plan](even_tangent_plan.md), [checks](../results/even_tangent_v1/result.json),
[audit](../results/even_tangent_v1/audit.json),
[recovery](../results/even_tangent_v1/recovery.md),
[receipt](../results/even_tangent_v1/receipt.json).
