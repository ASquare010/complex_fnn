# Feature flow: when internal depth first matters

Analytic audit during the frozen language campaign, before running the numerical
check below. It changes no architecture, acceptance threshold or training recipe.
The audit uses small synthetic FP64 arrays on CPU only, no model/checkpoint/data,
and makes no language or efficiency claim.

Write the mixing matrix as epsilon A and let D be diag(psi'(z)). For K identical
Euler steps of total nominal time one, Taylor expansion around epsilon=0 gives

```text
h[K] = z + epsilon A psi(z)
       + epsilon^2 (K-1)/(2K) A D A psi(z) + O(epsilon^3).
```

Induction: after t steps the first-order coefficient is t A psi(z)/K. The next
update adds t A D A psi(z)/K^2 to the second-order coefficient. Summing t=0..K-1
gives (K-1)/(2K). Thus K=3 adds epsilon^2 A D A psi(z)/3 relative to K=1. After
the common final phi, the leading difference is that quantity multiplied
coordinate-wise by phi'(z), before the common Q readout.

At M=0, every K has h=z and the same first derivative with respect to M: each
step contributes 1/K of the same response. Hence initialized one-step and
three-step models have equal functions and first gradients. Depth's additional
composition starts at second order in M, rather than adding independent
detectors at initialization. This does not imply equality after training or
prove that a particular learned M is too small to matter. Diagonal M can also
produce these higher-order terms without cross-neuron exchange.

Check the expansion with random two-group four-coordinate matrices, thirteen
CPU FP64 inputs, fixed phi(h)=.8*(h^2*sigmoid(h)-.03) and psi=tanh(phi). At
epsilon=.2/.1/.05/.025, compare the three-step minus one-step state/output
difference to the analytic second-order term. The residual should shrink
cubically, while the measured difference shrinks quadratically; record all
errors/scaled coefficients. Central differences at M=0 also check equal first
matrix derivatives for one and three steps. No optimization or selection uses
these synthetic results. Evidence will be `records/feature-flow-expansion.json`.

The registered retrained one-step and diagonal controls are therefore necessary:
a resource-efficient repeated operation alone does not show useful added depth
or interaction. Frozen one-step/diagonal removals complement those controls but
can change activation statistics and do not identify semantic information.
