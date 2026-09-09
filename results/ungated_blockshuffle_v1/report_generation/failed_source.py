"""Render H072 qualified control and scoped exact-function analysis."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/ungated_blockshuffle_v1")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


r = read(ROOT / "result.json")
a = read("results/verification/ungated_blockshuffle_analysis_v1.json")
process = read(ROOT / "process.json")
assert r["all_checks_pass"] and a["status"] == "PASS"
text = f"""# H072 - Ungated BlockShuffle qualification

**LOCALLY QUALIFIED as a conventional control.** The already implemented
`BlockShuffleFFN(gated=False)` path passes all 18 fixed checks at full-size
projection dimensions. No new activation, active model or training recipe is
introduced. Its learning quality and full-model resources remain unmeasured.

| Form | Hidden | Weights / FFN | FFN weights, L8 | Total model, L8 | FFN reduction |
|---|---:|---:|---:|---:|---:|
| Plain BlockShuffle SwiGLU | 2,048 | 350,208 | 2,801,664 | 9,099,648 | 70.3125% |
| Same-width BlockShuffle GELU | 2,048 | 233,472 | 1,867,776 | 8,165,760 | 80.2083% |
| Matched-weight BlockShuffle GELU | 3,264 | 350,208 | 2,801,664 | 9,099,648 | 70.3125% |
| Matched narrow dense GELU | 456 | 350,208 | 2,801,664 | 9,099,648 | 70.3125% |

Using two projections instead of three allows 59.375% more hidden features at
the retained parameter budget. It changes activation and width allocation;
the same-width control is needed to distinguish them. Four grouped matrix
operations replace six, with equal matrix MACs at the matched weight count.
Different nonlinear and allocation costs prevent an unmeasured speed claim.

## Numerical qualification

Twelve cases cover two widths, seeds 17/29/43 and CPU FP32/CUDA BF16. All 72
output/input/factor-gradient comparisons between eager and standard whole-FFN
checkpointing are bitwise identical, finite and nonzero in norm. GPU inputs are
[16,128,384]; CPU inputs [2,7,384]. These probes are not full-model optimizer
trajectory or training-memory qualifications.

Counts, all 64 canonical path counts of six, complete optimizer-group coverage
and initialization calibration pass. The down second-factor LR multiplier is
21.3333 at h2048 and 34 at h3264; other factor multipliers are four. Independent
factorized matrix reconstruction verifies all four represented projections.
Mean row norms match the frozen dense-equivalent targets within tolerance.

The process completes first attempt: 18 passed in 5.79 s, {process["elapsed_seconds"]:.2f} s
including overhead. There are zero optimizer updates, corpus targets or full-
model resource workers. Every prior active source/configuration/test byte remains
unchanged; these checks are separate from the previously passing 108-test suite.

## Exact functional tradeoff

For bias-free linear U,V,D, GELU phi(t)=t Phi(t), and SiLU s(t):

    G(x) = D phi(Ux)           -> G_odd(x) = D U x / 2
    S(x) = D[s(Ux)*(Vx)]       -> S_even(x) = D[(Ux)*(Vx)] / 2

Every single finite bias-free GELU FFN has a linear odd part. Every single finite
bias-free SwiGLU FFN has a homogeneous quadratic even part. Full-size sparse
single-coordinate witnesses show that neither exact-function family contains
the other. The GELU odd identity has maximum FP64 error 3.33e-16; the SwiGLU even
identity 9.71e-17. GELU's origin JVP matches DUv/2 exactly in the measured cases;
SwiGLU's is exactly zero. Singular factors and other network operations remain:
this provides no whole-network nonvanishing-gradient guarantee.

![Scalar witnesses for the parity restrictions](figures/ungated_blockshuffle.png)

For the multiplicative target T=ab+2bcd+a^2 d under independent standardized
uniform inputs, the best linear approximation of its odd part is d. Its residual
2bcd+(a^2-1)d has variance 24/5; the full target variance is 34/5. Hence every
single bias-free GELU FFN, of any width, has a relative population squared-error
floor **12/17 = 70.5882% of zero-predictor error** on this target. Even error may
add more; attainability is not claimed. An exact rational derivation and 81-node
polynomial quadrature agree, including an independent rotation/scaling check.

The cyclic d384 target has the corresponding isotropic covariances, so fixed
orthogonal output mixing and positive coordinate scales preserve this population
ratio. This is not an exact finite-held-out bound, a limit for biased/deeper GELU
networks, or a claim that SwiGLU learns the target efficiently. H071's observed
generalization failures remain. Read the [proof and its scope](ungated_blockshuffle_theory.md).

## Prior art and next decision

[GELU](https://arxiv.org/abs/1606.08415), [GLU variants](https://arxiv.org/abs/2002.05202)
and [BlockShuffle structured FFNs](https://arxiv.org/html/2406.16450v1#S2.SS1)
are established. The existing ungated branch, budget calculation and elementary
symmetry arguments establish no architectural novelty or practical superiority.
H037 already investigated adding a linear path to a gated FFN; its matched-rate
negative remains relevant and is not reopened by the origin-Jacobian observation.

Passing earns only a separately frozen learning comparison with matched dense/
narrow controls and a same-width ablation. That comparison must state whether
positive controls generalize and report absolute error, after H071 showed that
every selected model lost to zero on five target families. A larger hidden width
must not be treated as universally more expressive, nor as a remedy for the
proved odd-component restriction. No language allocation or active variant is
earned by local derivative tests. The broader research goal remains unmet.

- [Frozen plan](ungated_blockshuffle_plan.md).
- [Raw results](../results/ungated_blockshuffle_v1/result.json) and [process](../results/ungated_blockshuffle_v1/process.json).
- [Independent tensor/moment audit](../results/verification/ungated_blockshuffle_analysis_v1.json).
- [Final preservation audit](../results/verification/ungated_blockshuffle_final_v1.json).

Result SHA-256: `{sha(ROOT / "result.json")}`.
Protocol SHA-256: `{sha(ROOT / "protocol.json")}`.
"""
Path("research/ungated_blockshuffle_results.md").write_text(text, encoding="utf-8")
t = np.linspace(0.1, 3, 300)
phi = t * 0.5 * (1 + np.array([math.erf(v / math.sqrt(2)) for v in t]))
even = (phi - 0.5 * t) / t**2
odd = 0.5 * t * np.tanh(t / 2)
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes[0].plot(t, even, label="GELU(t)", color="#287f78")
axes[0].axhline(0.5, label="t SiLU(t)", color="#b47643", linestyle="--")
axes[0].set(title="Even component / t?", xlabel="Positive input t", ylabel="Ratio")
axes[1].axhline(0.5, label="GELU(t)", color="#287f78")
axes[1].plot(t, odd, label="t SiLU(t)", color="#b47643")
axes[1].set(title="Odd component / t", xlabel="Positive input t", ylabel="Ratio")
for ax in axes:
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
fig.suptitle("H072: exact single-layer bias-free witnesses, not measured learning curves")
for ext in ("png", "svg"):
    fig.savefig(f"research/figures/ungated_blockshuffle.{ext}", dpi=160)
plt.close(fig)
print(json.dumps({"status": "PASS", "report": "research/ungated_blockshuffle_results.md"}))
