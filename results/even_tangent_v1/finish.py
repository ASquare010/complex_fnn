"""Independent ridge/scoring audit and publication of the local diagnostic."""

import math
import statistics as st
from pathlib import Path

import numpy as np
import torch

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/even_tangent_v1")
p, r = [read(ROOT / n) for n in ("protocol_dtype64.json", "result.json")]
assert not (ROOT / "receipt.json").exists()
assert (ROOT / "check_exit.txt").read_text().strip() == "1"
assert (ROOT / "check_dtype64_exit.txt").read_text().strip() == "0"
hashes(read(ROOT / "protocol.json")["sources"])
hashes(p["sources"])
hashes(p["maintained_files"])
assert sha(ROOT / "protocol.json") == p["recovery"]["original_protocol_sha256"]
assert sha("results/readout_learning_v1/receipt.json") == p["prior_receipt"]
for path, digest in read("results/readout_learning_v1/receipt.json")["files"].items():
    target = (
        ROOT / "CURRENT_STATE.before.md"
        if path == "research/CURRENT_STATE.md"
        else ROOT / "README.before.md"
        if path == "README.md"
        else Path(path)
    )
    assert sha(target) == digest
assert len(r["cases"]) == 9 and r["ridge_solves"] == 81
checks = []
for c in r["cases"]:
    assert sha(c["path"]) == c["sha256"]
    s = {k: v.numpy() for k, v in torch.load(c["path"], weights_only=True).items()}
    y = s["target"]
    ym = float(s["ymean"])
    assert math.isclose(ym, float(y[:1024].mean()), abs_tol=1e-12)
    predictions = {}
    for fit in c["fits"]:
        name, lr = fit["arm"], fit["rate"]
        F = s[f"design_{name}"]
        coef = s[f"coef_{name}_{lr}"]
        rhs = F[:1024].T @ (y[:1024] - ym) / 1024
        residual = F[:1024].T @ (F[:1024] @ coef) / 1024 + lr * coef - rhs
        normal_error = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30))
        assert normal_error <= 1e-8
        pred = ym + F @ coef
        predictions[name, lr] = pred
        for key, lo, hi in [("validation_mse", 1024, 1536), ("reporting_mse", 1536, 2560)]:
            mse = float(np.mean((pred[lo:hi] - y[lo:hi]) ** 2))
            assert math.isclose(mse, fit[key], rel_tol=1e-10, abs_tol=1e-10)
        checks.append(
            dict(
                seed=c["seed"],
                probe=c["probe"],
                arm=name,
                rate=lr,
                normal_equation_error=normal_error,
            )
        )
    for lr in p["rates"]:
        assert np.max(np.abs(predictions["bias", lr] - predictions["duplicate", lr])) <= 1e-9
    for name, chosen in c["selected"].items():
        assert chosen == min(
            (f for f in c["fits"] if f["arm"] == name),
            key=lambda f: (f["validation_mse"], f["rate"]),
        )
    for replay in c["replays"]:
        pred = s[f"replay_{replay['arm']}_{replay['fraction']}"]
        mse = float(np.mean((pred - y[1536:]) ** 2))
        assert math.isclose(mse, replay["mse"], rel_tol=1e-10, abs_tol=1e-10)
assert not torch.cuda.is_initialized()
write_json(
    ROOT / "audit.json",
    dict(
        status="VERIFIED",
        ridge_checks=checks,
        training_updates=0,
        backwards=0,
        cuda_initialized=False,
    ),
)
rows = []
for c in r["cases"]:
    replay = next(v for v in c["replays"] if v["arm"] == "augmented" and v["fraction"] == 0.1)
    rows.append(
        f"| {c['seed']} | {c['probe']} | {c['bias_rank']} | {c['eta_novel_energy_fraction']:.4f} | {c['selected']['bias']['reporting_mse']:.4f} | {c['selected']['augmented']['reporting_mse']:.4f} | {c['tangent_ratio']:.4f} | {c['replay_ratio']:.4f} | {replay['eta_update_norm']:.2f} |"
    )
replayrows = []
for fraction in p["fractions"]:
    for arm in ("bias", "augmented"):
        values = [
            v["mse"] / c["zero_mse"]
            for c in r["cases"]
            for v in c["replays"]
            if v["arm"] == arm and v["fraction"] == fraction
        ]
        replayrows.append(
            f"| {fraction} | {arm} | {st.mean(values):.4f} | {st.median(values):.4f} | {st.variance(values):.3e} |"
        )
gate = (
    "PASS local even-direction diagnostic"
    if r["passed"]
    else "NO PROMOTION from even-direction diagnostic"
)
report = f"""# H153: direct even-shape tangents add little in this local test

**{gate}.** At the H152 initialization, the proposed control adds mathematically
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
{chr(10).join(rows)}

Distinct feature directions alone did not provide consistent useful improvement.
The local fitted model is not the nonlinear model after a finite parameter update.
Replay evaluates only its even output component; the remaining odd output error
is excluded, so these numbers must not be presented as full prediction MSE.

| Fixed step fraction | Direction set | Mean normalized replay MSE | Median | Sample variance |
|---:|---|---:|---:|---:|
{chr(10).join(replayrows)}

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
"""
reportpath = Path("research/even_tangent_results.md")
reportpath.write_text(report, encoding="utf-8")
for path, name in [
    (Path("research/CURRENT_STATE.md"), "CURRENT_STATE"),
    (Path("README.md"), "README"),
]:
    assert sha(path) == sha(ROOT / (name + ".before.md"))
    head, body = path.read_text(encoding="utf-8").split("\n\n", 1)
    if name == "CURRENT_STATE":
        body = body.replace("## Latest:", "## Previous:", 1)
        intro = f"## Latest: even-shape tangent diagnostic\n\n[H153](even_tangent_results.md): **{gate}**. Exact local parity,\ntwo-sided scalar inverse and finite-difference checks pass. Nine probes,81 ridge\nsolves and54 actual nonlinear replays do not meet all held-out gates. Original\nFP32-in-FP64 fixture failure and bounded dtype-only recovery retained. Zero GPU\ntraining; broad research goal stays open."
    else:
        body = body.replace("Latest:", "Earlier:", 1)
        intro = f"Latest: [H153 even-direction diagnostic](research/even_tangent_results.md):\n{gate}. Tangent-space gains do not establish nonlinear learning."
    path.write_text(head + "\n\n" + intro + "\n\n" + body, encoding="utf-8")
files = [
    f
    for f in ROOT.rglob("*")
    if f.is_file()
    and "unused_cache" not in f.parts
    and f.name not in ("receipt.json", "finish.log", "finish_exit.txt")
]
files += [
    reportpath,
    Path("research/even_tangent_plan.md"),
    Path("research/CURRENT_STATE.md"),
    Path("README.md"),
]
receipt = dict(
    study="H153",
    status="EVIDENCE_VERIFIED",
    gate=gate,
    training_updates=0,
    ridge_solves=81,
    goal_achieved=False,
    files={f.as_posix(): sha(f) for f in files},
)
write_json(ROOT / "receipt.json", receipt)
hashes(receipt["files"])
print({k: v for k, v in receipt.items() if k != "files"})
