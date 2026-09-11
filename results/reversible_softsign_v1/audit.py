"""Audit against independent NumPy forward-retained activations and VJPs."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import read, hashes, sha
from results.ordinary_long_training_v1.io import write_json
from pathlib import Path
import numpy as np

ROOT = Path("results/reversible_softsign_v1")
p, r = [read(ROOT / n) for n in ("protocol.json", "result.json")]
assert not (ROOT / "audit.json").exists()
assert (ROOT / "worker_exit.txt").read_text().strip() == "0"
hashes(p["sources"])
hashes(p["maintained_files"])
inputs = {
    f.as_posix(): sha(f)
    for f in ROOT.iterdir()
    if f.is_file() and f.suffix in (".py", ".pt", ".json")
}
write_json(
    ROOT / "audit_protocol.json",
    dict(
        inputs=inputs,
        method="NumPy FP64 forward-retained VJP, no inverse or autodiff",
        tolerances=dict(float32=1e-4, float64=1e-10),
    ),
)
assert len(r["cases"]) == 36 and len(r["boundaries"]) == 37
assert r["native_backwards"] == 36 and r["training_updates"] == 0
assert all(b["gpu"] == dict(allocated=0, reserved=0) for b in r["boundaries"])


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


rows = []
for i, c in enumerate(r["cases"]):
    assert c == read(ROOT / f"case{i:02d}.json") and sha(c["path"]) == c["sha256"]
    s = {
        k: v.numpy().astype(np.float64) for k, v in torch.load(c["path"], weights_only=True).items()
    }
    assert all(np.isfinite(v).all() for v in s.values())
    q, theta, bias = s["q"], s["theta"], s["bias"]
    rho = c["c"] / c["depth"]
    x = s["x"].copy()
    zs = []
    for j in range(c["depth"]):
        z = np.einsum("bi,ji->bj", x, q[j]) + bias[j]
        zs.append(z)
        x = z + rho * np.tanh(theta[j]) * z / (1 + np.abs(z))
    g = s["g"].copy()
    gt, gb = np.empty_like(theta), np.empty_like(bias)
    for j in reversed(range(c["depth"])):
        z = zs[j]
        t = np.tanh(theta[j])
        gt[j] = np.sum(g * (z / (1 + np.abs(z))), axis=0) * rho * (1 - t * t)
        dz = g * (1 + rho * t / (1 + np.abs(z)) ** 2)
        gb[j] = np.sum(dz, axis=0)
        g = np.einsum("bj,ji->bi", dz, q[j])
    errors = dict(output=rel(s["y"], x), reconstruction=rel(s["reconstructed"], s["x"]))
    for label, reference in [("x", g), ("theta", gt), ("bias", gb)]:
        errors["native_" + label] = rel(s["native_" + label], reference)
        errors["reverse_" + label] = rel(s["reverse_" + label], reference)
    tol = 1e-4 if c["dtype"] == "torch.float32" else 1e-10
    # Independent audit threshold checks each gradient family, not just their concatenation.
    passed = max(errors.values()) <= tol
    assert passed
    orthogonal_error = max(float(np.linalg.norm(v.T @ v - np.eye(16), ord=2)) for v in q)
    assert orthogonal_error <= (1e-6 if c["dtype"] == "torch.float32" else 1e-12)
    rows.append(dict(index=i, errors=errors, orthogonal_error=orthogonal_error, passed=passed))
assert not torch.cuda.is_initialized()
hashes(inputs)
write_json(
    ROOT / "audit.json",
    dict(
        status="VERIFIED",
        cases=rows,
        backwards=0,
        cuda_initialized=False,
        torch_version=torch.__version__,
        cuda_build=torch.version.cuda,
    ),
)
print("36 saved cases independently verified using forward-retained NumPy gradients.")
