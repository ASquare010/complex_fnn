"""Qualify the custom boundary and freeze the actual training screen."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.reversible_training_v1.model import Model, Reconstruct, ARMS, stack
from pathlib import Path

ROOT = Path("results/reversible_training_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/reversible_softsign_v1/receipt.json")["files"])
base = read("results/reversible_softsign_v1/protocol.json")
torch.manual_seed(148)
checks = []
for kind in ("learned", "affine"):
    m = Model(kind + ":reconstruct", d=12, depth=4).double()
    # Generate exactly FP64 orthogonal test matrices, not FP32 matrices upcast.
    m.q.copy_(torch.linalg.qr(torch.randn(4, 12, 12, dtype=torch.float64)).Q)
    x = torch.randn(3, 12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, t, b, q=m.q, rho=m.rho, kind=kind: Reconstruct.apply(x, t, b, q, rho, kind),
        (x, m.theta, m.bias),
        fast_mode=True,
    )
    out = m(x)
    ref = stack(x, m.theta, m.bias, m.q, m.rho, kind)
    ga = torch.autograd.grad(out.square().mean(), (x, m.theta, m.bias))
    gb = torch.autograd.grad(ref.square().mean(), (x, m.theta, m.bias))
    for u, v in zip(ga, gb, strict=True):
        torch.testing.assert_close(u, v, rtol=1e-10, atol=1e-12)
    if kind == "learned":
        ck = Model("learned:checkpoint", d=12, depth=4).double()
        ck.load_state_dict(m.state_dict())
        gc = torch.autograd.grad(ck(x).square().mean(), (x, ck.theta, ck.bias))
        for u, v in zip(ga, gc, strict=True):
            torch.testing.assert_close(u, v, rtol=1e-10, atol=1e-12)
    checks.append(dict(kind=kind, gradcheck=True))
traces = []
counts = {}
for arm in ARMS:
    m = Model(arm)
    counts[arm] = dict(
        parameters=sum(v.numel() for v in m.parameters()),
        buffer_bytes=sum(v.numel() * v.element_size() for v in m.buffers()),
        macs=m.macs,
    )
    x = torch.randn(2048, 384, requires_grad=True)
    par = {v.untyped_storage().data_ptr() for v in m.parameters()}
    buf = {v.untyped_storage().data_ptr() for v in m.buffers()}
    original = x.untyped_storage().data_ptr()
    records = []

    def pack(t):
        s = t.untyped_storage()
        ptr = s.data_ptr()
        records.append(
            dict(
                storage=ptr,
                bytes=s.nbytes(),
                shape=list(t.shape),
                kind="parameter"
                if ptr in par
                else "buffer"
                if ptr in buf
                else "input"
                if ptr == original
                else "activation",
            )
        )
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = m(x)
    unique = {v["storage"]: v for v in records}
    traces.append(
        dict(
            arm=arm,
            records=records,
            unique_bytes={
                kind: sum(v["bytes"] for v in unique.values() if v["kind"] == kind)
                for kind in ("parameter", "buffer", "input", "activation")
            },
        )
    )
    del y, m, x
assert (
    counts["learned:reconstruct"]["parameters"] == 6144
    and counts["fixed:reconstruct"]["parameters"] == 3072
)
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
extra = [
    Path("results/reversible_softsign_v1/model.py"),
    Path("results/modulation_depth_v1/model.py"),
    Path("results/split_modulation_v1/model.py"),
]
write_json(
    ROOT / "protocol.json",
    dict(
        study="H148",
        arms=ARMS,
        batches=[128, 2048],
        seeds=[173, 191, 211],
        counts=counts,
        checks=checks,
        traces=traces,
        training_updates=720,
        maintained_files=base["maintained_files"],
        prior_receipt=sha("results/reversible_softsign_v1/receipt.json"),
        sources={
            f.as_posix(): sha(f)
            for f in [*ROOT.glob("*.py"), Path("research/reversible_training_plan.md"), *extra]
        },
    ),
)
print(
    "Custom input/parameter gradchecks and execution equivalence pass; buffer-aware traces recorded."
)
