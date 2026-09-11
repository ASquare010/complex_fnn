"""Check checkpoint equivalence and measure forward-retained tensor storage."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.modulation_depth_v1.model import Model, ARMS, BASE_ARMS
from pathlib import Path

ROOT = Path("results/modulation_depth_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/split_modulation_v1/receipt.json")["files"])
base = read("results/split_modulation_v1/protocol.json")
hashes(base["maintained_files"])
torch.manual_seed(146)
checks = []
for arm in BASE_ARMS:
    eager = Model(arm + ":eager", d=12, depth=3).double()
    if "gate" in arm:
        with torch.no_grad():
            for block in eager.blocks:
                block.gate.weight.normal_(0, 0.02)
    ckpt = Model(arm + ":checkpoint", d=12, depth=3).double()
    ckpt.load_state_dict(eager.state_dict())
    x = torch.randn(4, 12, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = eager(x), ckpt(y)
    ga = torch.autograd.grad(a.square().mean(), (x, *eager.parameters()))
    gb = torch.autograd.grad(b.square().mean(), (y, *ckpt.parameters()))
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
    for u, v in zip(ga, gb, strict=True):
        torch.testing.assert_close(u, v, rtol=1e-10, atol=1e-12)
    checks.append(
        dict(
            arm=arm,
            max_gradient_absolute_error=max(
                (u - v).abs().max().item() for u, v in zip(ga, gb, strict=True)
            ),
        )
    )
traces = []
for arm in ARMS:
    model = Model(arm)
    x = torch.randn(2048, 384, requires_grad=True)
    params = {v.untyped_storage().data_ptr() for v in model.parameters()}
    original = x.untyped_storage().data_ptr()
    records = []

    def pack(t):
        storage = t.untyped_storage()
        ptr = storage.data_ptr()
        records.append(
            dict(
                storage=ptr,
                bytes=storage.nbytes(),
                shape=list(t.shape),
                stride=list(t.stride()),
                kind="parameter" if ptr in params else "input" if ptr == original else "activation",
            )
        )
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(x)
    unique = {v["storage"]: v for v in records}
    traces.append(
        dict(
            arm=arm,
            records=records,
            unique_bytes={
                kind: sum(v["bytes"] for v in unique.values() if v["kind"] == kind)
                for kind in ("parameter", "input", "activation")
            },
            hook_calls=len(records),
        )
    )
    del out, model, x
counts = {
    arm: dict(parameters=sum(v.numel() for v in Model(arm).parameters()), macs=Model(arm).macs)
    for arm in ARMS
}
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
assert not torch.cuda.is_initialized()
p = dict(
    study="H146",
    arms=ARMS,
    batches=[128, 2048],
    seeds=[83, 97, 109],
    counts=counts,
    training_updates=720,
    checks=checks,
    traces=traces,
    maintained_files=base["maintained_files"],
    prior_receipt=sha("results/split_modulation_v1/receipt.json"),
    sources={
        f.as_posix(): sha(f)
        for f in [
            *ROOT.glob("*.py"),
            Path("research/modulation_depth_plan.md"),
            Path("results/split_modulation_v1/model.py"),
        ]
    },
)
write_json(ROOT / "protocol.json", p)
print("Five gradient-equivalence checks and ten saved-storage traces passed; CUDA unused.")
