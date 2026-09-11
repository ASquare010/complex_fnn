"""Freeze the source states and validate local approximate-VJP semantics on CPU."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.checkpoint_fp16_v1.codec import compress_inputs
from src.core.config import ModelConfig
from src.core.transformer import Transformer
from pathlib import Path
import copy
import shutil

ROOT = Path("results/checkpoint_fp16_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/batch_scale_long_v1/receipt.json")["files"])
base = read("results/batch_scale_long_v1/protocol.json")
fixtures = []
for row in base["schedule"]:
    if row["arm"] != "ordinary":
        continue
    c = read(Path("results/batch_scale_long_v1") / f"case{row['index']:02d}" / "case.json")[
        "measurement"
    ]
    fixtures.append(
        dict(
            dataset=row["fixture"]["dataset"],
            seed=row["fixture"]["seed"],
            checkpoint=c["checkpoint"]["path"],
            checkpoint_sha256=c["checkpoint"]["sha256"],
            model_hash=c["final_model_hash"],
            optimizer_hash=c["final_optimizer_hash"],
            sampler_hash=c["final_sampler_hash"],
        )
    )
p = dict(
    study="H157",
    fixtures=fixtures,
    arms=["native0", "buffer", "identity", "offload4", "fp16", "native1"],
    datasets=base["datasets"],
    training_updates=0,
    gpu_backwards=36,
    diagnostic_forwards=6,
    maintained_files=base["maintained_files"],
    input_hashes={
        **base["input_hashes"],
        **{f["checkpoint"]: f["checkpoint_sha256"] for f in fixtures},
    },
    sources={
        f.as_posix(): sha(f) for f in [*ROOT.glob("*.py"), Path("research/checkpoint_fp16_plan.md")]
    },
    prior_receipt=sha("results/batch_scale_long_v1/receipt.json"),
    free_disk_bytes=shutil.disk_usage(".").free,
)
assert p["free_disk_bytes"] > 6 * 2**30
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
write_json(ROOT / "protocol.json", p)
source = torch.load(fixtures[0]["checkpoint"], map_location="cpu", weights_only=True)
model = Transformer(ModelConfig(**source["model_config"]), 101)
model.load_state_dict(source["model"])
model.set_recompute_scope("block")
block = model.blocks[0]
ref = copy.deepcopy(block)
ref.recompute_scope = "none"
gen = torch.Generator().manual_seed(157)
x = torch.randn(2, 8, 384, generator=gen).requires_grad_()
g = torch.randn(2, 8, 384, generator=gen)
xq = x.detach().half().float().requires_grad_()
with compress_inputs(model, half=True, allow_cpu=True) as logs:
    y = block(x)
    grads = torch.autograd.grad((y * g).sum(), (x, *block.parameters()))
with torch.no_grad():
    torch.testing.assert_close(y, ref(x), rtol=0, atol=0)
gref = torch.autograd.grad((ref(xq) * g).sum(), (xq, *ref.parameters()))
errors = [
    ((a - b).double().norm() / b.double().norm().clamp_min(1e-30)).item()
    for a, b in zip(grads, gref, strict=True)
]
assert max(errors) <= 1e-6
assert len(logs["records"]) == logs["unpack_count"][0] == 1
assert all("forward" not in b.__dict__ for b in model.blocks)
assert torch.isinf(torch.tensor([1e5], dtype=torch.float32).half()).all()
assert not torch.cuda.is_initialized()
write_json(
    ROOT / "checks.json",
    dict(
        passed=True,
        cpu_backwards=2,
        local_vjp_relative_errors=errors,
        forward_unchanged=True,
        overflow_demonstrated=True,
    ),
)
print(
    "Local quantized-input VJP and untouched-forward checks passed; overflow limitation demonstrated."
)
