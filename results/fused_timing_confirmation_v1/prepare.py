"""Freeze the paired confirmation, including all source model/optimizer states."""

from pathlib import Path

from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json

ROOT = Path("results/fused_timing_confirmation_v1")
assert not (ROOT / "protocol.json").exists()
hashes(read("results/fused_reconstruction_v1/receipt.json")["files"])
base = read("results/fused_reconstruction_v1/protocol.json")
arms = ["full_gelu:checkpoint", "full_swiglu:checkpoint", "learned:fused"]
cases = [
    c for c in read("results/fused_reconstruction_v1/result.json")["cases"] if c["arm"] in arms
]
assert len(cases) == 18
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
files = [*ROOT.glob("*.py"), Path("research/fused_timing_confirmation_plan.md")]
sources = {
    **base["sources"],
    **{f.as_posix(): sha(f) for f in files},
    **{c["state_path"]: c["state_sha256"] for c in cases},
}
write_json(
    ROOT / "protocol.json",
    dict(
        study="H150",
        arms=arms,
        batches=base["batches"],
        seeds=base["seeds"],
        counts={arm: base["counts"][arm] for arm in arms},
        source_cases=cases,
        training_updates=1080,
        maintained_files=base["maintained_files"],
        prior_receipt=sha("results/fused_reconstruction_v1/receipt.json"),
        sources=sources,
    ),
)
print("Frozen mirrored-order confirmation:18 saved states,36 segments,1080 updates.")
