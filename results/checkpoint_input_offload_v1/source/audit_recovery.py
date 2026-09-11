"""Resume the unchanged independent audit after a recorded pre-execution crash."""

from results.checkpoint_input_offload_v1.source.prepare import ROOT, hashes, read

hashes(read(ROOT / "recovery_protocol.json")["files"])
assert not list((ROOT / "runs").glob("*/replay.*"))

from results.checkpoint_input_offload_v1.source.audit import run  # noqa: E402

run()
