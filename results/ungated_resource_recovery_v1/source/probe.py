"""H075 construction-only import probes; no forward/backward/update operations."""

import argparse
import json
from pathlib import Path

import torch

from results.ungated_resource_v1.source import study as original
from src.core.native_recompute_audit import digest, finite_tree, read, sha, write_new
from src.core.optimization import group_summary, parameter_groups
from src.core.reproducibility import environment

ROOT = Path("results/ungated_resource_recovery_v1")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("probe", choices=("cpu_small", "full_1", "full_2"))
name = parser.parse_args().probe
protocol = read(ROOT / "protocol.json")
assert all(sha(n) == h for n, h in protocol["sources"].items())
assert environment() == protocol["environment"]
torch.set_num_threads(4)
torch.manual_seed(17)
torch.cuda.manual_seed_all(17)
torch.backends.cuda.matmul.allow_tf32 = False
if name == "cpu_small":
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.0012, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    assert not optimizer.state and finite_tree(parameter)
    evidence = {"parameters": 2, "device": "cpu", "empty_optimizer_state": True}
else:
    model, mc, tc = original.make_model("gelu_same")
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc), lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    reference = read("results/ungated_resource_v1/workers/gelu_same_none/initial_signature.json")
    assert digest(model.state_dict()) == reference["weights"]
    assert digest(optimizer.state_dict()) == reference["optimizer"]
    assert group_summary(optimizer.param_groups) == reference["groups"]
    assert not optimizer.state and finite_tree(model.state_dict())
    evidence = {
        "parameters": mc.total_parameters,
        "device": "cuda",
        "empty_optimizer_state": True,
        "initial_weights": digest(model.state_dict()),
        "initial_optimizer": digest(optimizer.state_dict()),
        "optimizer_groups": group_summary(optimizer.param_groups),
        "matches_h074_initial_state": True,
    }
assert all(sha(n) == h for n, h in protocol["sources"].items())
write_new(
    ROOT / "probes" / (name + ".json"),
    {
        "status": "PASS",
        "probe": name,
        "evidence": evidence,
        "forwards": 0,
        "backwards": 0,
        "optimizer_updates": 0,
        "corpus_targets": 0,
        "environment": environment(),
        "source_unchanged": True,
        "cause_identified": False,
    },
)
print(
    json.dumps(
        {"status": "PASS", "probe": name, "optimizer_updates": 0, "cause_identified": False}
    ),
    flush=True,
)
