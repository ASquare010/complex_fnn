"""Fresh native gradient replay at each workspace size, including BF16 warmup."""

# ruff: noqa: I001
from results.blas_workspace_v1.common import configure, warm, torch
from results.checkpoint_input_offload_v1.source import common as c
from pathlib import Path
import sys

ROOT = Path("results/blas_workspace_v1")


def one(row, prior):
    model, optimizer, data, x, y, initial = c.construct(row["fixture"], prior)
    score = warm(model, data)
    loss = c.training_loss(model, x, y, "fp32_default_native")
    loss.backward()
    saved = torch.load(
        row["measurement"]["gradient"]["path"], map_location="cpu", weights_only=True
    )
    exact = {
        k: c.tensor_hash(v.grad) == c.tensor_hash(saved[k]) for k, v in model.named_parameters()
    }
    exact_loss = all(loss.item() == x["loss"] for x in row["measurement"]["rows"])
    c.unchanged(model, optimizer, data, initial)
    return dict(
        dataset=row["fixture"]["dataset"],
        arm=row["arm"],
        gradients=exact,
        exact_loss=exact_loss,
        exact_evaluation=score == row["evaluation"],
        state_unchanged=True,
        passed=all(exact.values()) and exact_loss and score == row["evaluation"],
    )


def run():
    mode = sys.argv[1]
    p = c.read(ROOT / "protocol.json")
    for key in ("sources", "input_hashes", "maintained_files"):
        c.hashes(p[key])
    c.hashes(c.read(ROOT / "audit_protocol.json")["files"])
    prior = c.read("results/checkpoint_input_offload_v1/protocol.json")
    c.check_inputs(prior)
    settings = configure(mode)
    boundaries = [c.boundary()]
    rows = []
    for row in c.read(ROOT / (mode + "_result.json"))["cases"]:
        rows.append(one(row, prior))
        boundaries.append(c.boundary())
    c.write_json(
        ROOT / ("audit_" + mode + ".json"),
        dict(
            cases=rows,
            settings=settings,
            boundaries=boundaries,
            backwards=4,
            training_updates=0,
            diagnostic_targets=16384,
            evaluation_targets=16384,
            passed=all(x["passed"] for x in rows),
        ),
    )
    print(mode, "replay", all(x["passed"] for x in rows))


if __name__ == "__main__":
    run()
