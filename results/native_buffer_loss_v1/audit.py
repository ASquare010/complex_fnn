"""Replay native arithmetic without the candidate or offloading adapter."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import (
    torch,
    boundary,
    check_inputs,
    construct,
    hashes,
    read,
    training_loss,
    unchanged,
    write_json,
    tensor_hash,
)
from pathlib import Path

ROOT = Path("results/native_buffer_loss_v1")


def one(row, prior):
    model, optimizer, data, x, y, initial = construct(row["fixture"], prior)
    expected = row["measurement"]
    loss = training_loss(model, x, y, row["fixture"]["loss_policy"])
    loss.backward()
    saved = torch.load(expected["gradient"]["path"], map_location="cpu", weights_only=True)
    equality = {
        name: tensor_hash(value.grad) == tensor_hash(saved[name])
        for name, value in model.named_parameters()
    }
    exact_loss = all(loss.item() == v["loss"] for v in expected["rows"])
    unchanged(model, optimizer, data, initial)
    return dict(
        dataset=row["fixture"]["dataset"],
        arm=row["arm"],
        exact_loss=exact_loss,
        exact_gradients=equality,
        state_unchanged=True,
        passed=exact_loss and all(equality.values()),
    )


def run():
    p, r, ap = [read(ROOT / n) for n in ("protocol.json", "result.json", "audit_protocol.json")]
    for key in ("sources", "input_hashes", "maintained_files"):
        hashes(p[key])
    hashes(ap["files"])
    prior = read("results/checkpoint_input_offload_v1/protocol.json")
    check_inputs(prior)
    boundaries = [boundary()]
    rows = []
    for row in r["cases"]:
        result = one(row, prior)
        rows.append(result)
        boundaries.append(boundary())
        print(result["dataset"], result["arm"], result["passed"], flush=True)
    write_json(
        ROOT / "audit.json",
        dict(
            cases=rows,
            boundaries=boundaries,
            backwards=len(rows),
            training_updates=0,
            passed=bool(rows) and all(x["passed"] for x in rows),
        ),
    )


if __name__ == "__main__":
    run()
