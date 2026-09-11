"""Independent no-timer replay and NumPy symmetric gradient comparisons."""

# Runtime preloading in common must precede the historical audit imports.
# ruff: noqa: I001

from results.checkpoint_input_offload_v1.source.common import (
    ROOT,
    Ledger,
    attention_context,
    boundary,
    check_inputs,
    construct,
    hashes,
    read,
    save_gradient,
    torch,
    training_loss,
    tree_hash,
    unchanged,
    write_json,
)
from results.checkpoint_input_offload_v1.source.offload import install
from results.adam_update_sensitivity_v1.source.audit import arrays, error

import gc
import math


def load(info):
    value = torch.load(info["path"], map_location="cpu", weights_only=True)
    assert tree_hash(value) == info["tree_hash"]
    return arrays(value)


def compare(a, b, loss_a, loss_b, p):
    assert a.keys() == b.keys()
    global_error = error(a, b)["distance"]
    tensor_error = max(error({k: a[k]}, {k: b[k]})["distance"] for k in a)
    loss_error = abs(loss_a - loss_b) / max(abs(loss_a), abs(loss_b), 1e-12)
    assert all(math.isfinite(v) for v in (global_error, tensor_error, loss_error))
    return dict(
        global_relative=global_error,
        max_tensor_relative=tensor_error,
        loss_relative=loss_error,
        passed=global_error <= p["gradient_global_tolerance"]
        and tensor_error <= p["gradient_tensor_tolerance"]
        and loss_error <= p["loss_tolerance"],
    )


def replay(case, p):
    model, optimizer, data, x, y, initial = construct(case["fixture"], p)
    for key, value in case["provenance"].items():
        assert initial[key] == value
    ledger = Ledger(model, optimizer, data)
    ledger.mark("construction", [x, y])
    restore = install(model, case["mode"] == "offload")
    model.zero_grad(set_to_none=True)
    loss = training_loss(model, x, y, case["fixture"]["loss_policy"])
    ledger.mark("forward", [x, y, loss])
    with attention_context(case["fixture"]["loss_policy"]):
        loss.backward()
    ledger.mark("backward", [x, y, loss])
    scalar = loss.item()
    artifact = save_gradient(ROOT / "runs" / case["label"] / "replay.pt", model)
    ledger.mark("gradient_serialization", [x, y, loss])
    restore()
    del loss
    model.zero_grad(set_to_none=True)
    gc.collect()
    unchanged(model, optimizer, data, initial)
    ledger.mark("release_and_state_verification", [x, y])
    result = dict(
        label=case["label"],
        loss=scalar,
        gradient=artifact,
        phases=ledger.records,
        state_unchanged=True,
        backwards=1,
        training_updates=0,
    )
    write_json(ROOT / "runs" / case["label"] / "replay.json", result)
    print(case["label"], "independent replay complete", flush=True)
    return result


def run():
    assert not (ROOT / "audit.json").exists()
    p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    check_inputs(p)
    hashes(read(ROOT / "audit_protocol.json")["files"])
    boundaries, replays = [boundary()], []
    for c in r["cases"]:
        replays.append(replay(c, p))
        boundaries.append(boundary())
    comparisons = []
    for c in r["cases"]:
        rep = next(v for v in replays if v["label"] == c["label"])
        native = next(
            v for v in r["cases"] if v["fixture"] == c["fixture"] and v["mode"] == "native"
        )
        reference, reference_loss = load(c["untraced"]), c["repetitions"][-1]["loss"]
        for kind, info, scalar in (
            ("trace", c["traced"]["gradient"], c["traced"]["loss"]),
            ("replay", rep["gradient"], rep["loss"]),
            ("native", native["untraced"], native["repetitions"][-1]["loss"]),
        ):
            comparisons.append(
                dict(
                    label=c["label"],
                    kind=kind,
                    **compare(load(info), reference, scalar, reference_loss, p),
                )
            )
    hashes(read(ROOT / "audit_protocol.json")["files"])
    write_json(
        ROOT / "audit.json",
        dict(
            passed=all(v["passed"] for v in comparisons),
            comparisons=comparisons,
            replays=replays,
            boundaries=boundaries,
            backwards=8,
            training_updates=0,
            diagnostic_targets=8 * 4096,
            validation_scores=0,
            broad_goal_achieved=False,
        ),
    )
    print("Independent numerical replay passed:", all(v["passed"] for v in comparisons), flush=True)


if __name__ == "__main__":
    run()
