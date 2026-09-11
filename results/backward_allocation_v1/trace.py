"""Trace unchanged chunked-loss execution, then compare audited H121 gradients."""

# Preload the established runtime before other tensor imports.
# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import (
    torch,
    boundary,
    check_inputs,
    construct,
    hashes,
    read,
    save_gradient,
    training_loss,
    unchanged,
    write_json,
)
from results.checkpoint_input_offload_v1.source.offload import install
from results.adam_update_sensitivity_v1.source.audit import arrays, error

import gc
import gzip
import json
import types
from pathlib import Path

ROOT = Path("results/backward_allocation_v1")


def snapshot(path):
    value = torch.cuda.memory._snapshot()
    path.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
    return value


def run_case(f, mode, p, prior, reference):
    label = f["label"] + "__" + mode
    folder = ROOT / "runs" / label
    folder.mkdir(exist_ok=False)
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=p["max_entries"],
        clear_history=True,
    )
    model, optimizer, data, x, y, initial = construct(f, prior)
    restore = install(model, mode == "offload")
    phases = []

    def mark(name):
        torch.cuda.synchronize()
        phases.append(
            dict(
                name=name,
                allocated=torch.cuda.memory_allocated(),
                peak=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
            )
        )

    mark("construction")
    for i in range(2):
        model.zero_grad(set_to_none=True)
        loss = training_loss(model, x, y, f["loss_policy"])
        loss.backward()
        mark("warmup_" + str(i))
        del loss
    model.zero_grad(set_to_none=True)
    gc.collect()
    mark("warmup_release")
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=p["max_entries"],
        clear_history=True,
    )
    initial_snapshot = snapshot(folder / "initial.json.gz")
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    markers, backward = [], False

    def marker(name):
        # Snapshot metadata is CPU-only; creates an allocator history marker.
        current = torch.cuda.memory._snapshot()
        markers.append(
            dict(
                name=name,
                index=len(current["device_traces"][0]) - 1,
                allocated=torch.cuda.memory_allocated(),
            )
        )

    for i, block in enumerate(model.blocks):
        original = block._forward

        def observed(this, hidden, fn=original, index=i):
            if backward:
                marker(f"block_{index}_enter")
            try:
                return fn(hidden)
            finally:
                if backward:
                    marker(f"block_{index}_exit")

        block._forward = types.MethodType(observed, block)
    marker("forward_start")
    loss = training_loss(model, x, y, f["loss_policy"])
    marker("forward_end")
    backward = True
    loss.backward()
    torch.cuda.synchronize()
    marker("backward_end")
    peak, current = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
    final_snapshot = snapshot(folder / "final.json.gz")
    torch.cuda.memory._record_memory_history(enabled=None)
    assert len(final_snapshot["device_traces"][0]) < p["max_entries"]
    mark("traced_backward")
    gradient = save_gradient(folder / "gradient.pt", model)
    mark("serialization")
    expected = next(c for c in reference["cases"] if c["label"] == label)
    actual = arrays(torch.load(gradient["path"], map_location="cpu", weights_only=True))
    target = arrays(torch.load(expected["untraced"]["path"], map_location="cpu", weights_only=True))
    global_error = error(actual, target)["distance"]
    max_tensor = max(error({k: actual[k]}, {k: target[k]})["distance"] for k in actual)
    loss_error = abs(loss.item() / expected["repetitions"][-1]["loss"] - 1)
    assert global_error <= 1e-5 and max_tensor <= 1e-4 and loss_error <= 1e-6
    for block in model.blocks:
        del block._forward
    restore()
    del loss
    model.zero_grad(set_to_none=True)
    gc.collect()
    unchanged(model, optimizer, data, initial)
    mark("state_verification")
    output = dict(
        label=label,
        mode=mode,
        dataset=f["dataset"],
        baseline=baseline,
        peak=peak,
        final_allocated=current,
        markers=markers,
        phases=phases,
        gradient=gradient,
        gradient_global=global_error,
        gradient_max_tensor=max_tensor,
        loss_error=loss_error,
        backwards=3,
        training_updates=0,
        state_unchanged=True,
        h121_peak=expected["peak_allocated_bytes"],
        events=len(final_snapshot["device_traces"][0]),
    )
    write_json(folder / "result.json", output)
    print(label, "complete", flush=True)
    del initial_snapshot, final_snapshot
    return output


def run():
    assert not (ROOT / "result.json").exists()
    p = read(ROOT / "protocol.json")
    for field in ("sources", "inputs", "maintained_files"):
        hashes(p[field])
    prior, reference = [
        read("results/checkpoint_input_offload_v1/" + n) for n in ("protocol.json", "result.json")
    ]
    check_inputs(prior)
    cases, boundaries = [], [boundary()]
    for f in p["fixtures"]:
        for mode in f["mode_order"]:
            cases.append(run_case(f, mode, p, prior, reference))
            boundaries.append(boundary())
    write_json(
        ROOT / "result.json",
        dict(
            cases=cases,
            boundaries=boundaries,
            backwards=12,
            training_updates=0,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    run()
