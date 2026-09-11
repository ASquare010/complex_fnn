"""One common no-update probe loop; only checkpoint input storage differs."""

import gc
import math
import statistics as st
import time
import traceback

from results.checkpoint_input_offload_v1.source.common import (
    ROOT,
    Ledger,
    attention_context,
    boundary,
    check_inputs,
    construct,
    describe,
    read,
    save_gradient,
    torch,
    training_loss,
    unchanged,
    write_json,
)
from results.checkpoint_input_offload_v1.source.offload import Trace, install


def probe(model, x, y, policy, ledger, suffix):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    wall = time.perf_counter()
    events[0].record()
    loss = training_loss(model, x, y, policy)
    events[1].record()
    ledger.mark("forward_" + suffix, [x, y, loss])
    events[2].record()
    with attention_context(policy):
        loss.backward()
    events[3].record()
    ledger.mark("backward_" + suffix, [x, y, loss])
    wall = 1000 * (time.perf_counter() - wall)
    phases = dict(
        forward=events[0].elapsed_time(events[1]), backward=events[2].elapsed_time(events[3])
    )
    result = dict(
        loss=loss.item(), wall_ms=wall, event_ms=phases, event_sum_ms=sum(phases.values())
    )
    assert math.isfinite(result["loss"])
    return result  # loss/graph references are released on returning


def run_case(f, mode, p):
    label = f["label"] + "__" + mode
    folder = ROOT / "runs" / label
    folder.mkdir(exist_ok=False)
    model, optimizer, data, x, y, initial = construct(f, p)
    ledger = Ledger(model, optimizer, data)
    ledger.mark("construction", [x, y])
    restore = install(model, mode == "offload")
    rows = []
    for repetition in range(1, p["repetitions"] + 1):
        model.zero_grad(set_to_none=True)
        ledger.mark(f"release_previous_{repetition}", [x, y])
        row = probe(model, x, y, f["loss_policy"], ledger, str(repetition))
        row["repetition"] = repetition
        rows.append(row)
    untraced = save_gradient(folder / "untraced.pt", model)
    ledger.mark("untraced_gradient_serialization", [x, y])
    restore()
    model.zero_grad(set_to_none=True)
    gc.collect()
    ledger.mark("untraced_graph_release", [x, y])
    trace = Trace(mode == "offload")
    restore = install(model, mode == "offload", trace)
    traced = probe(model, x, y, f["loss_policy"], ledger, "trace")
    traced["gradient"] = save_gradient(folder / "traced.pt", model)
    ledger.mark("traced_gradient_serialization", [x, y])
    restore()
    model.zero_grad(set_to_none=True)
    gc.collect()
    traced["trace"] = trace.result()
    ledger.mark("traced_graph_release", [x, y])
    unchanged(model, optimizer, data, initial)
    ledger.mark("unchanged_state_diagnostics", [x, y])
    warm = rows[p["warmup"] :]
    timing = {key: describe([r[key] for r in warm]) for key in ("wall_ms", "event_sum_ms")}
    stability = {
        key: max(st.median(r[key] for r in warm[:10]), st.median(r[key] for r in warm[10:]))
        / min(st.median(r[key] for r in warm[:10]), st.median(r[key] for r in warm[10:]))
        for key in timing
    }
    normal = [r for r in ledger.records if r["phase"].split("_")[-1].isdigit()]
    initial.pop("parameter_ids")
    output = dict(
        label=label,
        fixture=f,
        mode=mode,
        provenance=initial,
        repetitions=rows,
        untraced=untraced,
        traced=traced,
        timing=timing,
        timing_stability=stability,
        phases=ledger.records,
        peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in ledger.records),
        peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in ledger.records),
        normal_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in normal),
        host_allocated_peak=max(r["host"]["allocated_bytes.peak"] for r in ledger.records),
        host_active_peak=max(r["host"]["active_bytes.peak"] for r in ledger.records),
        backwards=31,
        training_updates=0,
        diagnostic_targets=31 * 4096,
        parameters=9099648,
        state_unchanged=True,
        finite=True,
    )
    write_json(folder / "result.json", output)
    print(label, "31 backwards, zero updates", flush=True)
    return output


def run():
    assert not (ROOT / "result.json").exists()
    p = read(ROOT / "protocol.json")
    check_inputs(p)
    assert read(ROOT / "qualification.json")["passed"]
    assert (ROOT / "qualify_exit.txt").read_text().strip() == "0"
    start = time.perf_counter()
    cases, boundaries = [], [boundary()]
    for fixture in p["fixtures"]:
        for mode in fixture["mode_order"]:
            cases.append(run_case(fixture, mode, p))
            boundaries.append(boundary())
    write_json(
        ROOT / "result.json",
        dict(
            status="COMPLETE",
            cases=cases,
            boundaries=boundaries,
            wall_seconds=time.perf_counter() - start,
            backwards=248,
            training_updates=0,
            diagnostic_targets=248 * 4096,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "profile_failure.json", dict(traceback=traceback.format_exc()))
        raise
