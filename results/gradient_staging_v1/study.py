"""One matched probe loop with explicit gradient restoration and no updates."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import (
    torch,
    Ledger,
    boundary,
    check_inputs,
    construct,
    hashes,
    read,
    sha,
    tensor_hash,
    tree_hash,
    training_loss,
    unchanged,
    write_json,
)
from results.checkpoint_input_offload_v1.source.offload import install
from results.adam_update_sensitivity_v1.source.audit import arrays, error
from results.gradient_staging_v1.staging import GradientStager
import copy
import statistics as st
import time
from pathlib import Path

ROOT = Path("results/gradient_staging_v1")


def qualify():
    torch.manual_seed(123)
    base = torch.nn.Module()
    base.register_parameter(
        "weight", torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float64) * 0.1)
    )
    base.register_parameter("bias", torch.nn.Parameter(torch.randn(8, dtype=torch.float64) * 0.1))
    x = torch.randn(3, 8, dtype=torch.float64).cuda()
    reference = None
    for staged in (False, True):
        model = copy.deepcopy(base).cuda()
        store = GradientStager(model) if staged else None
        if store:
            store.begin()
        loss = (
            ((x @ model.weight + model.bias).tanh() + (x @ model.weight - model.bias).tanh())
            .square()
            .mean()
        )
        loss.backward()
        torch.cuda.synchronize()
        if store:
            store.restore(model)
            torch.cuda.synchronize()
        actual = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
        if reference is None:
            reference = actual
        else:
            for k in actual:
                torch.testing.assert_close(actual[k], reference[k], atol=1e-12, rtol=1e-12)
                assert tensor_hash(actual[k]) == tensor_hash(store.buffers[k])
            store.close()
    return dict(passed=True, backwards=2, shared_weight_uses=2)


def run_case(f, mode, p, prior, reference):
    label = f["dataset"] + "__" + mode
    folder = ROOT / "runs" / label
    folder.mkdir(exist_ok=False)
    model, optimizer, data, x, y, initial = construct(f, prior)
    restore_inputs = install(model, True)
    store = GradientStager(model) if mode == "staged" else None
    ledger = Ledger(model, optimizer, data)
    ledger.mark("construction", [x, y])
    rows = []
    for i in range(p["repetitions"]):
        model.zero_grad(set_to_none=True)
        if store:
            store.begin()
        ledger.mark("release_" + str(i), [x, y])
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        start = time.perf_counter()
        events[0].record()
        loss = training_loss(model, x, y, f["loss_policy"])
        events[1].record()
        ledger.mark("forward_" + str(i), [x, y, loss])
        events[2].record()
        loss.backward()
        events[3].record()
        ledger.mark("backward_" + str(i), [x, y, loss])
        grad_bytes_after_backward = sum(
            v.grad.numel() * v.grad.element_size() for v in model.parameters() if v.grad is not None
        )
        if store:
            assert grad_bytes_after_backward == 0
        events[4].record()
        if store:
            store.restore(model)
        events[5].record()
        ledger.mark("restore_" + str(i), [x, y, loss])
        wall = (time.perf_counter() - start) * 1000
        times = {
            name: events[2 * j].elapsed_time(events[2 * j + 1])
            for j, name in enumerate(("forward", "backward", "restore"))
        }
        rows.append(
            dict(
                repetition=i + 1,
                loss=loss.item(),
                wall_ms=wall,
                event_ms=times,
                event_sum_ms=sum(times.values()),
                grad_bytes_after_backward=grad_bytes_after_backward,
                hook_count=len(store.seen) if store else 0,
            )
        )
        del loss
    gradient = {k: v.grad.detach().cpu() for k, v in model.named_parameters()}
    assert all(bool(torch.isfinite(v).all()) for v in gradient.values())
    if store:
        assert all(tensor_hash(v) == tensor_hash(store.buffers[k]) for k, v in gradient.items())
    torch.save(gradient, folder / "gradient.pt")
    ledger.mark("gradient_serialization", [x, y])
    expected = next(
        c
        for c in reference["cases"]
        if c["fixture"]["label"] == f["label"] and c["mode"] == "offload"
    )
    for key in (
        "tokens_hash",
        "targets_hash",
        "model_hash",
        "moments_hash",
        "sampler_before",
        "sampler_after",
    ):
        assert initial[key] == expected["provenance"][key]
    target = arrays(torch.load(expected["untraced"]["path"], map_location="cpu", weights_only=True))
    actual = arrays(gradient)
    ge = error(actual, target)["distance"]
    te = max(error({k: actual[k]}, {k: target[k]})["distance"] for k in actual)
    le = max(abs(row["loss"] / expected["repetitions"][-1]["loss"] - 1) for row in rows)
    assert ge <= 1e-5 and te <= 1e-4 and le <= 1e-6
    unchanged(model, optimizer, data, initial)
    ledger.mark("state_verification", [x, y])
    if store:
        store.close()
    restore_inputs()
    warm = rows[p["warmup"] :]
    timing = {
        k: dict(
            mean=st.mean(v[k] for v in warm),
            median=st.median(v[k] for v in warm),
            variance=st.variance(v[k] for v in warm),
        )
        for k in ("wall_ms", "event_sum_ms")
    }
    result = dict(
        label=label,
        dataset=f["dataset"],
        mode=mode,
        rows=rows,
        timing=timing,
        phases=ledger.records,
        peak=max(v["peak_allocated_bytes"] for v in ledger.records),
        reserved_peak=max(v["peak_reserved_bytes"] for v in ledger.records),
        host_allocated_peak=max(v["host"]["allocated_bytes.peak"] for v in ledger.records),
        host_active_peak=max(v["host"]["active_bytes.peak"] for v in ledger.records),
        gradient=dict(
            path=(folder / "gradient.pt").as_posix(),
            sha256=sha(folder / "gradient.pt"),
            tree_hash=tree_hash(gradient),
        ),
        global_error=ge,
        tensor_error=te,
        loss_error=le,
        backwards=10,
        training_updates=0,
        state_unchanged=True,
        parameters=sum(v.numel() for v in model.parameters()),
        pinned_gradient_payload=sum(v.numel() * v.element_size() for v in store.buffers.values())
        if store
        else 0,
    )
    write_json(folder / "result.json", result)
    print(label, "complete", flush=True)
    return result


def run():
    assert not (ROOT / "result.json").exists()
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    prior, reference = [
        read("results/checkpoint_input_offload_v1/" + n) for n in ("protocol.json", "result.json")
    ]
    check_inputs(prior)
    boundaries = [boundary()]
    qualification = qualify()
    boundaries.append(boundary())
    write_json(ROOT / "qualification.json", qualification)
    cases = []
    for i, f in enumerate(p["fixtures"]):
        for mode in ("resident", "staged") if i == 0 else ("staged", "resident"):
            cases.append(run_case(f, mode, p, prior, reference))
            boundaries.append(boundary())
    write_json(
        ROOT / "result.json",
        dict(
            cases=cases,
            boundaries=boundaries,
            qualification=qualification,
            backwards=42,
            training_updates=0,
            diagnostic_targets=163840,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    run()
