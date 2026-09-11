"""Balanced timing without phase inventory barriers; passive GPU sampling."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import (
    torch,
    construct,
    unchanged,
    boundary,
    training_loss,
    tensor_hash,
    save_gradient,
    hashes,
    read,
    write_json,
)
from results.checkpoint_input_offload_v1.source.offload import install
from results.blas_workspace_mid_v1.common import configure
from results.blas_workspace_v1.common import warm
from results.native_buffer_layout_v1.operator import model_loss
import datetime
import gc
import subprocess
import time
from pathlib import Path

ROOT = Path("results/paired_workspace_timing_v1")
POLICY = "fp32_default_native"


def signature(model):
    result = {}
    for name, par in model.named_parameters():
        assert par.grad is not None and bool(torch.isfinite(par.grad).all())
        result[name] = tensor_hash(par.grad)
    return result


def setup(model, optimizer, data, mode):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    settings = configure(mode)
    score = warm(model, data)
    torch.cuda.synchronize()
    return settings, score


def reference(f, p, mode):
    configure(mode)
    model, optimizer, data, x, y, provenance = construct(f, p)
    restore = install(model, True)
    settings, score = setup(model, optimizer, data, mode)
    loss = training_loss(model, x, y, POLICY)
    loss.backward()
    torch.cuda.synchronize()
    item = dict(
        dataset=f["dataset"],
        mode=mode,
        settings=settings,
        evaluation=score,
        loss=float(loss.detach()),
        gradients=signature(model),
        provenance=provenance,
        artifact=save_gradient(ROOT / (f["dataset"] + "_" + mode + "_reference.pt"), model),
    )
    unchanged(model, optimizer, data, provenance)
    item["unchanged"] = True
    restore()
    return item


def corpus(f, p, refs):
    configure("low")
    model, optimizer, data, x, y, provenance = construct(f, p)
    restore = install(model, True)
    rows = []
    for control in p["controls"]:
        for cycle, order in enumerate(p["cycles"]):
            for position, letter in enumerate(order):
                arm = "reuse8" if letter == "A" else control
                mode = "high" if arm == "reuse32" else "low"
                settings, score = setup(model, optimizer, data, mode)
                fn = training_loss if arm == "native8" else model_loss
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                probes = []
                for rep in range(p["warmup"] + p["measured"]):
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    at_ns = time.time_ns()
                    cpu = time.process_time_ns()
                    wall = time.perf_counter_ns()
                    start.record()
                    loss = fn(model, x, y, POLICY)
                    loss.backward()
                    end.record()
                    end.synchronize()
                    wall_ms = (time.perf_counter_ns() - wall) / 1e6
                    cpu_ms = (time.process_time_ns() - cpu) / 1e6
                    until_ns = time.time_ns()
                    probes.append(
                        dict(
                            rep=rep,
                            measured=rep >= p["warmup"],
                            at_ns=at_ns,
                            until_ns=until_ns,
                            event_ms=start.elapsed_time(end),
                            wall_ms=wall_ms,
                            cpu_ms=cpu_ms,
                        )
                    )
                grad = signature(model)
                ref = refs[mode]
                assert grad == ref["gradients"] and float(loss.detach()) == ref["loss"]
                assert score == ref["evaluation"]
                for key in (
                    "model_hash",
                    "moments_hash",
                    "optimizer_hash",
                    "sampler_before",
                    "sampler_after",
                    "tokens_hash",
                    "targets_hash",
                ):
                    assert provenance[key] == ref["provenance"][key]
                item = dict(
                    dataset=f["dataset"],
                    control=control,
                    cycle=cycle,
                    position=position,
                    letter=letter,
                    arm=arm,
                    settings=settings,
                    evaluation=score,
                    loss=float(loss.detach()),
                    gradients=grad,
                    exact=True,
                    probes=probes,
                )
                rows.append(item)
                with (ROOT / "bursts.jsonl").open("a", encoding="utf-8") as out:
                    import json

                    out.write(json.dumps(item) + "\n")
                print(f"{f['dataset']} {control} {cycle}:{position} {arm} exact", flush=True)
    unchanged(model, optimizer, data, provenance)
    restore()
    return dict(dataset=f["dataset"], rows=rows, provenance=provenance, unchanged=True)


def run():
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    for values in p["base_verified"].values():
        hashes(values)
    assert not (ROOT / "bursts.jsonl").exists()
    fields = [
        "timestamp",
        "index",
        "uuid",
        "pstate",
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "power.limit",
        "utilization.gpu",
    ]
    results, references, boundaries = [], [], [boundary()]
    anchors = dict(
        start_ns=time.time_ns(),
        timezone=str(datetime.datetime.now().astimezone().tzinfo),
        fields=fields,
    )
    with (ROOT / "telemetry.csv").open("x") as log, (ROOT / "telemetry.stderr").open("x") as err:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
                "-lms",
                "200",
            ],
            stdout=log,
            stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            for f in p["fixtures"]:
                refs = {}
                for mode in ("low", "high"):
                    refs[mode] = reference(f, p, mode)
                    references.append(refs[mode])
                    write_json(ROOT / "references.json", references)
                    gc.collect()
                    boundaries.append(boundary())
                results.append(corpus(f, p, refs))
                gc.collect()
                boundaries.append(boundary())
            assert monitor.poll() is None, "telemetry process exited unexpectedly"
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            anchors.update(end_ns=time.time_ns(), monitor_stopped=monitor.poll() is not None)
            write_json(ROOT / "telemetry_metadata.json", anchors)
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    write_json(
        ROOT / "result.json",
        dict(
            corpora=results,
            references=references,
            boundaries=boundaries,
            backwards=260,
            updates=0,
            evaluations=36,
        ),
    )
    print("H133 scientific pass finished; awaiting independent CPU audit", flush=True)


if __name__ == "__main__":
    run()
