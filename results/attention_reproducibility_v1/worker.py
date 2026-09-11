"""Fresh matched probes with traced attention operators and no updates."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import (
    torch,
    boundary,
    construct,
    check_inputs,
    unchanged,
    training_loss,
    read,
    hashes,
    sha,
    write_json,
    environment,
)
from results.checkpoint_input_offload_v1.source.offload import install
from results.native_buffer_layout_v1.operator import model_loss
from results.attention_reproducibility_v1.compare import compare, digest
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext
from pathlib import Path
import os
import sys
import warnings

ROOT = Path("results/attention_reproducibility_v1")
COUNTS = dict(forwards=0, backward_attempts=0, backwards=0)


def one(fixture, prior, mode, arm, repetition):
    model, optimizer, data, x, y, initial = construct(fixture, prior)
    restore = install(model, arm != "native_resident")
    context = sdpa_kernel([SDPBackend.MATH]) if mode == "deterministic_math" else nullcontext()
    try:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            with context, profile(activities=[ProfilerActivity.CPU]) as prof:
                COUNTS["forwards"] += 1
                loss = (model_loss if arm == "reuse_offload" else training_loss)(
                    model, x, y, "fp32_default_native"
                )
                COUNTS["backward_attempts"] += 1
                loss.backward()
                torch.cuda.synchronize()
                COUNTS["backwards"] += 1
        gradient = {k: v.grad.detach().cpu() for k, v in model.named_parameters()}
        assert len(gradient) == 50 and all(bool(torch.isfinite(x).all()) for x in gradient.values())
        unchanged(model, optimizer, data, initial)
        initial.pop("parameter_ids")
        operators = sorted(
            {event.key for event in prof.key_averages() if "scaled_dot_product" in event.key}
        )
        assert operators
        row = dict(
            dataset=fixture["dataset"],
            arm=arm,
            repetition=repetition,
            loss=loss.item(),
            hashes={k: digest(v) for k, v in gradient.items()},
            operators=operators,
            warnings=[str(v.message) for v in emitted],
            provenance=initial,
            finite=True,
            state_unchanged=True,
            parameters=sum(v.numel() for v in model.parameters()),
        )
        return row, gradient
    finally:
        restore()


def run():
    mode = sys.argv[1]
    p = read(ROOT / "protocol.json")
    assert mode in p["policies"]
    for key in ("sources", "input_hashes", "maintained_files"):
        hashes(p[key])
    prior = read("results/checkpoint_input_offload_v1/protocol.json")
    check_inputs(prior)
    deterministic = mode != "original_default"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == (":4096:8" if deterministic else None)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    folder = ROOT / mode
    folder.mkdir(exist_ok=False)
    settings = dict(
        environment=environment(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        threads=torch.get_num_threads(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
    )
    write_json(folder / "environment.json", settings)
    boundaries = [boundary()]
    rows = []
    failure = None
    for index, fixture in enumerate(p["fixtures"]):
        stored = []
        arms = ("native_resident", "native_offload", "reuse_offload")
        if index:
            arms = tuple(reversed(arms))
        for repetition in (0, 1):
            for arm in arms:
                try:
                    row, gradient = one(fixture, prior, mode, arm, repetition)
                except RuntimeError as exc:
                    failure = dict(
                        dataset=fixture["dataset"], arm=arm, repetition=repetition, error=str(exc)
                    )
                    break
                if repetition == 0:
                    path = folder / (fixture["dataset"] + "__" + arm + ".pt")
                    torch.save(gradient, path)
                    row["artifact"] = dict(path=path.as_posix(), sha256=sha(path))
                stored.append((row, gradient))
                boundaries.append(boundary())
                write_json(
                    folder / "progress.json", dict(counts=COUNTS, completed=len(rows) + len(stored))
                )
                print(mode, fixture["dataset"], arm, repetition, "complete", flush=True)
            if failure:
                break
        if failure:
            rows.extend(row for row, _ in stored)
            del stored
            boundaries.append(boundary())
            break
        anchor, reference = next(
            (row, g)
            for row, g in stored
            if row["arm"] == "native_resident" and row["repetition"] == 0
        )
        for row, gradient in stored:
            first = next(v for v, _ in stored if v["arm"] == row["arm"] and v["repetition"] == 0)
            assert row["provenance"] == anchor["provenance"]
            row["comparison"] = compare(gradient, reference)
            row["exact_loss"] = row["loss"] == anchor["loss"]
            row["same_arm_exact"] = (
                row["hashes"] == first["hashes"] and row["loss"] == first["loss"]
            )
            rows.append(row)
        del stored, reference, gradient
    result = dict(
        policy=mode,
        settings=settings,
        cases=rows,
        boundaries=boundaries,
        counts=COUNTS,
        failure=failure,
        training_updates=0,
        diagnostic_targets=COUNTS["backwards"] * 4096,
        passed=failure is None
        and len(rows) == 12
        and all(x["comparison"]["exact"] and x["exact_loss"] and x["same_arm_exact"] for x in rows),
    )
    write_json(folder / "result.json", result)
    print("Policy gate:", result["passed"], flush=True)


if __name__ == "__main__":
    run()
