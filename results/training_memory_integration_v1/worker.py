"""Replay frozen and maintained one-step paths at every qualified source state."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import (
    torch,
    boundary,
    construct,
    host_stats,
    tensor_hash,
    tree_hash,
)
from results.fp32_classifier_profile_v1.source.common import cpu_tree
from results.ordinary_complete_training_v1.audit import distances
from results.partial_offload_training_v1.offload import install_subset
from results.native_buffer_layout_v1.operator import model_loss
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from src.core.training_memory import buffer_model_loss, offload_checkpoint_inputs
from contextlib import contextmanager
from pathlib import Path
import os

ROOT = Path("results/training_memory_integration_v1")


@contextmanager
def frozen_context(model):
    restore, selected = install_subset(model, 4)
    assert selected == [4, 5, 6, 7]
    try:
        yield
    finally:
        restore()


def one(f, p, arm, index):
    model, opt, data, x, y, provenance = construct(f, p)
    ctx = frozen_context(model) if arm == "frozen" else offload_checkpoint_inputs(model, 4)
    with ctx:
        assert [i for i, b in enumerate(model.blocks) if "forward" in b.__dict__] == [4, 5, 6, 7]
        loss = (
            model_loss(model, x, y, "fp32_default_native")
            if arm == "frozen"
            else buffer_model_loss(model, x, y)
        )
        loss.backward()
        raw = {k: v.grad.detach().cpu() for k, v in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        clipped = {k: v.grad.detach().cpu() for k, v in model.named_parameters()}
        opt.step()
    assert all("forward" not in b.__dict__ for b in model.blocks)
    assert provenance["parameter_ids"] == {k: id(v) for k, v in model.named_parameters()}
    assert provenance["state_keys"] == list(model.state_dict())
    assert tensor_hash(data.generator.get_state()) == provenance["sampler_after"]
    assert all(float(s["step"]) == 801 for s in opt.state.values())
    parameters = cpu_tree(model.state_dict())
    moments = {
        f"{i}.{key}": s[key].detach().cpu()
        for i, s in opt.state_dict()["state"].items()
        for key in ("exp_avg", "exp_avg_sq")
    }
    for group in (raw, clipped, parameters, moments):
        assert all(bool(torch.isfinite(v).all()) for v in group.values())
    torch.cuda.synchronize()
    assert bool(torch.isfinite(loss))
    output = dict(raw=raw, clipped=clipped, parameters=parameters, moments=moments)
    path = ROOT / f"case{index:02d}.pt"
    assert not path.exists()
    torch.save(output, path)
    result = dict(
        arm=arm,
        index=index,
        dataset=f["dataset"],
        seed=f["seed"],
        loss=loss.item(),
        peak_bytes=torch.cuda.max_memory_allocated(),
        host_peak_bytes=host_stats()["allocated_bytes.peak"],
        artifact=dict(path=path.as_posix(), sha256=sha(path), tree_hash=tree_hash(output)),
        provenance={k: v for k, v in provenance.items() if k != "parameter_ids"},
        restored=True,
    )
    write_json(ROOT / f"case{index:02d}.json", result)
    return result


def run():
    p = read(ROOT / "protocol.json")
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    assert torch.backends.cuda.cublas_workspace_size() == 8.125 * 2**20
    boundaries = [boundary()]
    cases = []
    checks = []
    for j, f in enumerate(p["fixtures"]):
        order = ("frozen", "maintained") if j % 2 == 0 else ("maintained", "frozen")
        peers = {}
        for arm in order:
            row = one(f, p, arm, len(cases))
            cases.append(row)
            peers[arm] = row
            boundaries.append(boundary())
            write_json(ROOT / "progress.json", dict(cases=cases, boundaries=boundaries))
            print("Completed", row["index"], arm, flush=True)
        old, new = peers["frozen"], peers["maintained"]
        assert old["provenance"] == new["provenance"]
        left, right = [
            torch.load(v["artifact"]["path"], map_location="cpu", weights_only=True)
            for v in (old, new)
        ]
        errors = {key: distances(left[key], right[key]) for key in left}
        bitwise = {key: tree_hash(left[key]) == tree_hash(right[key]) for key in left}
        loss_error = abs(old["loss"] - new["loss"]) / max(abs(old["loss"]), 1e-30)
        gates = dict(
            gradients=all(
                errors[k]["global_relative"] <= 1e-5 and errors[k]["tensor_relative"] <= 1e-4
                for k in ("raw", "clipped")
            ),
            state=all(errors[k]["global_relative"] <= 1e-6 for k in ("parameters", "moments")),
            loss=loss_error <= 1e-6,
            memory=new["peak_bytes"] <= old["peak_bytes"] + 2**20,
        )
        checks.append(
            dict(
                dataset=f["dataset"],
                seed=f["seed"],
                errors=errors,
                bitwise=bitwise,
                loss_relative_error=loss_error,
                peak_difference_bytes=new["peak_bytes"] - old["peak_bytes"],
                gates=gates,
                passed=all(gates.values()),
            )
        )
        del left, right
    assert len(cases) == 12 and len(boundaries) == 13
    for field in ("sources", "input_hashes", "maintained_files"):
        hashes(p[field])
    write_json(
        ROOT / "result.json",
        dict(
            passed=all(c["passed"] for c in checks),
            cases=cases,
            checks=checks,
            boundaries=boundaries,
            training_updates=12,
            backwards=12,
            training_targets=49152,
        ),
    )
    print("Integration replay:", all(c["passed"] for c in checks), flush=True)


if __name__ == "__main__":
    run()
