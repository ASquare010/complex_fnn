"""Reuse independent state/scoring checks and measure all repeat-pair distances."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from results.fp32_classifier_profile_v1.source.common import clear_boundary, tree_hash  # noqa: E402
from results.fp32_decoder_resource_v1.source.audit import independent_error  # noqa: E402
from results.fp32_training_replication_recovery_v1.source.audit import (  # noqa: E402
    audit_case,
    audit_initial,
)
from src.core.reproducibility import sha256, write_json  # noqa: E402

ROOT = Path("results/training_variability_v1")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def symmetric_distance(left: dict, right: dict) -> float:
    """Global relative L2 on CPU in double precision, symmetric in both arguments."""
    assert left.keys() == right.keys()
    error = scale_a = scale_b = 0.0
    for key in left:
        a, b = left[key].double(), right[key].double()
        assert a.shape == b.shape and a.device.type == b.device.type == "cpu"
        error += (a - b).square().sum().item()
        scale_a += a.square().sum().item()
        scale_b += b.square().sum().item()
    value = math.sqrt(error) / max((math.sqrt(scale_a) + math.sqrt(scale_b)) / 2, 1e-12)
    assert math.isfinite(value)
    return value


def repeat_distances(cases: list[dict]) -> tuple[list, list]:
    a, b = {"x": torch.tensor([3.0, 4.0])}, {"x": torch.zeros(2)}
    assert symmetric_distance(a, a) == symmetric_distance(b, b) == 0
    assert symmetric_distance(a, b) == symmetric_distance(b, a) == 2
    gradients = [
        torch.load(c["initial_probe"]["path"], map_location="cpu", weights_only=True) for c in cases
    ]
    for case, gradient in zip(cases, gradients, strict=True):
        assert tree_hash(gradient) == case["initial_probe"]["gradients_hash"]
    initial_pairs = [
        dict(
            left=cases[i]["label"],
            right=cases[j]["label"],
            same_policy=cases[i]["policy"] == cases[j]["policy"],
            distance=symmetric_distance(gradients[i], gradients[j]),
            bitwise_equal=tree_hash(gradients[i]) == tree_hash(gradients[j]),
        )
        for i, j in itertools.combinations(range(6), 2)
    ]
    del gradients
    checkpoint_pairs = []
    for step in (200, 400, 800):
        loaded = []
        for case in cases:
            item = (
                case["checkpoint"]
                if step == 800
                else next(c for c in case["intermediate_checkpoints"] if c["step"] == step)
            )
            assert sha256(Path(item["path"])) == item["sha256"]
            state = torch.load(item["path"], map_location="cpu", weights_only=True)
            assert state["step"] == step
            parts = dict(model=state["model"])
            for moment in ("exp_avg", "exp_avg_sq"):
                parts[moment] = {k: v[moment] for k, v in state["optimizer"]["state"].items()}
            loaded.append(parts)
        model_hashes = [tree_hash(s["model"]) for s in loaded]
        for i, j in itertools.combinations(range(6), 2):
            checkpoint_pairs.append(
                dict(
                    step=step,
                    left=cases[i]["label"],
                    right=cases[j]["label"],
                    same_policy=cases[i]["policy"] == cases[j]["policy"],
                    model_bitwise_equal=model_hashes[i] == model_hashes[j],
                    **{
                        key: symmetric_distance(loaded[i][key], loaded[j][key])
                        for key in ("model", "exp_avg", "exp_avg_sq")
                    },
                )
            )
        del loaded, state, parts
    return initial_pairs, checkpoint_pairs


def run() -> None:
    protocol, frozen, result = [
        read(ROOT / n) for n in ("protocol.json", "audit_protocol.json", "result.json")
    ]
    for mapping in (protocol["sources"], frozen["files"]):
        for path, digest in mapping.items():
            assert sha256(Path(path)) == digest, path
    previous = read(Path(protocol["previous_root"]) / "result.json")
    originals = [c for c in previous["cases"] if c["label"] in protocol["original_case_labels"]]
    all_cases = originals + result["cases"]
    assert len(all_cases) == 6
    boundaries = [clear_boundary()]
    initial = audit_initial(protocol["original_fixture"], all_cases, protocol)
    boundaries.append(clear_boundary())
    rows = []
    for case in result["cases"]:
        rows.append(audit_case(case, protocol))
        boundaries.append(clear_boundary())
    initial_pairs, checkpoint_pairs = repeat_distances(all_cases)
    paired_errors = []
    for repeat in (0, 1, 2):
        peers = (
            originals
            if repeat == 0
            else [c for c in result["cases"] if c["fixture"]["repetition"] == repeat]
        )
        native = next(c for c in peers if c["policy"] == "fp32_default_native")
        chunk = next(c for c in peers if c["policy"] == "fp32_default_chunks")
        a = torch.load(chunk["initial_probe"]["path"], map_location="cpu", weights_only=True)
        b = torch.load(native["initial_probe"]["path"], map_location="cpu", weights_only=True)
        error = independent_error(a, b)
        loss_error = abs(chunk["initial_probe"]["loss"] / native["initial_probe"]["loss"] - 1)
        paired_errors.append(
            dict(
                repetition=repeat,
                error=error,
                loss_relative_error=loss_error,
                passed=loss_error <= 1e-6
                and error["global_relative_l2"] <= 0.002
                and error["max_tensor_relative_l2"] <= 0.02,
            )
        )
    passed = (
        initial["passed"]
        and all(r["all_scores_passed"] and r["replay"]["passed"] for r in rows)
        and all(p["passed"] for p in paired_errors)
    )
    write_json(
        ROOT / "audit.json",
        dict(
            passed=passed,
            rows=rows,
            initial_native_score=initial,
            paired_initial_errors=paired_errors,
            initial_gradient_pairs=initial_pairs,
            checkpoint_pairs=checkpoint_pairs,
            boundaries=boundaries,
            native_validation_scores=13,
            gradient_replay_backward_passes=4,
            trained_checkpoints_verified=12,
            saved_gradient_hashes_verified=6,
            training_batches_verified=3200,
            regenerated_initializations=1,
            cpu_distance_sanity_checks=4,
            optimizer_updates=0,
            broad_goal_achieved=False,
        ),
    )
    print(
        json.dumps(
            dict(
                audit_passed=passed,
                native_scores=13,
                gradient_replays=4,
                initial_pairs=len(initial_pairs),
                checkpoint_pairs=len(checkpoint_pairs),
            )
        )
    )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "audit_failure.json", dict(traceback=traceback.format_exc()))
        raise
