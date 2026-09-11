"""NumPy checks actual first steps; separate GPU code verifies batches and scoring."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from results.adam_update_sensitivity_v1.source.audit import arrays, error, norm2  # noqa: E402
from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402
    clear_boundary,
    tensor_hash,
    tree_hash,
)
from results.optimizer_memory_v1.source.prepare import ROOT, hashes, read, sha  # noqa: E402
from results.streamed_evaluation_v1.source.evaluation import evaluate  # noqa: E402
from src.core.config import ModelConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.reproducibility import write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def numerical(case, protocol):
    source = load(case["fixture"]["checkpoint"])
    gradient = load(case["first_gradients"]["path"])
    first = load(case["first_state"]["path"])
    final = load(case["final_state"]["path"])
    assert source["step"] == 800 and first["step"] == 801 and final["step"] == 830
    assert tree_hash(gradient["raw"]) == case["first_gradients"]["raw_hash"]
    assert tree_hash(gradient["clipped"]) == case["first_gradients"]["clipped_hash"]
    initial, raw, clipped, actual = [
        arrays(v) for v in (source["model"], gradient["raw"], gradient["clipped"], first["model"])
    ]
    assert initial.keys() == raw.keys() == clipped.keys() == actual.keys()
    norm = math.sqrt(sum(norm2(g) for g in raw.values()))
    clip_error = error(clipped, {k: g * min(1, 1 / (norm + 1e-6)) for k, g in raw.items()})
    reference, expected_moments, actual_moments = {}, [{}, {}], [{}, {}]
    names = []
    for group in case["groups"]:
        opts = group["options"]
        assert opts["lr"] == 0.0006 and opts["eps"] == 1e-8 and opts["betas"] == [0.9, 0.95]
        assert all(opts[k] == v for k, v in protocol["modes"][case["mode"]].items())
        for name, index in zip(group["names"], group["ids"], strict=True):
            names.append(name)
            old, new, end = [s["optimizer"]["state"][index] for s in (source, first, final)]
            assert [float(s["step"]) for s in (old, new, end)] == [800, 801, 830]
            for i, (field, beta) in enumerate(
                zip(("exp_avg", "exp_avg_sq"), opts["betas"], strict=True)
            ):
                prev = old[field].numpy().astype(np.float64)
                g = clipped[name]
                expected_moments[i][name] = beta * prev + (1 - beta) * (
                    g if i == 0 else np.square(g)
                )
                actual_moments[i][name] = new[field].numpy().astype(np.float64)
                assert (
                    np.isfinite(end[field].numpy()).all() and np.isfinite(new[field].numpy()).all()
                )
            corrected_m = expected_moments[0][name] / (1 - opts["betas"][0] ** 801)
            corrected_v = expected_moments[1][name] / (1 - opts["betas"][1] ** 801)
            decay = 0.1 if initial[name].ndim >= 2 else 0.0
            assert decay == opts["weight_decay"]
            reference[name] = initial[name] * (1 - opts["lr"] * decay) - opts[
                "lr"
            ] * corrected_m / (np.sqrt(corrected_v) + opts["eps"])
    assert len(names) == len(set(names)) == len(initial) and set(names) == initial.keys()
    assert all(np.isfinite(t.numpy()).all() for t in final["model"].values())
    assert tree_hash(final["model"]) == case["final_state"]["model_hash"]
    assert tree_hash(final["optimizer"]) == case["final_state"]["optimizer_hash"]
    assert tree_hash(first["model"]) == case["first_state"]["model_hash"]
    assert tree_hash(first["optimizer"]) == case["first_state"]["optimizer_hash"]
    pe = error(actual, reference)
    me = [error(a, b) for a, b in zip(actual_moments, expected_moments, strict=True)]
    passed = (
        pe["distance"] <= protocol["parameter_relative_tolerance"]
        and pe["max_absolute"] <= protocol["parameter_absolute_tolerance"]
        and clip_error["distance"] <= protocol["clip_tolerance"]
        and all(e["distance"] <= protocol["moment_tolerance"] for e in me)
    )
    return dict(
        label=case["label"],
        parameter_error=pe,
        moment_errors=me,
        clip_error=clip_error,
        raw_norm_fp64=norm,
        passed=passed,
    )


def native_score_and_batches(case, protocol):
    fixture = case["fixture"]
    source = load(fixture["checkpoint"])
    final = load(case["final_state"]["path"])
    first = load(case["first_state"]["path"])
    data = TokenData(Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", 10101)
    data.generator.set_state(source["sampler_state"])
    for row in case["updates"]:
        x, y = data.batch(8, 512)
        assert tensor_hash(x) == row["tokens_hash"] and tensor_hash(y) == row["targets_hash"]
        if row["step"] == 1:
            assert tensor_hash(data.generator.get_state()) == tensor_hash(first["sampler_state"])
    assert (
        tensor_hash(data.generator.get_state())
        == tensor_hash(final["sampler_state"])
        == case["final_state"]["sampler_hash"]
    )
    model = Transformer(ModelConfig(**final["model_config"]), 101).cuda()
    model.load_state_dict(final["model"])
    score = evaluate(model, data, 8, 512, 10**9, "native")
    expected = case["after_score"]
    assert (
        score["targets"] == expected["targets"]
        and score["order_sha256"] == expected["order_sha256"]
    )
    discrepancy = abs(score["nll"] / expected["nll"] - 1)
    return dict(
        label=case["label"],
        native_score=score,
        relative_error=discrepancy,
        passed=discrepancy <= protocol["score_tolerance"],
        batches_verified=30,
    )


def run():
    assert not (ROOT / "audit.json").exists()
    p, r = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    for field in ("sources", "maintained_files", "input_hashes", "library_hashes"):
        hashes(p[field])
    hashes(read(ROOT / "audit_protocol.json")["files"])
    rows = []
    for c in r["cases"]:
        for key in ("first_gradients", "first_state", "final_state"):
            assert sha(c[key]["path"]) == c[key]["sha256"]
        rows.append(numerical(c, p))
    gradients = []
    for fixture in p["fixtures"]:
        cases = [c for c in r["cases"] if c["fixture"] == fixture]
        base = next(c for c in cases if c["mode"] == "default")
        reference = arrays(load(base["first_gradients"]["path"])["raw"])
        for c in cases:
            actual = arrays(load(c["first_gradients"]["path"])["raw"])
            distance = error(actual, reference)["distance"]
            maximum = max(error({k: actual[k]}, {k: reference[k]})["distance"] for k in actual)
            gradients.append(
                dict(
                    label=c["label"],
                    global_relative=distance,
                    max_tensor_relative=maximum,
                    passed=distance <= p["gradient_global_tolerance"]
                    and maximum <= p["gradient_tensor_tolerance"],
                )
            )
    del actual, reference
    scores, boundaries = [], [clear_boundary()]
    for case in r["cases"]:
        scores.append(native_score_and_batches(case, p))
        boundaries.append(clear_boundary())
    passed = all(x["passed"] for x in rows + gradients + scores)
    write_json(
        ROOT / "audit.json",
        dict(
            passed=passed,
            numerical=rows,
            gradient_comparisons=gradients,
            scores=scores,
            boundaries=boundaries,
            native_scores=12,
            batches_verified=360,
            tensor_artifacts_verified=36,
            optimizer_updates=0,
            backwards=0,
            broad_goal_achieved=False,
        ),
    )
    print("All independent numerical/data/scoring checks passed:", passed, flush=True)


if __name__ == "__main__":
    run()
