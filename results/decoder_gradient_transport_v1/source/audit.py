"""Independent native-forward recapture and saved-tensor arithmetic; no backwards."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

from results.decoder_gradient_transport_v1.source.common import (  # noqa: E402
    MODES,
    attention_context,
)
from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402
    clear_boundary,
    tree_hash,
)
from src.core.config import ModelConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.reproducibility import sha256, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402

ROOT = Path("results/decoder_gradient_transport_v1")


def independent_error(actual, reference):
    per, error_sum, reference_sum = {}, 0.0, 0.0
    for name in reference:
        difference = actual[name].double() - reference[name].double()
        e = difference.square().sum().item()
        n = reference[name].double().square().sum().item()
        per[name] = math.sqrt(e) / max(math.sqrt(n), 1e-8)
        error_sum += e
        reference_sum += n
    return dict(
        global_relative_l2=math.sqrt(error_sum) / max(math.sqrt(reference_sum), 1e-12),
        per_tensor_relative_l2=per,
        max_tensor_relative_l2=max(per.values()),
    )


def compare_error(actual, expected):
    for key in ("global_relative_l2", "max_tensor_relative_l2"):
        assert math.isclose(actual[key], expected[key], abs_tol=1e-12, rel_tol=1e-10), key
    for key, value in actual["per_tensor_relative_l2"].items():
        assert math.isclose(
            value, expected["per_tensor_relative_l2"][key], abs_tol=1e-12, rel_tol=1e-10
        ), key


def audit_condition(case, protocol):
    fixture, mode = case["fixture"], case["mode"]
    artifacts = {Path(p["path"]).name: p["path"] for p in case["artifacts"]}
    inputs = torch.load(artifacts["inputs.pt"], map_location="cpu", weights_only=True)
    first = torch.load(artifacts["first_passes.pt"], map_location="cpu", weights_only=True)
    source = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    assert tree_hash(source["model"]) == case["initial_model_hash"]
    model = Transformer(ModelConfig(**fixture["model_config"]), fixture["seed"]).cuda()
    model.load_state_dict(source["model"])
    model.set_recompute_scope(MODES[mode][2])
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 40000
    )
    x, y = data.batch(fixture["batch"], fixture["context"])
    assert torch.equal(x.cpu(), inputs["tokens"]) and torch.equal(y.cpu(), inputs["targets"])
    captured = {}

    def hook(module, args, output):
        captured["hidden"] = output.detach().cpu()

    handle = model.norm.register_forward_hook(hook)
    with (
        attention_context(mode),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=MODES[mode][0] == "bf16"),
    ):
        logits = model(x)
    handle.remove()
    native_recapture_equal = torch.equal(captured["hidden"], inputs["hidden"])
    drift = independent_error({"h": captured["hidden"]}, {"h": inputs["hidden"]})[
        "global_relative_l2"
    ]
    assert torch.isfinite(logits).all()
    del logits
    first_pair = case["paired_repetitions"][0]
    pair_values = {}
    for kind in ("decoder", "total"):
        recomputed = independent_error(first["chunks_fp32"][kind], first["native_fp32"][kind])
        compare_error(recomputed, first_pair[kind])
        pair_values[kind] = recomputed
    for name in first["native_fp32"]["trace"]:
        actual = first["chunks_fp32"]["trace"][name]
        reference = first["native_fp32"]["trace"][name]
        error = independent_error({"g": actual}, {"g": reference})["global_relative_l2"]
        assert math.isclose(
            error, first_pair["trace"][name]["relative_l2"], abs_tol=1e-12, rel_tol=1e-10
        )
    for kind, recorded in (
        ("hidden_gradient", "hidden_gradient"),
        ("weight_gradient", "weight_gradient"),
    ):
        actual = inputs["classifiers"]["chunks_fp32"][kind]
        reference = inputs["classifiers"]["native_fp32"][kind]
        error = independent_error({"g": actual}, {"g": reference})["global_relative_l2"]
        assert math.isclose(
            error,
            case["classifier_difference"][recorded]["relative_l2"],
            abs_tol=1e-12,
            rel_tol=1e-10,
        )
    for policy in ("native_fp32", "chunks_fp32"):
        value = first[policy]
        record = next(r for r in case["records"] if r["repetition"] == 0 and r["policy"] == policy)
        for kind in ("decoder", "total", "trace"):
            assert tree_hash(value[kind]) == record[kind + "_hash"]
        expected_embedding = (
            value["decoder"]["embedding.weight"] + inputs["classifiers"][policy]["weight_gradient"]
        )
        assert torch.equal(expected_embedding, value["total"]["embedding.weight"])
        for name in value["decoder"]:
            if name != "embedding.weight":
                assert torch.equal(value["decoder"][name], value["total"][name])
    print(
        json.dumps(dict(audited=case["label"], native_forward_equal=native_recapture_equal)),
        flush=True,
    )
    return dict(
        label=case["label"],
        native_forward_equal=native_recapture_equal,
        forward_relative_drift=drift,
        paired_errors=pair_values,
        arithmetic_verified=True,
        initial_gradient_hashes_verified=True,
    )


def run():
    frozen = json.loads((ROOT / "audit_protocol.json").read_text())
    for path, digest in frozen["files"].items():
        assert sha256(Path(path)) == digest, path
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for field in ("sources", "input_hashes"):
        for path, digest in protocol[field].items():
            assert sha256(Path(path)) == digest, path
    result = json.loads((ROOT / "result.json").read_text())
    rows, boundaries = [], [clear_boundary()]
    for case in result["conditions"]:
        rows.append(audit_condition(case, protocol))
        boundaries.append(clear_boundary())
    write_json(
        ROOT / "audit.json",
        dict(
            passed=all(r["native_forward_equal"] for r in rows),
            native_forward_recaptures=30,
            scalar_tensor_arithmetic_conditions=30,
            source_model_checks=30,
            saved_first_gradient_sets=60,
            tensor_artifacts=60,
            backward_passes=0,
            optimizer_updates=0,
            no_claim_of_new_gradient_replay=True,
            rows=rows,
            boundaries=boundaries,
        ),
    )


if __name__ == "__main__":
    run()
