"""Native forward/state/stream audit only; full gradient replay remains failed."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from results.fp32_classifier_profile_v1.source.audit import check_score  # noqa: E402
from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402
    clear_boundary,
    gradient_error,
    tensor_hash,
    tree_hash,
)
from results.streamed_evaluation_v1.source.evaluation import evaluate  # noqa: E402
from src.core.config import ModelConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.reproducibility import sha256, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402

ROOT = Path("results/fp32_classifier_profile_v1")


def check_fixture(fixture, cases, protocol):
    source = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**fixture["model_config"]), fixture["seed"]).cuda()
    model.load_state_dict(source["model"])
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 40000
    )
    initial_sampler = data.generator.get_state()
    initial_score = evaluate(model, data, fixture["batch"], fixture["context"], 10**9, "native")
    saved = {}
    for case in cases:
        gradient = torch.load(case["initial_probe"]["path"], map_location="cpu", weights_only=True)
        assert tree_hash(gradient) == case["initial_probe"]["gradients_hash"]
        assert all(bool(torch.isfinite(g).all()) for g in gradient.values())
        saved[case["policy"]] = gradient
    rows = []
    reference = next(c for c in cases if c["policy"] == "block_fp32")
    for case in cases:
        assert tree_hash(source["model"]) == case["initial_model_hash"]
        assert tree_hash(source["optimizer"]) == case["source_optimizer_hash"]
        assert tensor_hash(initial_sampler) == case["initial_sampler_hash"]
        data.generator.set_state(initial_sampler)
        order = hashlib.sha256()
        for index, record in enumerate(case["records"]):
            x, y = data.batch(fixture["batch"], fixture["context"])
            assert (
                tensor_hash(x) == record["tokens_hash"] and tensor_hash(y) == record["targets_hash"]
            )
            if index == 0:
                assert tensor_hash(x) == case["initial_probe"]["tokens_hash"]
                assert tensor_hash(y) == case["initial_probe"]["targets_hash"]
            order.update(record["tokens_hash"].encode())
        assert order.hexdigest() == case["data_order_hash"]
        assert tensor_hash(data.generator.get_state()) == case["final_sampler_hash"]
        checkpoint = torch.load(case["checkpoint"]["path"], map_location="cpu", weights_only=True)
        assert (
            checkpoint["step"] == fixture["source_step"] + 50
            and checkpoint["continuation_steps"] == 50
        )
        assert checkpoint["model_config"] == fixture["model_config"]
        assert tree_hash(checkpoint["model"]) == case["final_model_hash"]
        assert tree_hash(checkpoint["optimizer"]) == case["final_optimizer_hash"]
        assert tensor_hash(checkpoint["sampler_state"]) == case["final_sampler_hash"]
        assert all(
            float(s["step"]) == checkpoint["step"]
            for s in checkpoint["optimizer"]["state"].values()
        )
        assert all(
            bool(torch.isfinite(v).all())
            for s in checkpoint["optimizer"]["state"].values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )
        model.load_state_dict(checkpoint["model"])
        final_score = evaluate(model, data, fixture["batch"], fixture["context"], 10**9, "native")
        comparison = gradient_error(saved[case["policy"]], saved["block_fp32"])
        comparison["relative_loss_difference"] = abs(
            case["initial_probe"]["loss"] / reference["initial_probe"]["loss"] - 1
        )
        rows.append(
            dict(
                label=case["label"],
                source_native_validation=initial_score,
                final_native_validation=final_score,
                initial_evaluation_relative_drift=check_score(
                    initial_score, case["initial_validation"]
                ),
                final_evaluation_relative_drift=check_score(final_score, case["final_validation"]),
                probe_error_to_native_fp32=comparison,
                saved_gradient_hash_verified=True,
                training_batches_verified=50,
                full_gradient_replay_qualified=False,
            )
        )
    print(
        json.dumps(dict(completed_fixture=fixture["label"], native_scores=5, backwards=0)),
        flush=True,
    )
    return rows


def run():
    for name in (
        "completion_audit_protocol.json",
        "audit_recovery_protocol.json",
        "audit_protocol.json",
    ):
        for path, digest in json.loads((ROOT / name).read_text())["files"].items():
            assert sha256(Path(path)) == digest, path
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    result = json.loads((ROOT / "result.json").read_text())
    rows, boundaries = [], [clear_boundary()]
    for fixture in protocol["fixtures"]:
        cases = [c for c in result["cases"] if c["fixture"]["label"] == fixture["label"]]
        assert len(cases) == 4
        rows.extend(check_fixture(fixture, cases, protocol))
        boundaries.append(clear_boundary())
    write_json(
        ROOT / "completion_audit.json",
        dict(
            passed=True,
            scope="native scores, states, streams and saved gradient integrity",
            fixtures=14,
            cases=56,
            native_validation_scores=70,
            saved_gradient_hashes_verified=56,
            training_batches_verified=2800,
            optimizer_updates=0,
            additional_backward_passes=0,
            full_gradient_replay_qualified=False,
            candidate_advancement_held=True,
            failed_original_backward_count_range=[1, 4],
            diagnosis_backward_passes=6,
            failed_recovery_backward_passes=6,
            maximum_evaluation_relative_drift=max(
                max(r["initial_evaluation_relative_drift"], r["final_evaluation_relative_drift"])
                for r in rows
            ),
            rows=rows,
            boundaries=boundaries,
        ),
    )


if __name__ == "__main__":
    run()
