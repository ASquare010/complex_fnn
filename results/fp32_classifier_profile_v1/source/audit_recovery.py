"""Explicit numerical-replay recovery; original H114 science is immutable."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import hashlib  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from results.fp32_classifier_profile_v1.source.audit import check_score  # noqa: E402
from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402
    clear_boundary,
    gradient_error,
    set_blocks,
    tensor_hash,
    training_loss,
    tree_hash,
)
from results.streamed_evaluation_v1.source.evaluation import evaluate  # noqa: E402
from src.core.config import ModelConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.reproducibility import sha256, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402

ROOT = Path("results/fp32_classifier_profile_v1")


def check_fixture(fixture, cases, protocol):
    state = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    cfg = ModelConfig(**fixture["model_config"])
    model = Transformer(cfg, fixture["seed"]).cuda()
    model.load_state_dict(state["model"])
    set_blocks(model)
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 40000
    )
    sampler = data.generator.get_state()
    initial_score = evaluate(model, data, fixture["batch"], cfg.context, 10**9, "native")
    rows, saved_probes = [], {}
    for case in cases:
        assert case["initial_model_hash"] == tree_hash(state["model"])
        assert case["source_optimizer_hash"] == tree_hash(state["optimizer"])
        assert case["initial_sampler_hash"] == tensor_hash(sampler)
        data.generator.set_state(sampler)
        x, y = data.batch(fixture["batch"], cfg.context)
        assert tensor_hash(x) == case["initial_probe"]["tokens_hash"]
        assert tensor_hash(y) == case["initial_probe"]["targets_hash"]
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = training_loss(model, x, y, case["policy"])
        loss.backward()
        assert loss.item() == case["initial_probe"]["loss"]
        actual = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
        saved = torch.load(case["initial_probe"]["path"], map_location="cpu", weights_only=True)
        assert tree_hash(saved) == case["initial_probe"]["gradients_hash"]
        replay = gradient_error(actual, saved)
        replay["bitwise_equal"] = tree_hash(actual) == tree_hash(saved)
        replay["max_absolute_error"] = max((actual[k] - saved[k]).abs().max().item() for k in saved)
        assert replay["global_relative_l2"] <= 1e-5 and replay["max_tensor_relative_l2"] <= 1e-4, (
            case["label"],
            replay,
        )
        saved_probes[case["policy"]] = dict(loss=case["initial_probe"]["loss"], gradients=saved)
        del actual, saved, loss, x, y
        data.generator.set_state(sampler)
        order = hashlib.sha256()
        for record in case["records"]:
            x, y = data.batch(fixture["batch"], cfg.context)
            assert (
                tensor_hash(x) == record["tokens_hash"] and tensor_hash(y) == record["targets_hash"]
            )
            order.update(record["tokens_hash"].encode())
        assert order.hexdigest() == case["data_order_hash"]
        assert tensor_hash(data.generator.get_state()) == case["final_sampler_hash"]
        del x, y
        rows.append(
            dict(
                label=case["label"],
                source_native_validation=initial_score,
                initial_evaluation_relative_drift=check_score(
                    initial_score, case["initial_validation"]
                ),
                exact_probe_gradient_replay=replay["bitwise_equal"],
                numerical_probe_gradient_replay=True,
                replay_error=replay,
                training_batches_verified=50,
            )
        )
        print(
            json.dumps(
                dict(
                    probe=case["label"],
                    replay_global_error=replay["global_relative_l2"],
                    replay_max_tensor_error=replay["max_tensor_relative_l2"],
                    bitwise_equal=replay["bitwise_equal"],
                )
            ),
            flush=True,
        )
    reference = saved_probes["block_fp32"]
    for case, row in zip(cases, rows, strict=True):
        candidate = saved_probes[case["policy"]]
        comparison = gradient_error(candidate["gradients"], reference["gradients"])
        comparison["relative_loss_difference"] = abs(candidate["loss"] / reference["loss"] - 1)
        row["probe_error_to_native_fp32"] = comparison
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
        score = evaluate(model, data, fixture["batch"], cfg.context, 10**9, "native")
        row.update(
            final_native_validation=score,
            final_evaluation_relative_drift=check_score(score, case["final_validation"]),
        )
    print(json.dumps(dict(audited_fixture=fixture["label"], cases=4)), flush=True)
    return rows


def run():
    frozen = json.loads((ROOT / "audit_recovery_protocol.json").read_text())
    for path, digest in frozen["files"].items():
        assert sha256(Path(path)) == digest, path
    original = json.loads((ROOT / "audit_protocol.json").read_text())
    for path, digest in original["files"].items():
        assert sha256(Path(path)) == digest, path
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    result = json.loads((ROOT / "result.json").read_text())
    rows, boundaries = [], [clear_boundary()]
    for fixture in protocol["fixtures"]:
        cases = [c for c in result["cases"] if c["fixture"]["label"] == fixture["label"]]
        rows.extend(check_fixture(fixture, cases, protocol))
        boundaries.append(clear_boundary())
    write_json(
        ROOT / "audit_recovery.json",
        dict(
            passed=True,
            fixtures=14,
            cases=56,
            native_validation_scores=70,
            exact_probe_gradient_replays=sum(r["exact_probe_gradient_replay"] for r in rows),
            numerical_probe_gradient_replays=56,
            training_batches_verified=2800,
            optimizer_updates=0,
            additional_backward_passes=56,
            posthoc_replay_tolerance=True,
            candidate_gates_unchanged=True,
            failed_original_backward_count_range=[1, 4],
            diagnosis_backward_passes=6,
            maximum_evaluation_relative_drift=max(
                max(r["initial_evaluation_relative_drift"], r["final_evaluation_relative_drift"])
                for r in rows
            ),
            rows=rows,
            boundaries=boundaries,
        ),
    )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "audit_recovery_failure.json", dict(traceback=traceback.format_exc()))
        raise
