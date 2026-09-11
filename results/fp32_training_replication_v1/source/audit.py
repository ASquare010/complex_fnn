"""Regenerate initialization, replay FP32 gradients and score every saved endpoint."""

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

from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402
    clear_boundary,
    set_blocks,
    tensor_hash,
    tree_hash,
)
from results.fp32_decoder_resource_v1.source.audit import (  # noqa: E402
    independent_error,
    native_score,
    score_error,
)
from results.fp32_decoder_resource_v1.source.common import (  # noqa: E402
    attention_context,
    training_loss,
)
from results.fp32_training_replication_v1.source.initialize import initial_state  # noqa: E402
from src.core.config import ModelConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.reproducibility import sha256, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402

ROOT = Path("results/fp32_training_replication_v1")


def read(path):
    return json.loads(Path(path).read_text())


def audit_initial(fixture, peers, protocol):
    source = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    regenerated = initial_state(fixture)
    assert tree_hash(regenerated) == tree_hash(source)
    assert source["step"] == 0 and not source["optimizer"]["state"]
    model = Transformer(ModelConfig(**fixture["model_config"]), fixture["seed"]).cuda()
    model.load_state_dict(source["model"])
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 10000
    )
    score = native_score(model, data, fixture)
    errors = [score_error(score, c["initial_validation"]) for c in peers]
    return dict(
        fixture=fixture["label"],
        initialization_regenerated_exactly=True,
        score=score,
        maximum_relative_error=max(errors),
        passed=max(errors) <= 1e-6,
    )


def audit_case(case, protocol):
    fixture = case["fixture"]
    source = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    assert tree_hash(source["model"]) == case["initial_model_hash"]
    assert tree_hash(source["optimizer"]) == case["source_optimizer_hash"]
    saved = torch.load(case["initial_probe"]["path"], map_location="cpu", weights_only=True)
    assert tree_hash(saved) == case["initial_probe"]["gradients_hash"]
    cfg = ModelConfig(**fixture["model_config"])
    model = Transformer(cfg, fixture["seed"]).cuda()
    model.load_state_dict(source["model"])
    set_blocks(model)
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 10000
    )
    sampler = data.generator.get_state()
    assert tensor_hash(sampler) == case["initial_sampler_hash"]
    x, y = data.batch(fixture["batch"], cfg.context)
    assert tensor_hash(x) == case["initial_probe"]["tokens_hash"]
    assert tensor_hash(y) == case["initial_probe"]["targets_hash"]
    replay = None
    if case["policy"].startswith("fp32"):
        model.train()
        loss = training_loss(model, x, y, case["policy"])
        with attention_context(case["policy"]):
            loss.backward()
        actual = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
        error = independent_error(actual, saved)
        relative_loss = abs(loss.item() / case["initial_probe"]["loss"] - 1)
        replay = dict(
            error=error,
            loss_relative_error=relative_loss,
            passed=error["global_relative_l2"] <= 1e-5
            and error["max_tensor_relative_l2"] <= 1e-4
            and relative_loss <= 1e-6,
            bitwise_equal=tree_hash(actual) == case["initial_probe"]["gradients_hash"],
        )
        del loss, actual
        model.zero_grad(set_to_none=True)
    data.generator.set_state(sampler)
    order, sampler_hashes = hashlib.sha256(), {}
    for record in case["records"]:
        x, y = data.batch(fixture["batch"], cfg.context)
        assert tensor_hash(x) == record["tokens_hash"] and tensor_hash(y) == record["targets_hash"]
        order.update(record["tokens_hash"].encode())
        if record["step"] in (200, 400, 800):
            sampler_hashes[record["step"]] = tensor_hash(data.generator.get_state())
    assert order.hexdigest() == case["data_order_hash"]
    assert sampler_hashes[800] == case["final_sampler_hash"]
    checkpoints = [
        *case["intermediate_checkpoints"],
        dict(
            step=800,
            **case["checkpoint"],
            model_hash=case["final_model_hash"],
            optimizer_hash=case["final_optimizer_hash"],
            sampler_hash=case["final_sampler_hash"],
            validation=case["final_validation"],
        ),
    ]
    assert [c["step"] for c in checkpoints] == [200, 400, 800]
    scores = []
    for checkpoint in checkpoints:
        final = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
        assert tree_hash(final["model"]) == checkpoint["model_hash"]
        assert tree_hash(final["optimizer"]) == checkpoint["optimizer_hash"]
        assert (
            tensor_hash(final["sampler_state"])
            == checkpoint["sampler_hash"]
            == sampler_hashes[checkpoint["step"]]
        )
        assert final["step"] == checkpoint["step"]
        assert (
            final["model_config"] == fixture["model_config"]
            and final["training_config"] == case["training_config"]
        )
        assert all(float(s["step"]) == final["step"] for s in final["optimizer"]["state"].values())
        assert all(torch.isfinite(v).all() for v in final["model"].values())
        assert all(
            torch.isfinite(v).all()
            for s in final["optimizer"]["state"].values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )
        model.load_state_dict(final["model"])
        score = native_score(model, data, fixture)
        error = score_error(score, checkpoint["validation"])
        scores.append(
            dict(
                step=checkpoint["step"],
                native_score=score,
                relative_error=error,
                passed=error <= 1e-6,
                state_and_sampler_verified=True,
            )
        )
        del final
    row = dict(
        label=case["label"],
        checkpoint_scores=scores,
        final_native_score=scores[-1]["native_score"],
        final_score_relative_error=scores[-1]["relative_error"],
        final_score_passed=scores[-1]["passed"],
        all_scores_passed=all(r["passed"] for r in scores),
        replay=replay,
        gradient_hash_verified=True,
        source_and_final_states_verified=True,
        trained_checkpoints_verified=3,
        training_batches_verified=800,
    )
    print(
        json.dumps(
            dict(
                audited=case["label"],
                all_scores_passed=row["all_scores_passed"],
                replay_passed=None if replay is None else replay["passed"],
            )
        ),
        flush=True,
    )
    return row


def run():
    frozen, protocol = read(ROOT / "audit_protocol.json"), read(ROOT / "protocol.json")
    for path, digest in frozen["files"].items():
        assert sha256(Path(path)) == digest, path
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    result = read(ROOT / "result.json")
    boundaries, initials, rows, pairs = [clear_boundary()], [], [], []
    for fixture in protocol["fixtures"]:
        peers = [c for c in result["cases"] if c["fixture"]["label"] == fixture["label"]]
        initials.append(audit_initial(fixture, peers, protocol))
        boundaries.append(clear_boundary())
        for case in peers:
            rows.append(audit_case(case, protocol))
            boundaries.append(clear_boundary())
        candidate = next(c for c in peers if c["policy"] == "fp32_default_chunks")
        reference = next(c for c in peers if c["policy"] == "fp32_default_native")
        a = torch.load(candidate["initial_probe"]["path"], map_location="cpu", weights_only=True)
        b = torch.load(reference["initial_probe"]["path"], map_location="cpu", weights_only=True)
        pairs.append(
            dict(
                label=candidate["label"],
                reference=reference["label"],
                error=independent_error(a, b),
                loss_relative_error=abs(
                    candidate["initial_probe"]["loss"] / reference["initial_probe"]["loss"] - 1
                ),
            )
        )
    replays = [r["replay"] for r in rows if r["replay"] is not None]
    assert len(rows) == 18 and len(initials) == 6 and len(replays) == 12 and len(pairs) == 6
    write_json(
        ROOT / "audit.json",
        dict(
            passed=all(r["passed"] for r in initials + replays)
            and all(r["all_scores_passed"] for r in rows),
            initial_native_scores=initials,
            rows=rows,
            paired_initial_errors=pairs,
            boundaries=boundaries,
            regenerated_initializations=6,
            trained_checkpoints_verified=54,
            native_validation_scores=60,
            gradient_replay_backward_passes=12,
            replay_gates_passed=all(r["passed"] for r in replays),
            bitwise_gradient_replays=sum(r["bitwise_equal"] for r in replays),
            saved_gradient_hashes_verified=18,
            training_batches_verified=14400,
            optimizer_updates=0,
            fresh_training_runs=0,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        write_json(ROOT / "audit_failure.json", dict(traceback=traceback.format_exc()))
        raise
