"""Independent native scoring, FP32 gradient replay and complete stream/state audit."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

from torch.nn import functional as F  # noqa: E402

from results.fp32_classifier_profile_v1.source.common import (  # noqa: E402
    clear_boundary,
    set_blocks,
    tensor_hash,
    tree_hash,
)
from results.fp32_decoder_resource_v1.source.common import (  # noqa: E402
    attention_context,
    training_loss,
)
from src.core.config import ModelConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.reproducibility import sha256, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402

ROOT = Path("results/fp32_decoder_resource_v1")


def read(path):
    return json.loads(Path(path).read_text())


def independent_error(actual, reference):
    per, error_sum, reference_sum = {}, 0.0, 0.0
    assert set(actual) == set(reference)
    for name in reference:
        a, b = actual[name].double(), reference[name].double()
        e, n = (a - b).square().sum().item(), b.square().sum().item()
        assert math.isfinite(e) and math.isfinite(n)
        per[name] = math.sqrt(e) / max(math.sqrt(n), 1e-8)
        error_sum += e
        reference_sum += n
    return dict(
        global_relative_l2=math.sqrt(error_sum) / max(math.sqrt(reference_sum), 1e-12),
        per_tensor_relative_l2=per,
        max_tensor_relative_l2=max(per.values()),
    )


@torch.no_grad()
def native_score(model, data, fixture):
    model.eval()
    total, targets, order = 0.0, 0, hashlib.sha256()
    for x, y in data.validation(fixture["batch"], fixture["context"], 10**9):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        count = int((y != -100).sum().item())
        total += loss.item() * count
        targets += count
        order.update(tensor_hash(x).encode())
    assert targets > 0 and math.isfinite(total)
    return dict(nll=total / targets, targets=targets, order_sha256=order.hexdigest())


def score_error(actual, expected):
    assert actual["targets"] == expected["targets"]
    assert actual["order_sha256"] == expected["order_sha256"]
    return abs(actual["nll"] / expected["nll"] - 1)


def initial_score(fixture, peers, protocol):
    source = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**fixture["model_config"]), fixture["seed"]).cuda()
    model.load_state_dict(source["model"])
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 40000
    )
    score = native_score(model, data, fixture)
    errors = [score_error(score, c["initial_validation"]) for c in peers]
    return dict(
        fixture=fixture["label"],
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
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 40000
    )
    sampler = data.generator.get_state()
    assert tensor_hash(sampler) == case["initial_sampler_hash"]
    x, y = data.batch(fixture["batch"], cfg.context)
    assert tensor_hash(x) == case["initial_probe"]["tokens_hash"] == fixture["probe_tokens_sha256"]
    assert (
        tensor_hash(y) == case["initial_probe"]["targets_hash"] == fixture["probe_targets_sha256"]
    )
    replay = None
    if case["policy"].startswith("fp32"):
        model.train()
        loss = training_loss(model, x, y, case["policy"])
        with attention_context(case["policy"]):
            loss.backward()
        actual = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
        error = independent_error(actual, saved)
        loss_error = abs(loss.item() / case["initial_probe"]["loss"] - 1)
        replay = dict(
            error=error,
            loss_relative_error=loss_error,
            passed=error["global_relative_l2"] <= 1e-5
            and error["max_tensor_relative_l2"] <= 1e-4
            and loss_error <= 1e-6,
            bitwise_equal=tree_hash(actual) == case["initial_probe"]["gradients_hash"],
        )
        del loss, actual
        model.zero_grad(set_to_none=True)
    data.generator.set_state(sampler)
    order = hashlib.sha256()
    for record in case["records"]:
        x, y = data.batch(fixture["batch"], cfg.context)
        assert tensor_hash(x) == record["tokens_hash"] and tensor_hash(y) == record["targets_hash"]
        order.update(record["tokens_hash"].encode())
    assert order.hexdigest() == case["data_order_hash"]
    assert tensor_hash(data.generator.get_state()) == case["final_sampler_hash"]
    final = torch.load(case["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert tree_hash(final["model"]) == case["final_model_hash"]
    assert tree_hash(final["optimizer"]) == case["final_optimizer_hash"]
    assert tensor_hash(final["sampler_state"]) == case["final_sampler_hash"]
    assert final["step"] == fixture["source_step"] + 50 and final["continuation_steps"] == 50
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
    relative = score_error(score, case["final_validation"])
    row = dict(
        label=case["label"],
        final_native_score=score,
        final_score_relative_error=relative,
        final_score_passed=relative <= 1e-6,
        replay=replay,
        gradient_hash_verified=True,
        source_and_final_states_verified=True,
        training_batches_verified=50,
    )
    print(
        json.dumps(
            dict(
                audited=case["label"],
                native_error=relative,
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
    for field in ("sources", "checkpoint_hashes"):
        for path, digest in protocol[field].items():
            assert sha256(Path(path)) == digest, path
    result = read(ROOT / "result.json")
    boundaries, initials, rows, pairs = [clear_boundary()], [], [], []
    for fixture in protocol["fixtures"]:
        peers = [c for c in result["cases"] if c["fixture"]["label"] == fixture["label"]]
        initials.append(initial_score(fixture, peers, protocol))
        boundaries.append(clear_boundary())
        for case in peers:
            rows.append(audit_case(case, protocol))
            boundaries.append(clear_boundary())
        for backend in ("default", "math"):
            candidate = next(c for c in peers if c["policy"] == f"fp32_{backend}_chunks")
            reference = next(c for c in peers if c["policy"] == f"fp32_{backend}_native")
            a = torch.load(
                candidate["initial_probe"]["path"], map_location="cpu", weights_only=True
            )
            b = torch.load(
                reference["initial_probe"]["path"], map_location="cpu", weights_only=True
            )
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
    assert len(rows) == 30 and len(initials) == 6 and len(replays) == 24 and len(pairs) == 12
    write_json(
        ROOT / "audit.json",
        dict(
            passed=all(r["passed"] for r in initials + replays)
            and all(r["final_score_passed"] for r in rows),
            initial_native_scores=initials,
            rows=rows,
            paired_initial_errors=pairs,
            boundaries=boundaries,
            native_validation_scores=36,
            gradient_replay_backward_passes=24,
            replay_gates_passed=all(r["passed"] for r in replays),
            bitwise_gradient_replays=sum(r["bitwise_equal"] for r in replays),
            saved_gradient_hashes_verified=30,
            training_batches_verified=1500,
            optimizer_updates=0,
            fresh_training_runs=0,
            broad_goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    import traceback

    try:
        run()
    except Exception:
        write_json(ROOT / "audit_failure.json", dict(traceback=traceback.format_exc()))
        raise
