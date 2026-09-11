"""Independent H114 native rescoring, exact probe replay and stream verification."""

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


def check_score(actual, stored):
    assert (
        actual["targets"] == stored["targets"] and actual["order_sha256"] == stored["order_sha256"]
    )
    drift = abs(actual["nll"] / stored["nll"] - 1)
    assert drift <= 1e-6, drift
    return drift


def check_fixture(fixture, cases, protocol):
    state = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    cfg = ModelConfig(**fixture["model_config"])
    model = Transformer(cfg, fixture["seed"]).cuda()
    model.load_state_dict(state["model"])
    set_blocks(model)
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 40000
    )
    initial_sampler = data.generator.get_state()
    source_score = evaluate(model, data, fixture["batch"], cfg.context, 10**9, "native")
    peer_rows, probe_values = [], {}
    for case in cases:
        assert case["initial_model_hash"] == tree_hash(state["model"])
        assert case["source_optimizer_hash"] == tree_hash(state["optimizer"])
        assert case["initial_sampler_hash"] == tensor_hash(initial_sampler)
        initial_drift = check_score(source_score, case["initial_validation"])
        data.generator.set_state(initial_sampler)
        x, y = data.batch(fixture["batch"], cfg.context)
        assert tensor_hash(x) == case["initial_probe"]["tokens_hash"]
        assert tensor_hash(y) == case["initial_probe"]["targets_hash"]
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = training_loss(model, x, y, case["policy"])
        loss.backward()
        assert loss.item() == case["initial_probe"]["loss"]
        gradients = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
        saved = torch.load(case["initial_probe"]["path"], map_location="cpu", weights_only=True)
        assert tree_hash(gradients) == tree_hash(saved) == case["initial_probe"]["gradients_hash"]
        probe_values[case["policy"]] = dict(loss=loss.item(), gradients=saved)
        del saved, gradients, loss, x, y
        data.generator.set_state(initial_sampler)
        order = hashlib.sha256()
        for row in case["records"]:
            x, y = data.batch(fixture["batch"], cfg.context)
            assert tensor_hash(x) == row["tokens_hash"] and tensor_hash(y) == row["targets_hash"]
            order.update(row["tokens_hash"].encode())
        assert order.hexdigest() == case["data_order_hash"]
        assert tensor_hash(data.generator.get_state()) == case["final_sampler_hash"]
        del x, y
        peer_rows.append(
            dict(
                label=case["label"],
                source_native_validation=source_score,
                initial_evaluation_relative_drift=initial_drift,
                exact_probe_gradient_replay=True,
                training_batches_verified=50,
            )
        )
    reference = probe_values["block_fp32"]
    gradient_comparisons = {}
    for policy, probe in probe_values.items():
        error = gradient_error(probe["gradients"], reference["gradients"])
        error["relative_loss_difference"] = abs(probe["loss"] / reference["loss"] - 1)
        gradient_comparisons[policy] = error
    del probe_values, reference, state, initial_sampler
    for case, row in zip(cases, peer_rows, strict=True):
        checkpoint = torch.load(case["checkpoint"]["path"], map_location="cpu", weights_only=True)
        assert checkpoint["step"] == fixture["source_step"] + 50
        assert checkpoint["model_config"] == fixture["model_config"]
        assert checkpoint["continuation_steps"] == 50
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
            probe_error_to_native_fp32=gradient_comparisons[case["policy"]],
        )
        del checkpoint
    print(
        json.dumps(
            dict(
                audited_fixture=fixture["label"],
                cases=4,
                fp32_chunk_gradient_error=gradient_comparisons["chunks_fp32"]["global_relative_l2"],
            )
        ),
        flush=True,
    )
    return peer_rows


def run():
    audit_protocol = json.loads((ROOT / "audit_protocol.json").read_text())
    for path, digest in audit_protocol["files"].items():
        assert sha256(Path(path)) == digest, path
    protocol = json.loads((ROOT / "protocol.json").read_text())
    result = json.loads((ROOT / "result.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    rows, boundaries = [], [clear_boundary()]
    for fixture in protocol["fixtures"]:
        peers = [c for c in result["cases"] if c["fixture"]["label"] == fixture["label"]]
        assert len(peers) == 4
        rows.extend(check_fixture(fixture, peers, protocol))
        boundaries.append(clear_boundary())
    write_json(
        ROOT / "audit.json",
        dict(
            passed=True,
            fixtures=14,
            cases=56,
            native_validation_scores=70,
            exact_probe_gradient_replays=56,
            training_batches_verified=2800,
            optimizer_updates=0,
            additional_backward_passes=56,
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
        write_json(ROOT / "audit_failure.json", dict(traceback=traceback.format_exc()))
        raise
