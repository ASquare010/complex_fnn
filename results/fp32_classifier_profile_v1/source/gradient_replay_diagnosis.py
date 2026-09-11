"""Bounded zero-update diagnosis of the preserved exact-gradient audit stop."""

import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

assert not torch.cuda.is_initialized()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import json  # noqa: E402
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
from src.core.config import ModelConfig, TrainConfig  # noqa: E402
from src.core.data import TokenData  # noqa: E402
from src.core.optimization import parameter_groups  # noqa: E402
from src.core.reproducibility import sha256, write_json  # noqa: E402
from src.core.transformer import Transformer  # noqa: E402

ROOT = Path("results/fp32_classifier_profile_v1")


def probe(fixture, case, protocol, with_optimizer):
    payload = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**fixture["model_config"]), fixture["seed"]).cuda()
    model.load_state_dict(payload["model"])
    set_blocks(model)
    if with_optimizer:
        optimizer = torch.optim.AdamW(
            parameter_groups(model, TrainConfig()), lr=0.0006, betas=(0.9, 0.95), eps=1e-8
        )
        optimizer.load_state_dict(payload["optimizer"])
    data = TokenData(
        Path(protocol["datasets"][fixture["dataset"]]["path"]), "cuda", fixture["seed"] + 40000
    )
    evaluate(
        model,
        data,
        fixture["batch"],
        fixture["context"],
        10**9,
        "classifier_chunks" if with_optimizer else "native",
    )
    x, y = data.batch(fixture["batch"], fixture["context"])
    assert tensor_hash(x) == case["initial_probe"]["tokens_hash"]
    saved = torch.load(case["initial_probe"]["path"], map_location="cpu", weights_only=True)
    rows, previous = [], None
    for rep in range(3):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = training_loss(model, x, y, case["policy"])
        loss.backward()
        actual = {k: p.grad.detach().cpu() for k, p in model.named_parameters()}
        error = gradient_error(actual, saved)
        differences = {
            k: dict(
                relative_l2=error["per_tensor_relative_l2"][k],
                max_abs=(actual[k] - saved[k]).abs().max().item(),
                differing_elements=int((actual[k] != saved[k]).sum()),
            )
            for k in actual
            if not torch.equal(actual[k], saved[k])
        }
        rows.append(
            dict(
                with_optimizer_and_matching_evaluator=with_optimizer,
                repetition=rep,
                loss=loss.item(),
                loss_equal=loss.item() == case["initial_probe"]["loss"],
                gradients_equal=tree_hash(actual) == tree_hash(saved),
                repeat_equal=tree_hash(actual) == tree_hash(previous)
                if previous is not None
                else None,
                error=error,
                differences=differences,
            )
        )
        previous = actual
    return rows


if __name__ == "__main__":
    assert not (ROOT / "gradient_replay_diagnosis.json").exists()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    result = json.loads((ROOT / "result.json").read_text())
    fixture, case = protocol["fixtures"][0], result["cases"][0]
    output, boundaries = [], [clear_boundary()]
    for flag in (False, True):
        output.extend(probe(fixture, case, protocol, flag))
        boundaries.append(clear_boundary())
    write_json(
        ROOT / "gradient_replay_diagnosis.json",
        dict(
            fixture=fixture["label"],
            policy=case["policy"],
            rows=output,
            boundaries=boundaries,
            optimizer_updates=0,
            backward_passes=6,
            source_sha256=sha256(Path(__file__)),
            preserved_failure_sha256=sha256(ROOT / "audit_failure.json"),
        ),
    )
    print(json.dumps([{k: v for k, v in r.items() if k != "error"} for r in output], indent=2))
