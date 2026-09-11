"""Native recapture and independent closed-form FP64 classifier derivative audit."""

import json
from pathlib import Path

import sympy
import torch
import torch._dynamo

from results.classifier_precision_v1.source.common import clear, error, loss_for, tensor_hash
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/classifier_precision_v1")


def recapture(row, protocol, stored):
    checkpoint = torch.load(row["source_checkpoint"]["path"], map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**row["model_config"]), row["seed"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    data = TokenData(Path(protocol["datasets"][row["dataset"]]["path"]), "cpu", row["seed"] + 30000)
    tokens, targets = data.batch(8, 512)
    observed = []
    handle = model.norm.register_forward_hook(
        lambda module, inputs, output, values=observed: values.append(output.detach().cpu())
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(tokens.cuda())
    handle.remove()
    assert len(observed) == 1
    actual = {
        "hidden": observed[0],
        "weight": model.embedding.weight.detach().cpu(),
        "tokens": tokens,
        "targets": targets,
    }
    assert all(torch.equal(actual[k], stored[k]) for k in actual)
    assert {k: tensor_hash(v) for k, v in actual.items()} == row["tensor_hashes"]
    del model, data, checkpoint, observed, actual, tokens, targets, handle
    clear()


@torch.no_grad()
def closed_form(payload):
    h = payload["hidden"].cuda().double().reshape(-1, payload["hidden"].shape[-1])
    w, y = payload["weight"].cuda().double(), payload["targets"].cuda().reshape(-1)
    logits = h @ w.T
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    valid = y != -100
    index = torch.arange(y.numel(), device="cuda")[valid]
    loss = -logp[index, y[valid]].mean()
    g = logits.softmax(-1)
    g[index, y[valid]] -= 1
    g[~valid] = 0
    g /= valid.sum()
    dh, dw = g @ w, g.T @ h
    output = {
        "loss": loss.item(),
        "hidden_gradient": dh.reshape(payload["hidden"].shape).cpu(),
        "weight_gradient": dw.cpu(),
    }
    return output


def rerun(payload, policy, double=False):
    dtype = torch.float64 if double else torch.float32
    h = payload["hidden"].to(device="cuda", dtype=dtype).requires_grad_()
    w = payload["weight"].to(device="cuda", dtype=dtype).requires_grad_()
    y = payload["targets"].cuda()
    loss = loss_for(h, w, y, policy)
    dh, dw = torch.autograd.grad(loss, (h, w))
    return {"loss": loss.item(), "hidden_gradient": dh.cpu(), "weight_gradient": dw.cpu()}


def run():
    print(f"CPU preload passed: sympy={sympy.__version__}, torch={torch.__version__}", flush=True)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    for path, digest in protocol["checkpoint_hashes"].items():
        assert sha256(Path(path)) == digest, path
    result = json.loads((ROOT / "result.json").read_text())
    assert result["status"] == "COMPLETE" and result["optimizer_updates"] == 0
    records = []
    for row in result["rows"]:
        assert sha256(Path(row["path"])) == row["sha256"]
        assert sha256(Path(row["gradients_path"])) == row["gradients_sha256"]
        payload = torch.load(row["path"], map_location="cpu", weights_only=True)
        gradients = torch.load(row["gradients_path"], map_location="cpu", weights_only=True)
        recapture(row, protocol, payload)
        manual = closed_form(payload)
        clear()
        reference = gradients["native_fp64"]
        assert abs(manual["loss"] / reference["loss"] - 1) <= 1e-10
        for key in ("hidden_gradient", "weight_gradient"):
            torch.testing.assert_close(manual[key], reference[key], rtol=1e-10, atol=1e-12)
        checks = []
        for policy, original in gradients.items():
            current = rerun(
                payload,
                "native_fp32" if policy == "native_fp64" else policy,
                double=policy == "native_fp64",
            )
            assert current["loss"] == original["loss"], (row["label"], policy, "loss")
            assert all(
                torch.equal(current[k], original[k]) for k in ("hidden_gradient", "weight_gradient")
            ), (row["label"], policy, "gradients")
            if policy != "native_fp64":
                record = next(r for r in row["records"] if r["policy"] == policy)
                assert record["loss"] == original["loss"]
                assert record["relative_loss_error"] == abs(
                    original["loss"] / reference["loss"] - 1
                )
                for key in ("hidden_gradient", "weight_gradient"):
                    assert record[key] == error(original[key], reference[key])
            checks.append(policy)
            del current
            clear()
        records.append(
            {
                "label": row["label"],
                "recapture_exact": True,
                "independent_closed_form_passed": True,
                "exact_gradient_reruns": checks,
            }
        )
        del payload, gradients, manual, reference
        print(f"Audited {len(records)}/24: {row['label']}", flush=True)
    assert len(records) == 24
    write_json(
        ROOT / "audit.json",
        {
            "passed": True,
            "fixtures": 24,
            "native_recaptures": 24,
            "closed_form_reference_checks": 24,
            "exact_gradient_reruns": 144,
            "scalar_error_records": 240,
            "optimizer_updates": 0,
            "records": records,
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "audit_failure.json", {"traceback": traceback.format_exc()})
        raise
