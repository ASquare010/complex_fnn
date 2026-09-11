"""Audit all completed H110 trials with native unchunked scoring and stream replay."""

import gc
import hashlib
import json
import math
from pathlib import Path

import torch

from src.core.benchmark import evaluate
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/token_memory_duration_v1")


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def run():
    result_path = ROOT / "result.json"
    if not result_path.exists():
        result_path = ROOT / "partial_result.json"
    result = json.loads(result_path.read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert result["status"] in ("COMPLETE", "STOPPED_FIDELITY_GATE", "RUNTIME_INTERRUPTED")
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    records, scores = [], []
    for row in result["cases"]:
        label = row["label"]
        source_root = Path(row.get("source_root", ROOT))
        folder = source_root / "runs" / label
        assert (source_root / f"{label}_exit.txt").read_text().strip() == "0"
        data = TokenData(Path("data/wikitext2_v1"), "cuda", row["seed"] + 10000)
        cfg = ModelConfig(**row["model_config"])
        model = Transformer(cfg, row["seed"]).cuda()
        assert {k: tensor_hash(v) for k, v in model.state_dict().items()} == row[
            "initial_state_hashes"
        ]
        assert sum(p.numel() for p in model.parameters()) == row["parameter_count"] == 9099648
        assert cfg.unique_ffn_parameters == row["ffn_parameter_count"] == 2801664
        assert tensor_hash(data.generator.get_state()) == row["initial_sampler_sha256"]
        initial, count = evaluate(
            model, data.validation(row["batch"], row["context"], 16), "cuda", "bf16"
        )
        assert initial == row["evaluations"][0]["nll"] and count == row["evaluations"][0]["targets"]
        scores.append({"label": label, "step": 0, "scope": "subset", "difference": 0.0})
        stream = hashlib.sha256()
        for _ in range(800):
            x, _ = data.batch(row["batch"], row["context"])
            stream.update(tensor_hash(x).encode())
        assert stream.hexdigest() == row["data_order_sha256"]
        assert tensor_hash(data.generator.get_state()) == row["final_sampler_sha256"]
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        assert history == row["records"] and len(history) == 800
        assert [h["step"] for h in history] == list(range(1, 801))
        assert all(
            math.isfinite(h[k]) and h[k] >= 0
            for h in history
            for k in (
                "loss",
                "preclip_norm",
                "forward_ms",
                "backward_ms",
                "optimizer_ms",
                "update_ms",
            )
        )
        assert row["peak_training_allocated_bytes"] == max(
            h["peak_allocated_bytes"] for h in history
        )
        assert row["peak_training_reserved_bytes"] == max(h["peak_reserved_bytes"] for h in history)
        assert row["peak_job_allocated_bytes"] >= max(
            row["peak_training_allocated_bytes"], row["peak_validation_allocated_bytes"]
        )
        assert [v["step"] for v in row["evaluations"]] == [0, 100, 200, 400, 800]
        assert [v["step"] for v in row["diagnostics"]] == [0, 100, 200, 400, 800]
        assert all(a["finite"] for d in row["diagnostics"] for a in d["activations"].values())
        assert all(math.isfinite(v) for d in row["diagnostics"] for v in d["gradients"].values())
        assert [c["step"] for c in row["checkpoints"]] == [100, 400, 800]
        for c in row["checkpoints"]:
            assert sha256(Path(c["path"])) == c["sha256"]
            checkpoint = torch.load(c["path"], map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"], strict=True)
            assert (
                checkpoint["model_config"] == row["model_config"]
                and checkpoint["step"] == c["step"]
            )
            assert all(bool(torch.isfinite(v).all()) for v in checkpoint["model"].values())
            value, count = evaluate(
                model, data.validation(row["batch"], row["context"], 16), "cuda", "bf16"
            )
            original = next(v for v in row["evaluations"] if v["step"] == c["step"])
            assert value == original["nll"] and count == original["targets"], (
                label,
                c["step"],
                value,
                original["nll"],
            )
            scores.append(
                {
                    "label": label,
                    "step": c["step"],
                    "scope": "subset",
                    "difference": abs(value - original["nll"]),
                }
            )
            if c["step"] == 800:
                full, count = evaluate(
                    model, data.validation(row["batch"], row["context"], 10**9), "cuda", "bf16"
                )
                assert (
                    full == row["full_validation"]["nll"]
                    and count == row["full_validation"]["targets"]
                )
                scores.append(
                    {
                        "label": label,
                        "step": 800,
                        "scope": "full",
                        "difference": abs(full - row["full_validation"]["nll"]),
                    }
                )
                assert tensor_hash(checkpoint["sampler_state"]) == row["final_sampler_sha256"]
                states = checkpoint["optimizer"]["state"]
                assert len(states) == len(list(model.parameters()))
                assert all(s["step"].item() == 800 for s in states.values())
                assert all(
                    bool(torch.isfinite(v).all())
                    for state in states.values()
                    for v in state.values()
                    if isinstance(v, torch.Tensor)
                )
                assert (
                    sum(
                        v.numel() * v.element_size()
                        for state in states.values()
                        for v in state.values()
                        if isinstance(v, torch.Tensor)
                    )
                    == row["optimizer_bytes"]
                )
                assert all(
                    g["lr"] == 0.0006 and g["betas"] == (0.9, 0.95) and g["eps"] == 1e-8
                    for g in checkpoint["optimizer"]["param_groups"]
                )
            del checkpoint
        order = hashlib.sha256()
        for x, _ in data.validation(row["batch"], row["context"], 10**9):
            order.update(tensor_hash(x).encode())
        assert order.hexdigest() == row["full_validation"]["order_sha256"]
        records.append(
            {
                "label": label,
                "initial_state_and_sampler_exact": True,
                "all_800_training_batches_exact": True,
                "finite_800_step_history": True,
                "finite_final_optimizer_and_steps": True,
                "checkpoints": 3,
            }
        )
        print(f"Audited {label}: {len(scores)} exact NLL checks", flush=True)
        del model, data
        gc.collect()
        torch.cuda.empty_cache()
    assert len(scores) == 5 * len(result["cases"])
    write_json(
        ROOT / "audit.json",
        {
            "passed": True,
            "experiment_status": result["status"],
            "result_path": result_path.as_posix(),
            "trials": records,
            "scores": scores,
            "checkpoints": 3 * len(records),
            "initializations": len(records),
            "audit_updates": 0,
            "max_nll_difference": max(s["difference"] for s in scores),
        },
    )
    print("Independent duration audit passed", flush=True)


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
