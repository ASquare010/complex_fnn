"""Native rescoring, exact sampler replay and endpoint/optimizer audit for H112."""

import gc
import hashlib
import json
import math
from pathlib import Path

import torch

from results.streamed_evaluation_v1.source.evaluation import tensor_hash
from src.core.benchmark import evaluate_forward
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/whole_job_memory_workspace_v1")


def score(model, data, batches):
    order = hashlib.sha256()

    def stream():
        for x, y in data.validation(8, 512, batches):
            order.update(tensor_hash(x).encode())
            yield x, y

    nll, count = evaluate_forward(model, stream(), "cuda", "bf16")
    return {"nll": nll, "targets": count, "order_sha256": order.hexdigest()}


def run():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    result = json.loads((ROOT / "result.json").read_text())
    assert result["status"] == "COMPLETE" and len(result["cases"]) == 12
    assert result["updates"] == 9600 and len(result["boundaries"]) == 25
    assert all(
        b["allocated_bytes"] == 0 and b["workspace_bytes_subtracted_from_trial_measurements"] == 0
        for b in result["boundaries"]
    )
    scores, trials = [], []
    for row in result["cases"]:
        label = f"{row['dataset']}_{row['label']}"
        cfg = ModelConfig(**row["model_config"])
        data = TokenData(
            Path(protocol["datasets"][row["dataset"]]["path"]), "cuda", row["seed"] + 10000
        )
        model = Transformer(cfg, row["seed"]).cuda().eval()
        assert {k: tensor_hash(v) for k, v in model.state_dict().items()} == row[
            "initial_state_hashes"
        ]
        assert tensor_hash(data.generator.get_state()) == row["initial_sampler_sha256"]
        assert cfg.total_parameters == row["parameter_count"] == 9099648
        assert cfg.unique_ffn_parameters == row["ffn_parameter_count"] == 2801664

        def check_score(step, scope, original, batches, active_model=model, active_data=data):
            native = score(active_model, active_data, batches)
            assert (
                native["targets"] == original["targets"]
                and native["order_sha256"] == original["order_sha256"]
            )
            difference = abs(original["nll"] / native["nll"] - 1)
            assert difference <= 0.0001, (label, step, scope, difference)
            record = {
                "dataset": row["dataset"],
                "label": row["label"],
                "seed": row["seed"],
                "policy": row["policy"],
                "step": step,
                "scope": scope,
                "native_nll": native["nll"],
                "streamed_nll": original["nll"],
                "relative_difference_abs": difference,
                "targets": native["targets"],
                "order_sha256": native["order_sha256"],
            }
            scores.append(record)
            return record

        check_score(0, "subset", row["evaluations"][0], 16)
        order = hashlib.sha256()
        for _ in range(800):
            x, _ = data.batch(8, 512)
            order.update(tensor_hash(x).encode())
        assert order.hexdigest() == row["data_order_sha256"]
        assert tensor_hash(data.generator.get_state()) == row["final_sampler_sha256"]
        folder = Path(row["source_root"]) / "runs" / row["label"]
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
        assert row["peak_job_allocated_bytes"] >= max(
            row["peak_training_allocated_bytes"], row["peak_validation_allocated_bytes"]
        )
        assert all(a["finite"] for d in row["diagnostics"] for a in d["activations"].values())
        assert all(math.isfinite(v) for d in row["diagnostics"] for v in d["gradients"].values())
        checkpoints = sorted(row["checkpoints"] + row["extra_checkpoints"], key=lambda c: c["step"])
        assert [c["step"] for c in checkpoints] == [100, 200, 400, 800]
        final_score = None
        for c in checkpoints:
            path = Path(c["path"])
            assert sha256(path) == c["sha256"]
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            assert (
                checkpoint["step"] == c["step"]
                and checkpoint["model_config"] == row["model_config"]
            )
            assert all(bool(torch.isfinite(v).all()) for v in checkpoint["model"].values())
            model.load_state_dict(checkpoint["model"], strict=True)
            original = next(v for v in row["evaluations"] if v["step"] == c["step"])
            check_score(c["step"], "subset", original, 16)
            if c["step"] == 800:
                final_score = check_score(800, "full", row["full_validation"], 10**9)
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
        assert final_score is not None
        trials.append(
            {
                "dataset": row["dataset"],
                "label": row["label"],
                "seed": row["seed"],
                "policy": row["policy"],
                "native_full_nll": final_score["native_nll"],
                "exact_initialization": True,
                "exact_800_batch_stream": True,
                "finite_history_and_optimizer": True,
                "checkpoints": 4,
            }
        )
        print(f"Audited {label}: {len(scores)} native scores", flush=True)
        del check_score, model, data
        gc.collect()
        torch.cuda.empty_cache()
    assert len(scores) == 72 and len(trials) == 12
    write_json(
        ROOT / "audit.json",
        {
            "passed": True,
            "scores": scores,
            "trials": trials,
            "checkpoints": 48,
            "initializations": 12,
            "audit_updates": 0,
            "max_relative_score_difference": max(r["relative_difference_abs"] for r in scores),
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
