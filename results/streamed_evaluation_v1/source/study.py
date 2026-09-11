"""H111: preserved endpoint evaluation, with explicitly charged persistent fixtures."""

import gc
import json
import math
import statistics as st
import time
from pathlib import Path

import torch

from results.streamed_evaluation_v1.source.evaluation import POLICIES, evaluate, tensor_hash
from results.streamed_evaluation_v1.source.qualification import run as qualify
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import environment, sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/streamed_evaluation_v1")
OLD = Path("results/token_memory_duration_fresh_v1")


def snapshot(model, data):
    return {
        "model": {k: tensor_hash(v) for k, v in model.state_dict().items()},
        "gradients": {k: tensor_hash(p.grad) for k, p in model.named_parameters()},
        "training": model.training,
        "sampler": tensor_hash(data.generator.get_state()),
        "cpu_rng": tensor_hash(torch.random.get_rng_state()),
        "cuda_rng": tensor_hash(torch.cuda.get_rng_state()),
    }


def run():
    assert not (ROOT / "result.json").exists()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    checks = qualify()
    assert len(checks) == 36
    write_json(ROOT / "qualification.json", {"passed": True, "checks": checks})
    write_json(ROOT / "environment.json", environment())
    old = json.loads((OLD / "result.json").read_text())
    rows, job_peak = [], 0
    start = time.perf_counter()
    for case_index, case in enumerate(old["cases"]):
        data = TokenData(Path("data/wikitext2_v1"), "cuda", case["seed"] + 10000)
        cfg = ModelConfig(**case["model_config"])
        model = Transformer(cfg, case["seed"]).cuda().train()
        final_path = Path(case["checkpoints"][-1]["path"])
        assert sha256(final_path) == case["checkpoints"][-1]["sha256"]
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        moments = [
            {
                k: v.cuda() if isinstance(v, torch.Tensor) and k != "step" else v
                for k, v in state.items()
            }
            for state in final["optimizer"]["state"].values()
        ]
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        fixture_bytes = sum(
            v.numel() * v.element_size()
            for state in moments
            for v in state.values()
            if isinstance(v, torch.Tensor) and v.is_cuda
        )
        gradient_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters())
        assert gradient_bytes == case["gradient_bytes"]
        assert fixture_bytes + 4 * len(moments) == case["optimizer_bytes"]
        del final
        for endpoint_index, checkpoint in enumerate(case["checkpoints"]):
            path = Path(checkpoint["path"])
            assert sha256(path) == checkpoint["sha256"]
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"], strict=True)
            del state
            before = snapshot(model, data)
            policy_rows = []
            offset = (case_index + endpoint_index) % 3
            policies = POLICIES[offset:] + POLICIES[:offset]
            for policy in policies:
                job_peak = max(job_peak, torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
                score = evaluate(model, data, case["batch"], case["context"], 10**9, policy)
                allocated, reserved = (
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                )
                assert math.isfinite(score["nll"])
                assert snapshot(model, data) == before
                for _ in range(2):
                    evaluate(model, data, case["batch"], case["context"], 16, policy)
                times = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    evaluate(model, data, case["batch"], case["context"], 16, policy)
                    torch.cuda.synchronize()
                    times.append(1000 * (time.perf_counter() - begin))
                assert snapshot(model, data) == before
                job_peak = max(job_peak, torch.cuda.max_memory_allocated())
                policy_rows.append(
                    {
                        "policy": policy,
                        **score,
                        "peak_allocated_bytes": allocated,
                        "peak_reserved_bytes": reserved,
                        "timing_ms": {
                            "samples": times,
                            "median": st.median(times),
                            "mean": st.mean(times),
                            "sample_variance": st.variance(times),
                        },
                        "model_rng_gradients_mode_unchanged": True,
                    }
                )
            lookup = {r["policy"]: r for r in policy_rows}
            native = lookup["native"]
            if checkpoint["step"] == 800:
                assert native["nll"] == case["full_validation"]["nll"], (
                    case["label"],
                    native["nll"],
                    case["full_validation"]["nll"],
                )
            for row in policy_rows:
                assert (
                    row["targets"] == native["targets"]
                    and row["order_sha256"] == native["order_sha256"]
                )
                row["relative_nll_difference_abs"] = abs(row["nll"] / native["nll"] - 1)
                row["score_passes"] = row["relative_nll_difference_abs"] <= 0.0001
                row["evaluation_memory_ratio"] = (
                    row["peak_allocated_bytes"] / native["peak_allocated_bytes"]
                )
                row["evaluation_time_ratio"] = (
                    row["timing_ms"]["median"] / native["timing_ms"]["median"]
                )
                row["estimated_job_peak_bytes"] = max(
                    case["peak_training_allocated_bytes"], row["peak_allocated_bytes"]
                )
            output = {
                "label": case["label"],
                "context": case["context"],
                "batch": case["batch"],
                "seed": case["seed"],
                "training_policy": case["policy"],
                "checkpoint": checkpoint,
                "original_training_peak_bytes": case["peak_training_allocated_bytes"],
                "optimizer_fixture_cuda_bytes": fixture_bytes,
                "zero_gradient_fixture_bytes": gradient_bytes,
                "snapshot": before,
                "policies": policy_rows,
            }
            rows.append(output)
            with (ROOT / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(output) + "\n")
            print(
                json.dumps(
                    {
                        "state": len(rows),
                        "label": case["label"],
                        "step": checkpoint["step"],
                        "max_relative_score_difference": max(
                            r["relative_nll_difference_abs"] for r in policy_rows
                        ),
                        "native_mib": native["peak_allocated_bytes"] / 2**20,
                    }
                ),
                flush=True,
            )
        del model, data, moments
        gc.collect()
        torch.cuda.empty_cache()
    assert len(rows) == 24
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "rows": rows,
            "states": len(rows),
            "scores": 3 * len(rows),
            "optimizer_updates": 0,
            "actual_training_job": False,
            "peak_process_allocated_bytes": job_peak,
            "wall_seconds": time.perf_counter() - start,
            "broad_goal_achieved": False,
        },
    )
    print("H111 complete: 24 states / 72 scores / zero optimizer updates", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "failure.json", {"traceback": traceback.format_exc()})
        raise
