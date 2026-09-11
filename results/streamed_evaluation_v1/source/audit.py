"""Re-score all states through maintained native forward; check 72 comparisons."""

import gc
import json
import math
import statistics as st
from pathlib import Path

import torch

from results.streamed_evaluation_v1.source.evaluation import tensor_hash
from src.core.benchmark import evaluate_forward
from src.core.config import ModelConfig
from src.core.data import TokenData
from src.core.reproducibility import sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/streamed_evaluation_v1")


def run():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    result = json.loads((ROOT / "result.json").read_text())
    old = json.loads(Path("results/token_memory_duration_fresh_v1/result.json").read_text())
    cases = {r["label"]: r for r in old["cases"]}
    data = TokenData(Path("data/wikitext2_v1"), "cuda", 17)
    checks = []
    for row in result["rows"]:
        case = cases[row["label"]]
        path = Path(row["checkpoint"]["path"])
        assert (
            sha256(path)
            == row["checkpoint"]["sha256"]
            == protocol["checkpoint_hashes"][path.as_posix()]
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert {k: tensor_hash(v) for k, v in checkpoint["model"].items()} == row["snapshot"][
            "model"
        ]
        assert {k: tensor_hash(torch.zeros_like(v)) for k, v in checkpoint["model"].items()} == row[
            "snapshot"
        ]["gradients"]
        assert row["snapshot"]["training"]
        assert row["zero_gradient_fixture_bytes"] == case["gradient_bytes"]
        assert (
            row["optimizer_fixture_cuda_bytes"] + 4 * len(checkpoint["model"])
            == case["optimizer_bytes"]
        )
        model = Transformer(ModelConfig(**case["model_config"]), case["seed"]).cuda().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        score, count = evaluate_forward(
            model, data.validation(row["batch"], row["context"], 10**9), "cuda", "bf16"
        )
        native = next(p for p in row["policies"] if p["policy"] == "native")
        assert score == native["nll"] and count == native["targets"]
        for p in row["policies"]:
            assert p["targets"] == count and p["order_sha256"] == native["order_sha256"]
            error = abs(p["nll"] / score - 1)
            assert p["relative_nll_difference_abs"] == error and p["score_passes"] == (
                error <= 0.0001
            )
            assert p["model_rng_gradients_mode_unchanged"]
            times = p["timing_ms"]["samples"]
            assert len(times) == 5 and all(math.isfinite(t) and t > 0 for t in times)
            assert p["timing_ms"]["median"] == st.median(times)
            assert p["timing_ms"]["mean"] == st.mean(times)
            assert p["timing_ms"]["sample_variance"] == st.variance(times)
            assert p["evaluation_time_ratio"] == st.median(times) / native["timing_ms"]["median"]
            assert (
                p["evaluation_memory_ratio"]
                == p["peak_allocated_bytes"] / native["peak_allocated_bytes"]
            )
            assert p["estimated_job_peak_bytes"] == max(
                case["peak_training_allocated_bytes"], p["peak_allocated_bytes"]
            )
        checks.append(
            {
                "label": row["label"],
                "step": row["checkpoint"]["step"],
                "native_rescore_difference": abs(score - native["nll"]),
                "targets": count,
                "policy_comparisons": 3,
                "exact_state_and_fixture_hashes": True,
            }
        )
        print(
            f"Audited {len(checks)}/24: {row['label']} step{row['checkpoint']['step']}", flush=True
        )
        del model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    assert len(checks) == 24
    summary = json.loads((ROOT / "summary.json").read_text())
    for group in summary["groups"]:
        states = [r for r in result["rows"] if r["context"] == group["context"]]
        selected_rows = {}
        eligible = []
        for policy in ("native", "classifier_chunks", "sequence_chunks"):
            entries = [p for r in states for p in r["policies"] if p["policy"] == policy]
            selected_rows[policy] = entries
            gates = {
                "all_scores_within_0_01_percent": all(
                    p["relative_nll_difference_abs"] <= 0.0001 for p in entries
                ),
                "all_evaluation_memory_savings_at_least_10_percent": all(
                    p["evaluation_memory_ratio"] <= 0.9 for p in entries
                ),
                "median_evaluation_time_ratio_at_most_1_5": st.median(
                    p["evaluation_time_ratio"] for p in entries
                )
                <= 1.5,
            }
            assert group["policies"][policy]["gates"] == gates
            qualifies = policy != "native" and all(gates.values())
            assert group["policies"][policy]["eligible"] == qualifies
            if qualifies:
                eligible.append(policy)
        chosen = (
            min(
                eligible,
                key=lambda p: (
                    st.median(r["estimated_job_peak_bytes"] for r in selected_rows[p]),
                    st.median(r["timing_ms"]["median"] for r in selected_rows[p]),
                    0 if p == "sequence_chunks" else 1,
                ),
            )
            if eligible
            else None
        )
        assert group["selected_policy"] == chosen
    write_json(
        ROOT / "audit.json",
        {
            "passed": True,
            "all_gates_and_policy_selections_independently_verified": True,
            "native_rescores": 24,
            "policy_comparisons": 72,
            "checks": checks,
            "optimizer_updates": 0,
            "max_native_rescore_difference": max(r["native_rescore_difference"] for r in checks),
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
