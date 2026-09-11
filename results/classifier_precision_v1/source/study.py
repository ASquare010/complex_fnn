"""Fixed-state classifier gradients and local forward/backward resource comparison."""

import gc
import json
import statistics as st
import time
from pathlib import Path

import sympy
import torch
import torch._dynamo

from results.classifier_precision_v1.source.capture import capture_all
from results.classifier_precision_v1.source.common import (
    POLICIES,
    clear,
    error,
    inspect_casts,
    loss_for,
    qualify,
)
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/classifier_precision_v1")


def gradient_run(payload, policy, reference=False):
    clear()
    dtype = torch.float64 if reference else torch.float32
    h = payload["hidden"].to(device="cuda", dtype=dtype).requires_grad_()
    w = payload["weight"].to(device="cuda", dtype=dtype).requires_grad_()
    y = payload["targets"].cuda()
    repetitions, records, first, cast_info = (1 if reference else 4), [], None, None
    for repeat in range(repetitions):
        h.grad = w.grad = None
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = loss_for(h, w, y, policy)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        handles = []
        if repeat == 0 and not reference:
            events, handles, count = inspect_casts(loss, w)
            cast_info = {"weight_cast_nodes": count, "events": events}
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        output = {
            "loss": loss.item(),
            "hidden_gradient": h.grad.detach().cpu(),
            "weight_gradient": w.grad.detach().cpu(),
        }
        assert all(torch.isfinite(v).all().item() for v in (h.grad, w.grad))
        assert torch.isfinite(loss).item()
        if first is not None:
            assert output["loss"] == first["loss"]
            assert all(
                torch.equal(output[k], first[k]) for k in ("hidden_gradient", "weight_gradient")
            )
        first = output
        if reference or repeat > 0:
            records.append(
                {
                    "forward_ms": 1000 * (forward_end - start),
                    "backward_ms": 1000 * (end - forward_end),
                    "total_ms": 1000 * (end - start),
                    "peak_allocated_bytes": peak,
                    "peak_reserved_bytes": reserved,
                }
            )
        for handle in handles:
            handle.remove()
        del loss, handles
    del h, w, y
    clear()
    return first, {
        "repetitions": records,
        "cast_graph": cast_info,
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in records),
        "median_total_ms": st.median(r["total_ms"] for r in records),
    }


def run():
    assert not torch.cuda.is_initialized()
    print(f"CPU preload passed: sympy={sympy.__version__}, torch={torch.__version__}", flush=True)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    start = time.perf_counter()
    qualification = qualify()
    write_json(ROOT / "qualification.json", qualification)
    clear()
    captures = capture_all(protocol)
    write_json(ROOT / "captures.json", captures)
    rows = []
    for index, capture in enumerate(captures):
        payload = torch.load(capture["path"], map_location="cpu", weights_only=True)
        reference, ref_measure = gradient_run(payload, "native_fp32", reference=True)
        outputs = {"native_fp64": reference}
        records = []
        rotated = POLICIES[index % len(POLICIES) :] + POLICIES[: index % len(POLICIES)]
        for policy in rotated:
            gradients, measures = gradient_run(payload, policy)
            outputs[policy] = gradients
            records.append(
                {
                    "policy": policy,
                    "loss": gradients["loss"],
                    "relative_loss_error": abs(gradients["loss"] / reference["loss"] - 1),
                    **{
                        k: error(gradients[k], reference[k])
                        for k in ("hidden_gradient", "weight_gradient")
                    },
                    **measures,
                }
            )
        cached, uncached = [outputs[p] for p in ("chunks_cached", "chunks_uncached")]
        comparison = {
            "loss_equal": cached["loss"] == uncached["loss"],
            "hidden_gradient_equal": torch.equal(
                cached["hidden_gradient"], uncached["hidden_gradient"]
            ),
            "weight_gradient_equal": torch.equal(
                cached["weight_gradient"], uncached["weight_gradient"]
            ),
            "weight_difference": error(uncached["weight_gradient"], cached["weight_gradient"]),
        }
        target = ROOT / "gradients" / (capture["label"] + ".pt")
        assert not target.exists()
        torch.save(outputs, target)
        row = {
            **capture,
            "reference_measurement": ref_measure,
            "records": records,
            "comparison": comparison,
            "gradients_path": target.as_posix(),
            "gradients_sha256": sha256(target),
        }
        rows.append(row)
        write_json(
            ROOT / "progress.json",
            {
                "completed_fixtures": len(rows),
                "classifier_backward_passes": 21 * len(rows),
                "optimizer_updates": 0,
            },
        )
        print(
            json.dumps(
                {
                    "fixture": capture["label"],
                    "completed": len(rows),
                    "comparison": comparison,
                    "weight_errors": {
                        r["policy"]: r["weight_gradient"]["relative_l2"] for r in records
                    },
                }
            ),
            flush=True,
        )
        del payload, reference, outputs, gradients, cached, uncached
    write_json(
        ROOT / "result.json",
        {
            "status": "COMPLETE",
            "rows": rows,
            "fixtures": 24,
            "methods": 5,
            "classifier_backward_passes": 504,
            "optimizer_updates": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "full_training_job_measured": False,
            "broad_goal_achieved": False,
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

        write_json(
            ROOT / "failure.json", {"traceback": traceback.format_exc(), "optimizer_updates": 0}
        )
        raise
