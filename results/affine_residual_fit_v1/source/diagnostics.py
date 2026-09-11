"""Loss/gradient evolution and shared PReLU shape, after all training is complete."""

import gc
import gzip
import json
import statistics as st
from pathlib import Path

import torch

from results.affine_residual_fit_v1.source.model import Student
from src.dense_ffn import DenseFFN

ROOT = Path("results/affine_residual_fit_v1")


@torch.no_grad()
def memory_profile(model, inputs):
    model.cuda().eval()
    fixed = inputs.cuda()
    for _ in range(10):
        model(fixed)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        model(fixed)
    torch.cuda.synchronize()
    record = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "live_model_and_input_bytes": torch.cuda.memory_allocated(),
    }
    model.cpu()
    del fixed
    gc.collect()
    torch.cuda.empty_cache()
    return record


def run():
    result = json.loads((ROOT / "result.json").read_text())
    records, slopes = [], []
    for row in result["rows"]:
        identity = {k: row[k] for k in ("teacher", "layer", "seed", "form")}
        for endpoint in row["endpoints"][1:]:
            path = Path(endpoint["checkpoint"]["path"]).with_suffix(".json")
            run = json.loads(path.read_text())
            history = run["history"]
            blocks = []
            for start in range(0, len(history), 50):
                chunk = history[start : start + 50]
                blocks.append(
                    {
                        "first_step": start + 1,
                        "last_step": start + len(chunk),
                        "mean_loss": st.mean(c["loss"] for c in chunk),
                        "mean_preclip_norm": st.mean(c["preclip_norm"] for c in chunk),
                        "max_preclip_norm": max(c["preclip_norm"] for c in chunk),
                        "clip_fraction": st.mean(c["preclip_norm"] > 1 for c in chunk),
                        "mean_update_ms": st.mean(c["update_ms"] for c in chunk),
                    }
                )
            records.append(
                {
                    **identity,
                    "rate": endpoint["rate"],
                    "blocks": blocks,
                    "activation_and_gradient_snapshots": run["diagnostics"],
                    "optimizer_bytes": run["optimizer_bytes"],
                    "all_finite": run["all_finite"],
                    "clip_fraction": run["clip_fraction"],
                }
            )
        if row["form"] == "prelu":
            values = []
            for endpoint in row["endpoints"]:
                state = torch.load(
                    endpoint["checkpoint"]["path"], map_location="cpu", weights_only=True
                )
                values.append(
                    {
                        "rate": endpoint["rate"],
                        "steps": endpoint["steps"],
                        "alpha": state["alpha"].item(),
                    }
                )
            slopes.append(
                {**identity, "endpoints": values, "selected_index": row["selected_index"]}
            )
    payload = {
        "runs": records,
        "prelu": slopes,
        "note": "Only PReLU learns a scalar activation parameter. Candidate nonlinearities are fixed GELU/SwiGLU; their projection weights learn. Curves are max(x,0)+alpha*min(x,0). Full per-update histories remain local.",
    }
    (ROOT / "diagnostics.json.gz").write_bytes(
        gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0)
    )
    summary = {
        "runs": len(records),
        "all_finite": all(r["all_finite"] for r in records),
        "max_preclip_norm": max(b["max_preclip_norm"] for r in records for b in r["blocks"]),
        "max_clip_fraction": max(r["clip_fraction"] for r in records),
        "forms": {
            form: {
                "mean_clip_fraction": st.mean(
                    r["clip_fraction"] for r in records if r["form"] == form
                ),
                "max_preclip_norm": max(
                    b["max_preclip_norm"] for r in records if r["form"] == form for b in r["blocks"]
                ),
            }
            for form in sorted({r["form"] for r in records})
        },
        "prelu_selected_slopes": [
            {
                "teacher": r["teacher"],
                "layer": r["layer"],
                "seed": r["seed"],
                "alpha": r["endpoints"][r["selected_index"]]["alpha"],
            }
            for r in slopes
        ],
    }
    (ROOT / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # This post-run measurement does not change the frozen training/latency gates.
    # It separately measures no-gradient deployment memory, with no corpus on GPU.
    memory, full_memory = [], []
    current_teacher, current_label, checkpoint, probe = None, None, None, None
    for row in result["rows"]:
        kind, layer, seed = row["teacher"], row["layer"], row["seed"]
        label = f"{kind}_l{layer}_s{seed}"
        if current_teacher != kind:
            checkpoint = torch.load(
                f"results/ungated_duration_v1/runs/full_{kind}_seed17/checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )["model"]
            current_teacher = kind
        if current_label != label:
            data = torch.load(
                ROOT / "pairs" / label / "data.pt", map_location="cpu", weights_only=True
            )
            probe = data["x"][:256].clone()
            del data
            prefix = f"blocks.{layer}.ffn."
            state = {
                k.removeprefix(prefix): v for k, v in checkpoint.items() if k.startswith(prefix)
            }
            full = DenseFFN(384, state["up.weight"].shape[0], kind)
            full.load_state_dict(state, strict=True)
            full_memory.append(
                {"teacher": kind, "layer": layer, "seed": seed, **memory_profile(full, probe)}
            )
            del full
            current_label = label
        model = Student(row["form"])
        model.load_state_dict(
            torch.load(row["raw_checkpoint"]["path"], map_location="cpu", weights_only=True)
        )
        memory.append(
            {
                "teacher": kind,
                "layer": layer,
                "seed": seed,
                "form": row["form"],
                **memory_profile(model, probe),
            }
        )
        del model
    (ROOT / "inference_memory.json").write_text(
        json.dumps(
            {
                "posthoc": True,
                "batch": 256,
                "gradients": False,
                "precision": "FP32",
                "warmup_forwards": 10,
                "measured_forwards": 10,
                "includes": "Raw folded FFN parameters, input batch and forward temporaries; no teacher, optimizer, normalization buffers or dataset on GPU",
                "students": memory,
                "full_teachers": full_memory,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run()
