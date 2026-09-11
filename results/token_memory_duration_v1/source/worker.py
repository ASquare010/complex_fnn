"""One fresh, isolated 800-update H110 trial. Existing output is never overwritten."""

import gc
import hashlib
import json
import math
import statistics as st
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from results.token_memory_duration_v1.source.common import (
    configurations,
    evaluate,
    state_hashes,
    tensor_hash,
)
from results.token_memory_v1.source.execution import install
from src.core.benchmark import autocast
from src.core.data import TokenData
from src.core.diagnostics import gradient_stats, inspect_layers
from src.core.optimization import group_summary, parameter_groups
from src.core.reproducibility import sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/token_memory_duration_v1")


def run(batch, context, seed, policy):
    label = f"b{batch}_t{context}_s{seed}_{policy}"
    folder = ROOT / "runs" / label
    folder.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    data = TokenData(Path("data/wikitext2_v1"), "cuda", seed + 10000)
    cfg, tc = configurations(batch, context, seed)
    model = Transformer(cfg, seed).cuda()
    initial_hashes = state_hashes(model)
    install(model, policy, 512)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc), lr=tc.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    fixed_x, _ = next(data.validation(batch, context, 1))
    torch.cuda.reset_peak_memory_stats()
    subset = evaluate(model, data, batch, context, 16)
    evaluations = [{"step": 0, **subset}]
    model.eval()
    diagnostics = [
        {"step": 0, "activations": inspect_layers(model, fixed_x, "cuda", "bf16"), "gradients": {}}
    ]
    model.train()
    initial_rng = tensor_hash(data.generator.get_state())
    records, checkpoints = [], []
    input_order = hashlib.sha256()
    beginning = time.perf_counter()
    validation_peak = torch.cuda.max_memory_allocated()
    job_peak = validation_peak
    for step in range(1, tc.steps + 1):
        x, y = data.batch(batch, context)
        input_order.update(tensor_hash(x).encode())
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        job_peak = max(job_peak, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with autocast("cuda", "bf16"):
            loss = model.loss(x, y)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        record = {
            "step": step,
            "loss": loss.item(),
            "preclip_norm": norm.item(),
            "forward_ms": 1000 * (forward_end - start),
            "backward_ms": 1000 * (backward_end - forward_end),
            "optimizer_ms": 1000 * (end - backward_end),
            "update_ms": 1000 * (end - start),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        assert math.isfinite(record["loss"]) and math.isfinite(record["preclip_norm"])
        records.append(record)
        job_peak = max(job_peak, record["peak_allocated_bytes"])
        with (folder / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if step in (100, 200, 400, 800):
            torch.cuda.reset_peak_memory_stats()
            value = evaluate(model, data, batch, context, 16)
            evaluations.append({"step": step, **value})
            gradients = gradient_stats(model)
            model.eval()
            diagnostics.append(
                {
                    "step": step,
                    "activations": inspect_layers(model, fixed_x, "cuda", "bf16"),
                    "gradients": gradients,
                }
            )
            model.train()
            validation_peak = max(validation_peak, torch.cuda.max_memory_allocated())
            job_peak = max(job_peak, validation_peak)
            if step in (100, 400, 800):
                path = folder / f"step{step}.pt"
                checkpoint = {
                    "model_config": asdict(cfg),
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "seed": seed,
                    "step": step,
                }
                if step == 800:
                    # Convert the optimizer recursively only for the final durable state.
                    def cpu(value):
                        if isinstance(value, torch.Tensor):
                            return value.detach().cpu()
                        if isinstance(value, dict):
                            return {k: cpu(v) for k, v in value.items()}
                        if isinstance(value, list):
                            return [cpu(v) for v in value]
                        return value

                    checkpoint["optimizer"] = cpu(optimizer.state_dict())
                    checkpoint["sampler_state"] = data.generator.get_state().cpu()
                torch.save(checkpoint, path)
                checkpoints.append({"step": step, "path": path.as_posix(), "sha256": sha256(path)})
                del checkpoint
            print(
                json.dumps(
                    {
                        "trial": label,
                        "step": step,
                        "subset_nll": value["nll"],
                        "training_peak_mib": max(r["peak_allocated_bytes"] for r in records)
                        / 2**20,
                        "elapsed_seconds": time.perf_counter() - beginning,
                    }
                ),
                flush=True,
            )
    job_peak = max(job_peak, torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    full = evaluate(model, data, batch, context, 10**9)
    validation_peak = max(validation_peak, torch.cuda.max_memory_allocated())
    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters()) and all(
        bool(torch.isfinite(v).all())
        for state in optimizer.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    )
    assert finite
    timed = records[20:]
    output = {
        "status": "COMPLETE",
        "label": label,
        "batch": batch,
        "context": context,
        "seed": seed,
        "policy": policy,
        "model_config": asdict(cfg),
        "training_config": asdict(tc),
        "optimizer_groups": group_summary(optimizer.param_groups),
        "initial_state_hashes": initial_hashes,
        "initial_sampler_sha256": initial_rng,
        "final_sampler_sha256": tensor_hash(data.generator.get_state()),
        "data_order_sha256": input_order.hexdigest(),
        "checkpoints": checkpoints,
        "evaluations": evaluations,
        "full_validation": full,
        "records": records,
        "diagnostics": diagnostics,
        "steps": 800,
        "training_targets": 800 * batch * context,
        "parameter_count": cfg.total_parameters,
        "ffn_parameter_count": cfg.unique_ffn_parameters,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "gradient_bytes": sum(p.grad.numel() * p.grad.element_size() for p in model.parameters()),
        "optimizer_bytes": sum(
            v.numel() * v.element_size()
            for state in optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        ),
        "peak_training_allocated_bytes": max(r["peak_allocated_bytes"] for r in records),
        "peak_training_reserved_bytes": max(r["peak_reserved_bytes"] for r in records),
        "peak_validation_allocated_bytes": validation_peak,
        "peak_job_allocated_bytes": max(
            job_peak, validation_peak, *(r["peak_allocated_bytes"] for r in records)
        ),
        "weights_and_moments_finite": finite,
        "clip_fraction": sum(r["preclip_norm"] > 1 for r in records) / 800,
        "timing": {
            k: {
                "mean": st.mean(r[k] for r in timed),
                "median": st.median(r[k] for r in timed),
                "sample_variance": st.variance(r[k] for r in timed),
            }
            for k in ("update_ms", "forward_ms", "backward_ms", "optimizer_ms")
        },
        "early_late_update_ms": {
            "21_to_100": st.mean(r["update_ms"] for r in records[20:100]),
            "701_to_800": st.mean(r["update_ms"] for r in records[700:]),
        },
        "wall_seconds": time.perf_counter() - beginning,
    }
    write_json(folder / "metrics.json", output)
    print(
        json.dumps(
            {
                "completed": label,
                "full_nll": full["nll"],
                "peak_mib": output["peak_training_allocated_bytes"] / 2**20,
                "median_update_ms": output["timing"]["update_ms"]["median"],
            }
        ),
        flush=True,
    )
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = sys.argv[1:]
    try:
        run(int(args[0]), int(args[1]), int(args[2]), args[3])
    except Exception:
        import traceback

        write_json(
            ROOT / f"worker_failure_b{args[0]}_t{args[1]}_s{args[2]}_{args[3]}.json",
            {"traceback": traceback.format_exc()},
        )
        raise
