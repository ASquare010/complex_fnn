"""Post-selection exact-algebra fold and native FP32 inference measurement."""

import gc
import os
import statistics as stats
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from results.blast_operator_recovery_v1.source.storage import (
    read,
    record_observation,
    sha,
    write_json,
)
from results.input_basis_fit_v1.source.study import make_model
from src.core.reproducibility import environment

ROOT = Path("results/input_basis_fold_v1")
OLD = Path("results/input_basis_fit_v1")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


class FoldedEven(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.up = nn.Linear(384, 176, bias=False).to(original.up.weight)
        self.down = nn.Linear(560, 384, bias=False).to(original.down.weight)
        with torch.no_grad():
            self.up.weight.copy_(original.up.weight)
            a, v, w = torch.split(original.down.weight, [384, 176, 176], dim=1)
            self.down.weight.copy_(torch.cat((a, v + w), 1))

    def forward(self, x):
        z = self.up(x)
        even = z * (z / (1 + z.abs()))
        return self.down(torch.cat((x, even), -1))


def cpu_pair(model, x, cotangent):
    x = x.detach().requires_grad_()
    y = model(x)
    grad = torch.autograd.grad(y, x, cotangent)[0]
    return [y.detach().cpu(), grad.detach().cpu()]


@torch.inference_mode()
def score(model, x, y):
    total = 0.0
    for start in range(0, len(x), 256):
        prediction = model(x[start : start + 256].cuda())
        total += (prediction - y[start : start + 256].cuda()).square().sum().item()
    return total / y.numel()


@torch.inference_mode()
def measure(model, probe):
    for _ in range(20):
        model(probe)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    device_ms, wall_ms = [], []
    for _ in range(7):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        for _ in range(20):
            model(probe)
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - start) * 1000 / 20)
        device_ms.append(begin.elapsed_time(end) / 20)
    return {
        "gpu_event_ms": device_ms,
        "wall_ms": wall_ms,
        "median_gpu_event_ms": stats.median(device_ms),
        "median_wall_ms": stats.median(wall_ms),
        "peak_bytes": torch.cuda.max_memory_allocated(),
    }


def run():
    previous = read(OLD / "result.json")
    assert previous["status"] == "PASS"
    hashes = dict(read(OLD / "before.json")["source_hashes"])
    hashes[Path(__file__).relative_to(Path.cwd()).as_posix()] = sha(__file__)
    hashes["research/input_basis_fold_plan.md"] = sha("research/input_basis_fold_plan.md")
    assert all(sha(p) == h for p, h in hashes.items())
    write_json(
        ROOT / "before.json",
        {
            "source_hashes": hashes,
            "h094_result_sha256": sha(OLD / "result.json"),
            "optimizer_updates": 0,
        },
    )
    write_json(
        ROOT / "protocol.json",
        {
            **read(ROOT / "before.json"),
            "environment": environment(),
            "fp64_atol_rtol": 1e-11,
            "fp32_atol_rtol": 1e-5,
            "warmups": 20,
            "timing_blocks": 7,
            "forwards_per_block": 20,
            "batch": 256,
        },
    )
    (ROOT / "observations").mkdir()
    raw = read(OLD / "raw_result.json")
    assert sha(OLD / "data.pt") == raw["data_sha256"]
    data = torch.load(OLD / "data.pt", map_location="cpu", weights_only=True)
    selected = [r for r in raw["rows"] if r["selected"]]
    lookup = {(r["task"], r["seed"], r["form"]): r for r in selected}
    results = []
    checkpoint_hashes = {}
    duplicates = [r for r in selected if r["form"] == "duplicate"]
    for index, row in enumerate(duplicates):
        task, seed = row["task"], row["seed"]

        def load(form):
            selected_row = lookup[task, seed, form]
            checkpoint_hashes[selected_row["checkpoint"]] = selected_row["checkpoint_sha256"]
            assert sha(selected_row["checkpoint"]) == selected_row["checkpoint_sha256"]
            state = torch.load(selected_row["checkpoint"], map_location="cpu", weights_only=True)
            model = make_model(form, seed)
            model.load_state_dict(state["model"])
            return model

        original = load("duplicate")
        probe_cpu = data["x"][69632:69888]
        cotangent = torch.randn(256, 384, generator=torch.Generator().manual_seed(9850))
        if index == 0:
            original.double()
            folded = FoldedEven(original)
            a, b = (cpu_pair(m, probe_cpu.double(), cotangent.double()) for m in (original, folded))
            for aa, bb in zip(a, b):
                torch.testing.assert_close(aa, bb, atol=1e-11, rtol=1e-11)
            record_observation(
                ROOT / "observations",
                "fp64",
                "fold",
                {"atol_rtol": 1e-11},
                {"original": a, "folded": b},
            )
            original.float()
            del folded
        original.cuda()
        folded = FoldedEven(original)
        assert sum(p.numel() for p in folded.parameters()) == 282624
        probe, cotangent_gpu = probe_cpu.cuda(), cotangent.cuda()
        a, b = (cpu_pair(m, probe, cotangent_gpu) for m in (original, folded))
        for aa, bb in zip(a, b):
            torch.testing.assert_close(aa, bb, atol=1e-5, rtol=1e-5)
        mse = score(folded, data["x"][69632:], data["targets"][task][69632:])
        delta = abs(mse - row["reporting_mse"])
        assert delta <= 1e-5 and delta / row["reporting_mse"] <= 1e-5
        record_observation(
            ROOT / "observations",
            f"{task}_{seed}",
            "fold",
            {
                "atol_rtol": 1e-5,
                "reporting_mse": mse,
                "original_mse": row["reporting_mse"],
                "absolute_mse_change": delta,
            },
            {"original": a, "folded": b},
        )
        del folded, original, cotangent_gpu
        gc.collect()
        torch.cuda.empty_cache()
        forms = ("folded", "duplicate", "narrow_gelu", "full_gelu", "full_swiglu")
        offset = index % len(forms)
        timing = {}
        for form in forms[offset:] + forms[:offset]:
            if form == "folded":
                base = load("duplicate")
                model = FoldedEven(base).cuda()
                del base
            else:
                model = load(form).cuda()
            timing[form] = measure(model, probe)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        del probe
        results.append(
            {
                "task": task,
                "seed": seed,
                "folded_mse": mse,
                "original_mse": row["reporting_mse"],
                "timing": timing,
            }
        )
        print(task, seed, "PASS", flush=True)
    observations = list((ROOT / "observations").glob("*.json"))
    assert len(results) == 12 and len(observations) == 13
    for path in observations:
        obs = read(path)
        p = path.with_name(obs["tensor_file"])
        assert sha(p) == obs["tensor_sha256"]
        pair = torch.load(p, map_location="cpu", weights_only=True)
        for a, b in zip(pair["original"], pair["folded"]):
            torch.testing.assert_close(a, b, atol=obs["atol_rtol"], rtol=obs["atol_rtol"])
    assert all(sha(p) == h for p, h in hashes.items())
    assert all(sha(p) == h for p, h in checkpoint_hashes.items())
    write_json(
        ROOT / "result.json",
        {
            "status": "PASS",
            "optimizer_updates": 0,
            "folded_parameters": 282624,
            "ffn_reduction_percent": 100 * (1 - 282624 / 1179648),
            "checkpoint_checks": 12,
            "saved_tensor_pairs_rechecked": 26,
            "rows": results,
            "mean_median_gpu_event_ms": {
                form: stats.mean(r["timing"][form]["median_gpu_event_ms"] for r in results)
                for form in forms
            },
            "checkpoint_hashes": checkpoint_hashes,
            "observations": {str(p): sha(p) for p in observations},
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "research_goal_achieved": False,
        },
    )


if __name__ == "__main__":
    status = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "fold_process.json", status)
    start = time.perf_counter()
    try:
        run()
        status["status"] = "PASS"
    except BaseException as exc:
        status.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
        raise
    finally:
        status.update(
            elapsed_seconds=time.perf_counter() - start,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(ROOT / "fold_process.json", status, exclusive=False)
