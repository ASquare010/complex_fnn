"""H057 fixed checkpoint probes; no optimizer updates or held-out scoring."""

import argparse
import copy
import hashlib
import json
import math
import zipfile
from pathlib import Path

ROOT = Path("results/residual_geometry_v1")
PLAN = Path("research/residual_geometry_plan.md")
CACHE = Path("data/wikitext2_v1")


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rms_vjp(x, gamma, adjoint, epsilon=1e-5):
    """Real-arithmetic RMSNorm VJP; callers choose reference precision explicitly."""
    s = (x.square().mean(-1, keepdim=True) + epsilon).sqrt()
    weighted = gamma * adjoint
    return weighted / s - x * (x * weighted).mean(-1, keepdim=True) / s.pow(3)


def relative(actual, expected):
    return (actual.double() - expected.double()).norm().item() / max(
        expected.double().norm().item(), 1e-30
    )


def quantiles(x):
    import torch

    x = x.detach().double().flatten()
    assert torch.isfinite(x).all()
    values = torch.quantile(
        x, torch.tensor([0.1, 0.5, 0.9], dtype=x.dtype, device=x.device)
    ).tolist()
    return dict(zip(("p10", "median", "p90"), values, strict=True))


def probe(model, x, y, precision):
    import torch
    from torch.nn import functional as F

    from src.core.benchmark import autocast
    from src.core.transformer import RMSNorm

    model.eval()
    model.zero_grad(set_to_none=True)
    norms, branches, handles = {}, {}, []
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()

    def hook_norm(name):
        def capture(module, inputs, output):
            output.retain_grad()
            norms[name] = (module, inputs[0].detach(), output)

        return capture

    def hook_ffn(name):
        def capture(module, inputs, output):
            output.retain_grad()
            branches[name] = output

        return capture

    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            handles.append(module.register_forward_hook(hook_norm(name)))
    for i, block in enumerate(model.blocks):
        handles.append(block.ffn.register_forward_hook(hook_ffn(str(i))))
    try:
        with autocast(str(x.device), precision):
            logits = model(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        loss.backward()
    finally:
        for h in handles:
            h.remove()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    records = {}
    for name, (module, raw, output) in norms.items():
        adjoint = output.grad.detach()
        local = raw.detach().requires_grad_(True)
        actual = torch.autograd.grad(module(local), local, adjoint)[0].detach()
        ref = rms_vjp(raw.double(), module.weight.detach().double(), adjoint.double())
        error = relative(actual, ref)
        assert error <= 5e-5, (name, error)
        s = (raw.double().square().mean(-1) + 1e-5).sqrt()
        gain = actual.double().norm(dim=-1) / adjoint.double().norm(dim=-1).clamp_min(1e-30)
        records[name] = {
            "input_rms": quantiles(raw.double().square().mean(-1).sqrt()),
            "inverse_scale": quantiles(1 / s),
            "unweighted_radial_gain": quantiles(1e-5 / s.pow(3)),
            "local_adjoint_gain": quantiles(gain),
            "local_vjp_relative_error": error,
            "gamma_abs_min": module.weight.detach().abs().min().item(),
            "gamma_abs_max": module.weight.detach().abs().max().item(),
            "output_adjoint_rms": adjoint.double().square().mean().sqrt().item(),
            "input_local_adjoint_rms": actual.double().square().mean().sqrt().item(),
        }
    branch_records = {
        name: {
            "output_rms": value.detach().double().square().mean().sqrt().item(),
            "output_adjoint_rms": value.grad.detach().double().square().mean().sqrt().item(),
        }
        for name, value in branches.items()
    }
    assert len(records) == 17 and len(branch_records) == 8
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(p, state[n]) for n, p in model.named_parameters())
    return {
        "probe_training_prefix_cross_entropy": loss.item(),
        "precision": precision,
        "norms": records,
        "ffns": branch_records,
        "weights_and_cpu_rng_unchanged": True,
        "all_parameter_gradients_finite": True,
    }, logits.detach()


def gauge_check(model, x, y, original_logits):
    import torch
    from torch.nn import functional as F

    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for block in model.blocks:
            block.ffn.value.mul_(8)
            block.ffn.down.div_(8)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
    loss.backward()
    logit_error = relative(logits, original_logits)
    errors, norm_ratios = {}, {}
    for n, p in model.named_parameters():
        expected = (
            gradients[n] / 8
            if n.endswith(".ffn.value")
            else gradients[n] * 8
            if n.endswith(".ffn.down")
            else gradients[n]
        )
        errors[n] = relative(p.grad, expected)
        assert errors[n] <= 5e-5, (n, errors[n])
        if n.endswith((".ffn.value", ".ffn.down")):
            norm_ratios[n] = p.grad.norm().item() / max(gradients[n].norm().item(), 1e-30)
    original_loss = F.cross_entropy(original_logits.float().flatten(0, 1), y.flatten()).item()
    assert logit_error <= 5e-5 and abs(loss.item() - original_loss) <= 5e-5
    model.load_state_dict(state)
    assert all(torch.equal(p, state[n]) for n, p in model.named_parameters())
    return {
        "logit_relative_error": logit_error,
        "loss_absolute_error": abs(loss.item() - original_loss),
        "maximum_predicted_gradient_relative_error": max(errors.values()),
        "gradient_norm_ratios": norm_ratios,
        "original_weights_restored_exactly": True,
        "optimizer_updates": 0,
        "modified_checkpoint_written": False,
    }


def spectra(model):
    import torch

    records = {}
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            maps = {}
            for name in ("input_projection", "output_projection"):
                projection = copy.deepcopy(getattr(block.ffn, name)).cpu().double()
                matrix = projection(torch.eye(projection.input_width, dtype=torch.float64)).T
                values = torch.linalg.svdvals(matrix)
                assert torch.isfinite(values).all()
                smallest, largest = values[-1].item(), values[0].item()
                maps[name] = {
                    "singular_min": smallest,
                    "singular_max": largest,
                    "condition_number": largest / smallest if smallest else None,
                    "frobenius_norm": matrix.norm().item(),
                    "numerical_rank": int(
                        (
                            values > largest * max(matrix.shape) * torch.finfo(torch.float64).eps
                        ).sum()
                    ),
                }
            records[str(i)] = {
                "maps": maps,
                "private_norms": {
                    name: quantiles(
                        getattr(block.ffn, name).detach().double().flatten(1).norm(dim=-1)
                    )
                    for name in ("gate", "value", "down")
                },
            }
    return records


def preflight():
    from src.core.reproducibility import environment, provenance

    assert not ROOT.exists()
    tested = read("results/verification/overcomplete_screen_tests_v2.json")
    assert tested["returncode"] == 0 and tested["source_unchanged"]
    assert all(sha(n) == h for n, h in tested["source_files"].items())
    final = read("results/verification/overcomplete_screen_final_v1.json")
    assert final["status"] == "PASS" and not final["earns_separate_longer_comparison"]
    assert sha("results/overcomplete_screen_v1/result.json") == final["result_sha256"]
    screen = read("results/overcomplete_screen_v1/result.json")
    old = read("results/overcomplete_screen_v1/preflight.json")
    selected = {**screen["references"], "overcomplete_headwise": screen["selected"]}
    controls = {r["run"]: r for r in old["controls"]}
    new = {r["run"]: r for r in screen["trials"]}
    runs = list(dict.fromkeys([*selected.values(), *new]))
    references = {}
    for run in runs:
        path = Path("results/runs") / run
        expected = (
            new[run]["files"]
            if run in new
            else {
                "metrics.json": controls[run]["metrics_sha256"],
                "checkpoint.pt": controls[run]["checkpoint_sha256"],
                "source.zip": controls[run]["source_archive_sha256"],
            }
        )
        assert all(sha(path / n) == h for n, h in expected.items())
        references[run] = {"path": path.as_posix(), "files": expected}
    assert all(sha(CACHE / n) == h for n, h in old["data_hashes"].items())
    protocol = {
        "plan_sha256": sha(PLAN),
        "provenance": provenance(),
        "environment": environment(),
        "data_hashes": old["data_hashes"],
        "selected": selected,
        "runs": runs,
        "references": references,
        "cpu_probes": 14,
        "gpu_probes": 6,
        "optimizer_updates": 0,
        "validation_or_test_targets_scored": 0,
    }
    ROOT.mkdir()
    write(ROOT / "protocol.json", protocol)
    with zipfile.ZipFile(ROOT / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for n in protocol["provenance"]["source_files"]:
            archive.write(n, n)
        for p in (PLAN, Path("research/normalization_geometry.md")):
            archive.write(p, p.as_posix())
    print(
        json.dumps({"status": "PASS", "verified_checkpoints": len(runs), "optimizer_updates": 0}),
        flush=True,
    )


def analyze(device):
    import torch

    from src.core.config import ModelConfig, TrainConfig
    from src.core.data import TokenData
    from src.core.optimization import initialize_dense_width
    from src.core.transformer import Transformer

    phase = "cpu" if device == "cpu" else "gpu"
    assert not (ROOT / f"{phase}.json").exists()
    protocol = read(ROOT / "protocol.json")
    assert sha(PLAN) == protocol["plan_sha256"]
    assert all(sha(n) == h for n, h in protocol["provenance"]["source_files"].items())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda":
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    data = TokenData(CACHE, device, 10017)
    sampler_before = data.generator.get_state().clone()
    x, y = data.train[:256].reshape(2, 128), data.train[1:257].reshape(2, 128)
    batch_hash = hashlib.sha256(x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()).hexdigest()
    cases = [(recipe + "_final", run, False) for recipe, run in protocol["selected"].items()]
    if device == "cpu":
        cases += [(recipe + "_initial", run, True) for recipe, run in protocol["selected"].items()]
        cases += [
            (run, run, False)
            for run in protocol["runs"]
            if run not in protocol["selected"].values()
        ]
    records = {}
    for label, run, initial in cases:
        path = Path(protocol["references"][run]["path"])
        assert all(sha(path / n) == h for n, h in protocol["references"][run]["files"].items())
        m = read(path / "metrics.json")
        mc, tc = ModelConfig(**m["model"]), TrainConfig(**m["training"])
        model = Transformer(mc, tc.seed)
        if initial:
            initialize_dense_width(model, tc)
        else:
            ckpt = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
            assert ckpt["step"] == 200 and ckpt["model_config"] == m["model"]
            model.load_state_dict(ckpt["model"])
            del ckpt
        model.to(device)
        precision = "fp32" if device == "cpu" else "bf16"
        record, logits = probe(model, x, y, precision)
        record.update(run=run, initial=initial)
        if label == "overcomplete_headwise_final" and device == "cpu":
            record["gauge_qualification"] = gauge_check(model, x, y, logits)
        if mc.variant == "overcomplete_headwise_swiglu" and device == "cpu":
            record["factor_geometry"] = spectra(model)
        records[label] = record
        write(ROOT / f"{phase}_partial.json", {"completed": records, "batch_sha256": batch_hash})
        print(
            json.dumps(
                {
                    "phase": phase,
                    "case": label,
                    "probe_cross_entropy": record["probe_training_prefix_cross_entropy"],
                    "maximum_vjp_error": max(
                        v["local_vjp_relative_error"] for v in record["norms"].values()
                    ),
                }
            ),
            flush=True,
        )
        del model, logits
    assert torch.equal(data.generator.get_state(), sampler_before)
    assert len(records) == protocol[f"{phase}_probes"]
    write(
        ROOT / f"{phase}.json",
        {
            "status": "complete",
            "batch_sha256": batch_hash,
            "records": records,
            "sampler_unchanged": True,
            "optimizer_updates": 0,
            "precision": precision,
        },
    )


def finish():
    protocol = read(ROOT / "protocol.json")
    cpu, gpu = read(ROOT / "cpu.json"), read(ROOT / "gpu.json")
    assert not (ROOT / "result.json").exists()
    assert (
        cpu["status"] == gpu["status"] == "complete" and cpu["batch_sha256"] == gpu["batch_sha256"]
    )
    assert sha(PLAN) == protocol["plan_sha256"]
    ratios, gains = {}, {}
    for phase, data in (("cpu", cpu), ("gpu", gpu)):
        gains[phase] = {
            recipe: math.exp(
                sum(
                    math.log(
                        data["records"][recipe + "_final"]["norms"][f"blocks.{i}.ffn_norm"][
                            "local_adjoint_gain"
                        ]["median"]
                    )
                    for i in range(1, 8)
                )
                / 7
            )
            for recipe in protocol["selected"]
        }
        ratios[phase] = {
            r: gains[phase]["overcomplete_headwise"] / gains[phase][r]
            for r in protocol["selected"]
            if r != "overcomplete_headwise"
        }
    write(
        ROOT / "result.json",
        {
            "status": "complete",
            "plan_sha256": sha(PLAN),
            "files": {p.name: sha(p) for p in ROOT.iterdir() if p.is_file()},
            "geometric_mean_later_ffn_norm_median_vjp_gain": gains,
            "candidate_to_control_local_gain_ratio": ratios,
            "ordering_agrees_cpu_bf16": all(
                (ratios["cpu"][r] < 1) == (ratios["gpu"][r] < 1) for r in ratios["cpu"]
            ),
            "optimizer_updates": 0,
            "validation_or_test_targets_scored": 0,
            "causal_explanation_established": False,
            "remedy_earned": False,
            "research_target_passes": False,
        },
    )
    print(json.dumps(read(ROOT / "result.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "cpu", "gpu", "finish"))
    args = parser.parse_args()
    if args.phase in ("cpu", "gpu"):
        analyze("cpu" if args.phase == "cpu" else "cuda")
    else:
        {"preflight": preflight, "finish": finish}[args.phase]()
