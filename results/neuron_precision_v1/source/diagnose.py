"""One H089 diagnostic of where activation values and gradients round."""

import json
from pathlib import Path

import torch

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.neuron_geometry_v1.source.model import CANDIDATES, GeometryFFN
from results.neuron_geometry_v1.source.study import save_tensor
from results.neuron_geometry_v1.source.test_qualification import reference

ROOT = Path("results/neuron_precision_v1")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def comparisons(a, b, atol, rtol):
    result = {}
    assert a.keys() == b.keys()
    for name in a:
        first, second = a[name], b[name]
        result[name] = {
            "elements": first.numel(),
            "mismatches": int((~torch.isclose(first, second, atol=atol, rtol=rtol)).sum()),
            "max_abs": (first.double() - second.double()).abs().max().item(),
            "exact": torch.equal(first, second),
        }
    return result


def main():
    before = read(ROOT / "before.json")
    assert all(sha(name) == value for name, value in before["source_hashes"].items())
    rows = []
    for form in CANDIDATES:
        for precision in ("fp32", "bf16"):
            model = GeometryFFN(form).cuda()
            with torch.no_grad():
                model.curve.theta.copy_(
                    torch.linspace(-0.25, 0.3, model.curve.theta.numel(), device="cuda").reshape_as(
                        model.curve.theta
                    )
                )
            generator = torch.Generator().manual_seed(9830)
            original = torch.randn(3, 2, 384, generator=generator).transpose(0, 1).cuda()
            cotangent = torch.randn(2, 3, 384, generator=generator).cuda()
            modes = {}
            for mode in ("native", "reference", "island", "reference_island"):
                x = original.detach().clone().requires_grad_()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                    z = model.up(x)
                    if mode in ("island", "reference_island"):
                        nonlinear = (
                            model.curve(z.float())
                            if mode == "island"
                            else reference(z.float(), model.curve)
                        )
                        activated = nonlinear.to(z.dtype)
                    else:
                        activated = (
                            model.curve(z) if mode == "native" else reference(z, model.curve)
                        )
                    output = model.down(activated)
                gradients = torch.autograd.grad(output, (x, *model.parameters()), cotangent)
                names = ["output", "input_gradient", *dict(model.named_parameters())]
                tensors = dict(zip(names, (output, *gradients)))
                assert all(torch.isfinite(value).all() for value in tensors.values())
                modes[mode] = {name: value.detach().cpu() for name, value in tensors.items()}
            atol = rtol = 0.02 if precision == "bf16" else 1e-5
            original_pair = comparisons(modes["native"], modes["reference"], atol, rtol)
            island_pair = comparisons(modes["island"], modes["reference_island"], atol, rtol)
            unchanged = comparisons(modes["native"], modes["island"], 0, 0)
            name = f"{form}_{precision}"
            path = ROOT / (name + ".pt")
            save_tensor(path, modes)
            row = {
                "form": form,
                "precision": precision,
                "atol": atol,
                "rtol": rtol,
                "original_reference": original_pair,
                "island_reference": island_pair,
                "native_vs_island": unchanged,
                "tensor_file": path.as_posix(),
                "tensor_sha256": sha(path),
            }
            write_json(ROOT / (name + ".json"), row)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "form": form,
                        "precision": precision,
                        "original_mismatches": sum(
                            value["mismatches"] for value in original_pair.values()
                        ),
                        "island_mismatches": sum(
                            value["mismatches"] for value in island_pair.values()
                        ),
                        "native_island_exact": all(value["exact"] for value in unchanged.values()),
                    }
                ),
                flush=True,
            )
    preserved = {}
    for name, metadata in before["h088_files"].items():
        p = Path(name)
        preserved[name] = (
            sha(p) == metadata["sha256"]
            and p.stat().st_size == metadata["bytes"]
            and p.stat().st_mtime_ns == metadata["mtime_ns"]
        )
    assert all(preserved.values())
    assert all(sha(name) == value for name, value in before["source_hashes"].items())
    fp32_exact = all(
        value["exact"]
        for row in rows
        if row["precision"] == "fp32"
        for value in row["native_vs_island"].values()
    )
    island_passes = all(
        value["mismatches"] == 0 for row in rows for value in row["island_reference"].values()
    )
    result = {
        "status": "COMPLETE",
        "evaluations": 32,
        "optimizer_updates": 0,
        "training_examples": 0,
        "original_h088_preserved": True,
        "fp32_native_island_exact": fp32_exact,
        "island_reference_all_pass": island_passes,
        "earns_explicit_qualification_recovery": fp32_exact and island_passes,
        "rows": rows,
        "research_goal_achieved": False,
        "before_sha256": sha(ROOT / "before.json"),
    }
    write_json(ROOT / "result.json", result)
    print(
        json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
