"""Independent autograd derivatives, reconstructed statistics and deployment scores."""

import gc
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.core.reproducibility import sha256, write_json

ROOT = Path("results/sobolev_selection_v1")


def read(path):
    return json.loads(Path(path).read_text())


def hidden(state, x):
    z = F.linear(x, state["up.weight"])
    return (
        F.gelu(z) if "gate.weight" not in state else F.silu(z) * F.linear(x, state["gate.weight"])
    )


def forward(state, x):
    return F.linear(hidden(state, x), state["down.weight"], state.get("down.bias"))


def probes(x, sx, seed):
    return (
        torch.randint(0, 2, x.shape, generator=torch.Generator().manual_seed(seed)).float() * 2 - 1
    ) * sx.float()


@torch.no_grad()
def automatic(state, x, v, feature: bool, batch: int):
    cuda = {k: a.cuda() for k, a in state.items()}
    values, derivatives = [], []
    function = (lambda a: hidden(cuda, a)) if feature else (lambda a: forward(cuda, a))
    for start in range(0, len(x), batch):
        value, derivative = torch.func.jvp(
            function, (x[start : start + batch].cuda(),), (v[start : start + batch].cuda(),)
        )
        values.append(value.cpu())
        derivatives.append(derivative.cpu())
    return torch.cat(values), torch.cat(derivatives)


def reconstruct(state, x, seed):
    sx = x.double().std(0, correction=0).clamp_min(1e-3)
    h, k = automatic(state, x, probes(x, sx, 27000 + seed), True, 512)
    # Compute target on CUDA exactly as the original layer, but derive K by autograd.
    dw = state["down.weight"].cuda()
    y = torch.cat([F.linear(part.cuda(), dw).cpu() for part in h.split(512)])
    t = torch.cat([F.linear(part.cuda(), dw).cpu() for part in k.split(512)])
    del dw
    h, k, y, t = (a.double() for a in (h, k, y, t))
    mean_h, mean_y = h.mean(0), y.mean(0)
    scale_h, scale_y = (
        h.std(0, correction=0).clamp_min(1e-6),
        (y - mean_y).square().mean().sqrt().clamp_min(1e-6),
    )
    hv, kv, yy, tt = (h - mean_h) / scale_h, k / scale_h, (y - mean_y) / scale_y, t / scale_y
    energy = tt.square().mean().item()
    return {
        "value_gram": hv.T @ hv / len(x),
        "derivative_gram": kv.T @ kv / len(x),
        "value_cross": hv.T @ yy / len(x),
        "derivative_cross": kv.T @ tt / len(x),
        "mean_h": mean_h,
        "mean_y": mean_y,
        "scale_h": scale_h,
        "scale_y": scale_y.item(),
        "sx": sx,
        "derivative_energy": energy,
        "beta": 0.1 / max(energy, 1e-12),
    }


def check_gains(stats, orders, traces, primary, half):
    checks = []
    for name in ("variance_pivot", "value_greedy", "sobolev_greedy"):
        beta = stats["beta"] if name == "sobolev_greedy" else 0
        g = (
            stats["value_gram"]
            + beta * stats["derivative_gram"]
            + 1e-4 * torch.eye(len(stats["value_gram"]), dtype=torch.float64)
        )
        c = stats["value_cross"] + beta * stats["derivative_cross"]
        order = orders[name]
        for n in (0, 1, 7, 31, 127, primary - 1, half - 1, len(order) - 1):
            selected = order[:n]
            if n:
                solved = torch.linalg.solve(
                    g[selected][:, selected], torch.cat((g[selected], c[selected]), dim=1)
                )
                diag = g.diag() - (g[:, selected] * solved[:, : len(g)].T).sum(1)
                cross = c - g[:, selected] @ solved[:, len(g) :]
            else:
                diag, cross = g.diag().clone(), c.clone()
            gain = cross.square().sum(1) / diag.clamp_min(1e-30)
            j = int(order[n])
            recorded = traces[name]["gains"][n]
            delta = abs(gain[j].item() - recorded)
            assert delta <= 1e-7 + 3e-5 * abs(recorded), (name, n, delta)
            ranking = diag.clone() if name == "variance_pivot" else gain.clone()
            ranking[selected] = -torch.inf
            assert ranking[j] >= ranking.max() - (1e-7 + 3e-5 * abs(ranking[j].item()))
            checks.append(
                {
                    "selector": name,
                    "prefix_length": n,
                    "gain_difference": delta,
                    "maximal_gain_or_variance": True,
                }
            )
    return checks


def run():
    result, protocol = read(ROOT / "result.json"), read(ROOT / "protocol.json")
    assert len(result["rows"]) == 432 and result["neural_updates"] == 0
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    assert (
        sha256(Path("results/affine_residual_fit_v1/result.json"))
        == protocol["prior_result_sha256"]
    )
    records, scores, gains = [], [], []
    for kind, teacher_info in protocol["teachers"].items():
        assert sha256(Path(teacher_info["path"])) == teacher_info["sha256"]
        full = torch.load(teacher_info["path"], map_location="cpu", weights_only=True)["model"]
        for case in [c for c in result["cases"] if c["teacher"] == kind]:
            layer, seed = case["layer"], case["seed"]
            label = f"{kind}_l{layer}_s{seed}"
            folder = ROOT / "cases" / label
            original = Path("results/affine_residual_fit_v1/pairs") / label / "data.pt"
            assert sha256(original) == case["input_data_sha256"]
            data = torch.load(original, map_location="cpu", weights_only=True)
            x, report = data["x"][:8192].clone(), data["x"][-4096:].clone()
            del data
            assert sha256(folder / "statistics.pt") == case["statistics_sha256"]
            saved = torch.load(folder / "statistics.pt", map_location="cpu", weights_only=True)
            stats, orders, traces = saved["statistics"], saved["orders"], saved["traces"]
            prefix = f"blocks.{layer}.ffn."
            teacher = {k.removeprefix(prefix): v for k, v in full.items() if k.startswith(prefix)}
            rebuilt = reconstruct(teacher, x, seed)
            differences = {}
            for key, value in rebuilt.items():
                expected = stats[key]
                value, expected = torch.as_tensor(value), torch.as_tensor(expected)
                derivative_key = key.startswith("derivative") or key == "beta"
                torch.testing.assert_close(
                    value,
                    expected,
                    atol=2e-6 if derivative_key else 2e-9,
                    rtol=1e-5 if derivative_key else 1e-9,
                )
                differences[key] = (value - expected).abs().max().item()
            gains.extend(
                {"label": label, **g}
                for g in check_gains(
                    stats,
                    orders,
                    traces,
                    protocol["widths"][kind]["primary"],
                    protocol["widths"][kind]["half"],
                )
            )
            keep = max(protocol["widths"][kind].values())
            random = torch.randperm(
                len(stats["scale_h"]), generator=torch.Generator().manual_seed(28000 + seed)
            )[:keep]
            torch.testing.assert_close(random, orders["random"], atol=0, rtol=0)
            norm = teacher["down.weight"].double().norm(dim=0)
            for method, importance in (
                ("weight_norm", norm),
                ("fluctuation", norm * stats["scale_h"]),
            ):
                torch.testing.assert_close(
                    torch.argsort(importance, descending=True, stable=True)[:keep],
                    orders[method],
                    atol=0,
                    rtol=0,
                )
            assert all(len(set(order.tolist())) == keep for order in orders.values())
            assert sha256(folder / "reference.pt") == case["reference_sha256"]
            reference = torch.load(folder / "reference.pt", map_location="cpu", weights_only=True)
            report_probes = [
                probes(report, stats["sx"], offset + seed) for offset in (29000, 30000)
            ]
            for index, probe in enumerate(report_probes):
                y, t = automatic(teacher, report, probe, False, 257)
                torch.testing.assert_close(y, reference["value"], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(t, reference["derivative"][index], atol=2e-6, rtol=2e-5)
            target_energy = reference["derivative"].double().square().mean().item()
            records.append(
                {
                    "label": label,
                    "statistics_max_differences": differences,
                    "autograd_reference_values_and_derivatives": True,
                    "orders": True,
                }
            )
            for row in [
                r
                for r in result["rows"]
                if (r["teacher"], r["layer"], r["seed"]) == (kind, layer, seed)
            ]:
                path = Path(row["checkpoint"]["path"])
                assert sha256(path) == row["checkpoint"]["sha256"]
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                state, selected, coefficient = (
                    checkpoint["state_dict"],
                    checkpoint["selected"],
                    checkpoint["coefficient"],
                )
                assert all(bool(torch.isfinite(v).all()) for v in state.values())
                method = protocol["methods"][row["method"]]
                torch.testing.assert_close(
                    selected, orders[method[0]][: row["hidden"]], atol=0, rtol=0
                )
                assert selected.tolist() == row["selected"]
                beta = stats["beta"] if method[1] else 0
                gram = (
                    stats["value_gram"]
                    + beta * stats["derivative_gram"]
                    + 1e-4 * torch.eye(len(stats["scale_h"]), dtype=torch.float64)
                )
                cross = stats["value_cross"] + beta * stats["derivative_cross"]
                residual = (
                    (gram[selected][:, selected] @ coefficient - cross[selected]).abs().max().item()
                )
                assert residual < 1e-8
                raw_weights = coefficient / stats["scale_h"][selected, None] * stats["scale_y"]
                raw_bias = stats["mean_y"] - stats["mean_h"][selected] @ raw_weights
                torch.testing.assert_close(
                    state["up.weight"], teacher["up.weight"][selected], atol=0, rtol=0
                )
                if "gate.weight" in state:
                    torch.testing.assert_close(
                        state["gate.weight"], teacher["gate.weight"][selected], atol=0, rtol=0
                    )
                torch.testing.assert_close(
                    state["down.weight"], raw_weights.T.float(), atol=0, rtol=0
                )
                torch.testing.assert_close(state["down.bias"], raw_bias.float(), atol=0, rtol=0)
                assert sum(v.numel() for v in state.values()) == row["parameters"]
                assert (
                    abs(
                        row["parameter_reduction"]
                        - (1 - row["parameters"] / case["teacher_resources"]["parameters"])
                    )
                    < 1e-15
                )
                outputs, jvps = [], []
                for probe in report_probes:
                    value, derivative = automatic(state, report, probe, False, 257)
                    outputs.append(value)
                    jvps.append(derivative)
                mse = (
                    outputs[0].double() - reference["value"].double()
                ).square().mean().item() / stats["scale_y"] ** 2
                jmse = (
                    torch.stack(jvps).double() - reference["derivative"].double()
                ).square().mean().item() / target_energy
                for key, actual in (("value_nmse", mse), ("derivative_relative_mse", jmse)):
                    difference = abs(row[key] - actual)
                    assert difference <= 2e-7 + 1e-5 * abs(row[key]), (
                        label,
                        row["method"],
                        row["budget"],
                        key,
                        difference,
                    )
                    scores.append(
                        {
                            "label": label,
                            "method": row["method"],
                            "budget": row["budget"],
                            "metric": key,
                            "absolute_difference": difference,
                        }
                    )
                del state, checkpoint, outputs, jvps, coefficient, gram, cross
            print(
                f"Verified {label}: {len(scores)} scores, {len(gains)} marginal checks", flush=True
            )
            del rebuilt, saved, stats, orders, traces, x, report, reference, report_probes
            gc.collect()
            torch.cuda.empty_cache()
    assert len(scores) == 864 and len(gains) == 432 and len(records) == 18
    write_json(
        ROOT / "audit.json",
        {
            "passed": True,
            "statistics_cases": records,
            "scores": scores,
            "greedy_marginals": gains,
            "checkpoints": 432,
            "score_count": len(scores),
            "max_score_difference": max(r["absolute_difference"] for r in scores),
            "max_gain_difference": max(r["gain_difference"] for r in gains),
            "neural_updates": 0,
        },
    )
    print("Independent audit passed", flush=True)


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
