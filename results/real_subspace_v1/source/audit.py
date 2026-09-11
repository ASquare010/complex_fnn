"""Independent teacher recapture, chunked statistics, and explicit factor evaluation."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from results.spectral_discovery_v1.source.study import tensor_sha
from src.core.config import ModelConfig
from src.core.reproducibility import sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/real_subspace_v1")


def read(path):
    return json.loads(Path(path).read_text())


def energy(y):
    norm = (y * y).sum(-1)
    norm /= norm.mean()
    w = norm / (1 + norm)
    return w - w.mean()


def outer(x, y, weights=None):
    value = torch.zeros(x.shape[1], y.shape[1], dtype=torch.float64)
    for start in range(0, len(x), 4096):
        xx, yy = x[start : start + 4096], y[start : start + 4096]
        if weights is not None:
            yy = yy * weights[start : start + 4096, None]
        value += xx.T @ yy
    return value / len(x)


def teacher_call(x, state):
    up = F.linear(x, state["up.weight"])
    if "gate.weight" in state:
        up = F.silu(up) * F.linear(x, state["gate.weight"])
    else:
        up = F.gelu(up)
    return F.linear(up, state["down.weight"])


@torch.no_grad()
def run():
    protocol, result = read(ROOT / "protocol.json"), read(ROOT / "result.json")
    assert result["status"] == "COMPLETE" and len(result["rows"]) == 378
    for name, digest in protocol["sources"].items():
        assert sha256(Path(name)) == digest, name
    tokens = torch.from_numpy(np.load("data/wikitext2_v1/train.npy").astype("int64"))
    windows = tokens[: len(tokens) // 128 * 128].reshape(-1, 128)
    records, recaptures = [], []
    for activation, entry in protocol["teachers"].items():
        path = Path(entry["path"])
        assert sha256(path) == entry["sha256"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["step"] == 3200
        model = Transformer(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        for seed in protocol["seeds"]:
            collection = next(
                r for r in result["collections"] if (r["teacher"], r["seed"]) == (activation, seed)
            )
            ids = torch.randperm(
                len(windows), generator=torch.Generator().manual_seed(16000 + seed)
            )[:320]
            assert ids.tolist() == collection["window_ids"]
            assert set(ids[:256].tolist()).isdisjoint(ids[256:].tolist())
            chosen = windows[ids]
            assert tensor_sha(chosen) == collection["window_sha256"]
            # Re-run the entire decoder, including vocabulary projection, without
            # calling the capture helper. Check every captured row, not just probes.
            model.cuda().eval()
            captured = {layer: {"x": [], "y": []} for layer in protocol["layers"]}
            handles = []

            def hook_for(layer):
                def hook(module, args, out):
                    captured[layer]["x"].append(args[0].flatten(0, 1).float().cpu())
                    captured[layer]["y"].append(out.flatten(0, 1).float().cpu())

                return hook

            for layer in captured:
                handles.append(model.blocks[layer].ffn.register_forward_hook(hook_for(layer)))
            for batch in chosen.split(8):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(batch.cuda())
            for h in handles:
                h.remove()
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            for layer in protocol["layers"]:
                folder = ROOT / "pairs" / f"{activation}_l{layer}_s{seed}"
                pair = torch.load(folder / "data.pt", weights_only=True)
                stats = torch.load(folder / "statistics.pt", weights_only=True)
                metadata = read(folder / "metrics.json")
                assert sha256(folder / "data.pt") == metadata["data_sha256"]
                assert sha256(folder / "statistics.pt") == metadata["statistics_sha256"]
                for key in ("x", "y"):
                    assert torch.equal(torch.cat(captured[layer][key]), pair[key])
                    assert tensor_sha(pair[key][:32768]) == metadata["training_pair_sha256"][key]
                recaptures.append(
                    {"teacher": activation, "layer": layer, "seed": seed, "all_rows_exact": True}
                )
                x, y = pair["x"][:32768].double(), pair["y"][:32768].double()
                torch.testing.assert_close(x.mean(0), stats["mx"], atol=0, rtol=0)
                torch.testing.assert_close(y.mean(0), stats["my"], atol=0, rtol=0)
                x, y = x - stats["mx"], y - stats["my"]
                cov, cross = outer(x, x), outer(x, y)
                torch.testing.assert_close(cov, stats["covariance"], atol=1e-10, rtol=1e-10)
                ridge = stats["ridge"]
                torch.testing.assert_close(
                    (cov + ridge * torch.eye(384, dtype=torch.float64)) @ stats["affine"],
                    cross,
                    atol=1e-9,
                    rtol=1e-9,
                )
                w = energy(y)
                wr = energy(y - x @ stats["affine"])
                wp = w[
                    torch.randperm(len(w), generator=torch.Generator().manual_seed(15000 + seed))
                ]
                weights = {
                    k: v.detach().double() for k, v in model.blocks[layer].ffn.state_dict().items()
                }
                l = weights["up.weight"]
                if "gate.weight" in weights:
                    l = torch.cat((l, weights["gate.weight"]))
                h, t = stats["h"], stats["t"]
                torch.testing.assert_close(
                    h @ t, torch.eye(384, dtype=torch.float64), atol=1e-8, rtol=1e-8
                )
                matrices = {
                    "pca": cov,
                    "weight_input": l.T @ l,
                    "linear_aware": h @ (l.T @ l) @ h,
                    "energy": outer(x, x, w),
                    "white_energy": t @ outer(x, x, w) @ t,
                    "white_residual": t @ outer(x, x, wr) @ t,
                    "white_permuted": t @ outer(x, x, wp) @ t,
                }
                for method, m in matrices.items():
                    b = stats["bases"][method]
                    torch.testing.assert_close(m, stats["matrices"][method], atol=1e-7, rtol=1e-7)
                    torch.testing.assert_close(
                        m @ b, b * stats["spectra"][method], atol=1e-7, rtol=1e-7
                    )
                    torch.testing.assert_close(
                        b.T @ b, torch.eye(384, dtype=torch.float64), atol=1e-9, rtol=1e-9
                    )
                cfull = stats["output_basis"]
                torch.testing.assert_close(
                    outer(y, y) @ cfull, cfull * stats["output_spectrum"], atol=1e-9, rtol=1e-9
                )
                gpu_weights = {k: v.float().cuda() for k, v in weights.items()}
                inputs = pair["x"][32768:]
                reference = torch.cat(
                    [teacher_call(b.cuda(), gpu_weights).cpu() for b in inputs.split(512)]
                )
                del gpu_weights
                variance = (reference.double() - stats["my"]).square().mean().item()
                assert abs(variance - metadata["variance"]) < 1e-12
                rows = [
                    r
                    for r in result["rows"]
                    if (r["teacher"], r["layer"], r["seed"]) == (activation, layer, seed)
                ]
                for row in rows:
                    rank, method = row["rank"], row["method"]
                    b, c = stats["bases"][method][:, -rank:], cfull[:, -rank:]
                    if method in ("pca", "weight_input", "energy"):
                        encoder, decoder = b, b.T
                    else:
                        encoder, decoder = t @ b, b.T @ h
                    offset = stats["mx"] - decoder.T @ (encoder.T @ stats["mx"])
                    a = encoder.T.float().cuda()
                    up = (weights["up.weight"] @ decoder.T).float().cuda()
                    bias = (weights["up.weight"] @ offset).float().cuda()
                    down = (c.T @ weights["down.weight"]).float().cuda()
                    final = c.float().cuda()
                    final_bias = (stats["my"] - c @ (c.T @ stats["my"])).float().cuda()
                    if "gate.weight" in weights:
                        gate = (weights["gate.weight"] @ decoder.T).float().cuda()
                        gbias = (weights["gate.weight"] @ offset).float().cuda()
                    predictions = []
                    for xx in inputs.split(1024):
                        latent = F.linear(xx.cuda(), a)
                        z = F.linear(latent, up, bias)
                        z = (
                            F.silu(z) * F.linear(latent, gate, gbias)
                            if "gate.weight" in weights
                            else F.gelu(z)
                        )
                        predictions.append(F.linear(F.linear(z, down), final, final_bias).cpu())
                    value = (
                        (torch.cat(predictions).double() - reference.double())
                        .square()
                        .mean()
                        .item()
                    )
                    assert abs(value - row["mse"]) <= 1e-7 + 1e-5 * row["mse"], (row, value)
                    assert (
                        abs(value / variance - row["normalized_mse"])
                        <= 1e-7 + 1e-5 * row["normalized_mse"]
                    )
                    count = (
                        2 * 384 * rank
                        + (3 if activation == "swiglu" else 2)
                        * l.shape[0]
                        // (2 if activation == "swiglu" else 1)
                        * rank
                        + l.shape[0]
                        + 384
                    )
                    assert count == row["parameters"]
                    assert row["pipeline_peak_cuda_bytes"] == max(
                        row["peak_allocated_bytes"], collection["peak_cuda_bytes"]
                    )
                    records.append(
                        {
                            "teacher": activation,
                            "layer": layer,
                            "seed": seed,
                            "rank": rank,
                            "method": method,
                            "independent_mse": value,
                            "absolute_error": abs(value - row["mse"]),
                            "passed": True,
                        }
                    )
                print(f"Audited {folder}: {len(records)} scores", flush=True)
            del captured
        del model
    write_json(
        ROOT / "audit.json",
        {
            "passed": True,
            "score_count": len(records),
            "recaptured_pairs": recaptures,
            "rows": records,
            "score_tolerance": "absolute 1e-7 plus relative 1e-5; independent batch partition and factor algebra",
            "all_frozen_sources_match": True,
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        run()
    except Exception:
        import traceback

        write_json(ROOT / "audit_failure.json", {"traceback": traceback.format_exc()})
        raise
