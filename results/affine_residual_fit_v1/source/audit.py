"""Post-run independent data recapture, tensor algebra and endpoint audit."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.core.config import ModelConfig
from src.core.ffn_capture import capture_ffn_pairs
from src.core.reproducibility import sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/affine_residual_fit_v1")
TRAIN, SELECT = 32768, 36864


def read(path):
    return json.loads(Path(path).read_text())


def forward(state, form, x):
    """No experimental Student/fold/score helper is used in this audit."""
    up = x @ state["up.weight"].T + state["up.bias"]
    kind = form.removeprefix("affine_")
    if kind == "swiglu":
        hidden = F.silu(up) * (x @ state["gate.weight"].T + state["gate.bias"])
    elif kind == "gelu":
        hidden = F.gelu(up)
    elif kind == "relu":
        hidden = up.clamp_min(0)
    elif kind == "leaky_relu":
        hidden = torch.where(up >= 0, up, 0.01 * up)
    elif kind == "prelu":
        hidden = torch.where(up >= 0, up, state["alpha"] * up)
    elif kind == "silu":
        hidden = F.silu(up)
    else:
        raise ValueError(form)
    output = hidden @ state["down.weight"].T
    if form.startswith("affine_"):
        output = output + x @ state["affine.weight"].T + state["affine.bias"]
    else:
        output = output + state["down.bias"]
    return output


@torch.no_grad()
def outputs(state, form, x):
    cuda_state = {k: v.cuda() for k, v in state.items()}
    output = torch.cat([forward(cuda_state, form, part.cuda()).cpu() for part in x.split(769)])
    assert torch.isfinite(output).all()
    return output


def run():
    result, protocol = read(ROOT / "result.json"), read(ROOT / "protocol.json")
    assert result["status"] == "COMPLETE" and len(result["rows"]) == 144
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    for path, digest in protocol["prerequisites"].items():
        assert sha256(Path(path)) == digest, path
    prior = read("results/real_subspace_v1/result.json")
    excluded = {v for c in prior["collections"] for v in c["window_ids"]}
    tokens = torch.from_numpy(np.load("data/wikitext2_v1/train.npy").astype("int64"))
    windows = tokens[: len(tokens) // 128 * 128].reshape(-1, 128)
    available = torch.tensor([i for i in range(len(windows)) if i not in excluded])
    chosen = available[
        torch.randperm(len(available), generator=torch.Generator().manual_seed(17001))[:960]
    ]
    datasets, endpoints, folds, initializations = [], [], [], []
    total_updates = 0
    for kind, info in protocol["teachers"].items():
        assert sha256(Path(info["path"])) == info["sha256"]
        checkpoint = torch.load(info["path"], map_location="cpu", weights_only=True)
        teacher = Transformer(ModelConfig(**checkpoint["model_config"]))
        teacher.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        for index, seed in enumerate(protocol["seeds"]):
            ids = chosen[index * 320 : (index + 1) * 320]
            collection = next(
                c for c in result["collections"] if (c["teacher"], c["seed"]) == (kind, seed)
            )
            assert collection["window_ids"] == ids.tolist()
            teacher.cuda().eval()
            # Independent outer run through all intervening decoder blocks.
            recaptured = capture_ffn_pairs(teacher, windows[ids], (0, 3, 7))
            teacher.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            for layer in (0, 3, 7):
                label = f"{kind}_l{layer}_s{seed}"
                folder = ROOT / "pairs" / label
                diag = next(
                    d
                    for d in result["diagnostics"]
                    if (d["teacher"], d["layer"], d["seed"]) == (kind, layer, seed)
                )
                assert sha256(folder / "data.pt") == diag["data_sha256"]
                data = torch.load(folder / "data.pt", map_location="cpu", weights_only=True)
                for name in ("x", "y"):
                    torch.testing.assert_close(data[name], recaptured[layer][name], atol=0, rtol=0)
                del recaptured[layer]
                torch.testing.assert_close(data["window_ids"], ids, atol=0, rtol=0)
                xx, yy = data["x"][:TRAIN].double(), data["y"][:TRAIN].double()
                norm = {
                    "mx": xx.mean(0),
                    "sx": xx.std(0, correction=0).clamp_min(1e-3),
                    "my": yy.mean(0),
                    "sy": (yy - yy.mean(0)).square().mean().sqrt().clamp_min(1e-6),
                }
                for k, v in norm.items():
                    torch.testing.assert_close(v, data["normalization"][k], atol=0, rtol=0)
                x = ((data["x"].double() - norm["mx"]) / norm["sx"]).float()
                y = ((data["y"].double() - norm["my"]) / norm["sy"]).float()
                expected_stream = torch.randint(
                    TRAIN, (600, 256), generator=torch.Generator().manual_seed(19000 + seed)
                )
                torch.testing.assert_close(data["stream"], expected_stream, atol=0, rtol=0)
                datasets.append(
                    {
                        "label": label,
                        "exact_recapture": True,
                        "normalization": True,
                        "fresh_split": True,
                    }
                )
                bank = teacher.blocks[layer].ffn
                permutation = torch.randperm(
                    bank.up.weight.shape[0], generator=torch.Generator().manual_seed(18000 + seed)
                )
                for row in [
                    r
                    for r in result["rows"]
                    if (r["teacher"], r["layer"], r["seed"]) == (kind, layer, seed)
                ]:
                    form = row["form"]
                    selected = min(
                        range(3), key=lambda i: (row["endpoints"][i]["selection_mse"], i)
                    )
                    assert selected == row["selected_index"]
                    assert row["reporting_mse"] == row["endpoints"][selected]["reporting_mse"]
                    assert row["teacher_rows"] == permutation[: len(row["teacher_rows"])].tolist()
                    state_outputs = []
                    for index_endpoint, endpoint in enumerate(row["endpoints"]):
                        path = Path(endpoint["checkpoint"]["path"])
                        assert sha256(path) == endpoint["checkpoint"]["sha256"]
                        state = torch.load(path, map_location="cpu", weights_only=True)
                        count = sum(v.numel() for v in state.values())
                        assert count == row["parameters"]
                        assert (
                            abs(
                                row["parameter_reduction"]
                                - (1 - count / diag["teacher_parameters"])
                            )
                            < 1e-15
                        )
                        output = outputs(state, form, x[TRAIN:])
                        state_outputs.append(output)
                        for split, start, stop in (
                            ("selection", 0, 4096),
                            ("reporting", 4096, 8192),
                        ):
                            mse = (
                                (
                                    output[start:stop].double()
                                    - y[TRAIN + start : TRAIN + stop].double()
                                )
                                .square()
                                .mean()
                                .item()
                            )
                            saved = endpoint[f"{split}_mse"]
                            assert abs(saved - mse) <= 2e-7 + 1e-5 * abs(saved), (
                                label,
                                form,
                                index_endpoint,
                                split,
                                saved,
                                mse,
                            )
                            endpoints.append(
                                {
                                    "label": label,
                                    "form": form,
                                    "endpoint": index_endpoint,
                                    "split": split,
                                    "absolute_difference": abs(saved - mse),
                                }
                            )
                        if index_endpoint == 0:
                            picked = torch.tensor(row["teacher_rows"])
                            w = bank.up.weight[picked].double()
                            torch.testing.assert_close(
                                state["up.weight"], (w * norm["sx"]).float(), atol=0, rtol=0
                            )
                            torch.testing.assert_close(
                                state["up.bias"], (w @ norm["mx"]).float(), atol=0, rtol=0
                            )
                            if "swiglu" in form:
                                if bank.gate is None:
                                    assert torch.count_nonzero(state["gate.weight"]) == 0
                                    assert bool((state["gate.bias"] == 1).all())
                                else:
                                    w = bank.gate.weight[picked].double()
                                    torch.testing.assert_close(
                                        state["gate.weight"],
                                        (w * norm["sx"]).float(),
                                        atol=0,
                                        rtol=0,
                                    )
                                    torch.testing.assert_close(
                                        state["gate.bias"], (w @ norm["mx"]).float(), atol=0, rtol=0
                                    )
                            # CPU FP64 normal-equation residual after rounding the readout to FP32.
                            with torch.no_grad():
                                z = (
                                    x[:TRAIN].double() @ state["up.weight"].double().T
                                    + state["up.bias"].double()
                                )
                                if "swiglu" in form:
                                    h = F.silu(z) * (
                                        x[:TRAIN].double() @ state["gate.weight"].double().T
                                        + state["gate.bias"].double()
                                    )
                                else:
                                    h = {
                                        "gelu": lambda z: F.gelu(z),
                                        "relu": lambda z: z.clamp_min(0),
                                        "leaky_relu": lambda z: torch.where(z >= 0, z, 0.01 * z),
                                        "prelu": lambda z: torch.where(z >= 0, z, 0.25 * z),
                                        "silu": lambda z: F.silu(z),
                                    }[form.removeprefix("affine_")](z)
                                design = (
                                    torch.cat((x[:TRAIN].double(), h), 1)
                                    if form.startswith("affine_")
                                    else h
                                )
                                mean, sd = (
                                    design.mean(0),
                                    design.std(0, correction=0).clamp_min(1e-6),
                                )
                                standardized = (design - mean) / sd
                                coefficient = (
                                    torch.cat((state["affine.weight"].T, state["down.weight"].T))
                                    if form.startswith("affine_")
                                    else state["down.weight"].T
                                )
                                fitted = coefficient.double() * sd[:, None]
                                cross = (
                                    standardized.T
                                    @ (y[:TRAIN].double() - y[:TRAIN].double().mean(0))
                                    / TRAIN
                                )
                                residual = (
                                    standardized.T @ (standardized @ fitted) / TRAIN
                                    + 1e-4 * fitted
                                    - cross
                                )
                                maximum = residual.abs().max().item()
                                assert maximum < 1e-5, (label, form, maximum)
                                initializations.append(
                                    {
                                        "label": label,
                                        "form": form,
                                        "prefix": True,
                                        "ridge_stationarity_max": maximum,
                                    }
                                )
                                del z, h, design, standardized, coefficient, fitted, cross, residual
                        else:
                            history = read(path.with_suffix(".json"))
                            assert len(history["history"]) == 600 and history["all_finite"]
                            assert [h["step"] for h in history["history"]] == list(range(1, 601))
                            assert all(
                                np.isfinite(h[k])
                                for h in history["history"]
                                for k in (
                                    "loss",
                                    "preclip_norm",
                                    "update_ms",
                                    "forward_ms",
                                    "backward_ms",
                                )
                            )
                            total_updates += 600
                        del state
                    assert (
                        sha256(Path(row["raw_checkpoint"]["path"]))
                        == row["raw_checkpoint"]["sha256"]
                    )
                    raw = torch.load(
                        row["raw_checkpoint"]["path"], map_location="cpu", weights_only=True
                    )
                    raw_output = outputs(raw, form, data["x"][TRAIN:])
                    expected = state_outputs[selected].double() * norm["sy"] + norm["my"]
                    relative = (raw_output.double() - expected).square().mean().item() / norm[
                        "sy"
                    ].square().item()
                    assert relative < 1e-9, (label, form, relative)
                    folds.append({"label": label, "form": form, "normalized_mse": relative})
                    del raw, raw_output, state_outputs, expected
                    gc.collect()
                    torch.cuda.empty_cache()
                print(f"Verified {label}: {len(endpoints)} scores", flush=True)
                del data, x, y, xx, yy
        del teacher
    assert total_updates == result["neural_updates"] == 172800
    assert len(endpoints) == 864 and len(folds) == len(initializations) == 144
    write_json(
        ROOT / "audit.json",
        {
            "passed": True,
            "sources_verified": len(protocol["sources"]),
            "datasets": datasets,
            "scores": endpoints,
            "exports": folds,
            "initializations": initializations,
            "selected": len(folds),
            "neural_updates_checked": total_updates,
            "max_score_difference": max(e["absolute_difference"] for e in endpoints),
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
