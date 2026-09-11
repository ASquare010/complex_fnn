"""Independent raw-state/autograd endpoint audit, with exact fresh-input recapture."""

import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from results.sobolev_selection_v1.source.audit import automatic, forward, probes
from src.core.config import ModelConfig
from src.core.ffn_capture import capture_ffn_pairs
from src.core.reproducibility import sha256, write_json
from src.core.transformer import Transformer

ROOT = Path("results/sobolev_learning_v1")


def read(path):
    return json.loads(Path(path).read_text())


@torch.no_grad()
def values(state, x, batch):
    cuda = {k: v.cuda() for k, v in state.items()}
    return torch.cat([forward(cuda, part.cuda()).cpu() for part in x.split(batch)])


def run():
    result, protocol, manifest = [
        read(ROOT / name) for name in ("result.json", "protocol.json", "data_manifest.json")
    ]
    assert result["status"] == "COMPLETE" and result["neural_updates"] == 86400
    for path, digest in protocol["sources"].items():
        assert sha256(Path(path)) == digest, path
    assert (
        sha256(Path("results/sobolev_selection_v1/result.json")) == protocol["prior_result_sha256"]
    )
    older = [
        read(f"results/{name}/result.json")
        for name in ("real_subspace_v1", "affine_residual_fit_v1")
    ]
    excluded = {i for source in older for c in source["collections"] for i in c["window_ids"]}
    assert sorted(excluded) == manifest["excluded_window_ids"]
    tokens = torch.from_numpy(np.load("data/wikitext2_v1/train.npy").astype("int64"))
    windows = tokens[: len(tokens) // 128 * 128].reshape(-1, 128)
    allowed = torch.tensor([i for i in range(len(windows)) if i not in excluded])
    chosen = allowed[
        torch.randperm(len(allowed), generator=torch.Generator().manual_seed(18001))[:960]
    ]
    assert chosen.tolist() == manifest["window_ids"] and len(set(chosen.tolist())) == 960
    scores, datasets, training = [], [], []
    for kind, info in protocol["teachers"].items():
        assert sha256(Path(info["path"])) == info["sha256"]
        ckpt = torch.load(info["path"], map_location="cpu", weights_only=True)
        decoder = Transformer(ModelConfig(**ckpt["model_config"]))
        decoder.load_state_dict(ckpt["model"])
        del ckpt
        for seed_index, seed in enumerate((71, 83, 97)):
            ids = chosen[seed_index * 320 : (seed_index + 1) * 320]
            collection = next(
                c for c in result["collections"] if (c["teacher"], c["seed"]) == (kind, seed)
            )
            assert collection["window_ids"] == ids.tolist()
            decoder.cuda().eval()
            captured = capture_ffn_pairs(decoder, windows[ids], (0, 3, 7))
            decoder.cpu()
            for layer in (0, 3, 7):
                label = f"{kind}_l{layer}_s{seed}"
                folder = ROOT / "pairs" / label
                case = next(
                    c
                    for c in result["cases"]
                    if (c["teacher"], c["layer"], c["seed"]) == (kind, layer, seed)
                )
                assert sha256(folder / "data.pt") == case["data_sha256"]
                data = torch.load(folder / "data.pt", map_location="cpu", weights_only=True)
                torch.testing.assert_close(captured.pop(layer)["x"], data["x"], atol=0, rtol=0)
                torch.testing.assert_close(ids, data["window_ids"], atol=0, rtol=0)
                expected_stream = torch.randint(
                    32768, (600, 256), generator=torch.Generator().manual_seed(31000 + seed)
                )
                torch.testing.assert_close(data["stream"], expected_stream, atol=0, rtol=0)
                oldstats = torch.load(
                    Path("results/sobolev_selection_v1/cases") / label / "statistics.pt",
                    map_location="cpu",
                    weights_only=True,
                )["statistics"]
                assert data["sy"] == oldstats["scale_y"] == case["sy"]
                torch.testing.assert_close(data["sx"], oldstats["sx"], atol=0, rtol=0)
                teacher = decoder.blocks[layer].ffn.state_dict()
                actual_y = values(teacher, data["x"], 512)
                torch.testing.assert_close(actual_y, data["y"], atol=0, rtol=0)
                report_x = data["x"][36864:]
                directions = [
                    probes(report_x, data["sx"], offset + seed) for offset in (32000, 33000)
                ]
                derivative_checks = []
                for i, direction in enumerate(directions):
                    _, target = automatic(teacher, report_x, direction, False, 512)
                    relative = (
                        target.double() - data["target_j"][i].double()
                    ).square().mean().item() / data["target_j"][i].double().square().mean().item()
                    assert relative < 1e-11, (label, relative)
                    derivative_checks.append(relative)
                datasets.append(
                    {
                        "label": label,
                        "exact_input_recapture": True,
                        "exact_smooth_targets": True,
                        "exact_stream_and_old_normalizers": True,
                        "autograd_target_relative_mse": derivative_checks,
                    }
                )
                for row in [
                    r
                    for r in result["rows"]
                    if (r["teacher"], r["layer"], r["seed"]) == (kind, layer, seed)
                ]:
                    source = row["initialization"]
                    assert sha256(Path(source["path"])) == source["sha256"]
                    initial = torch.load(source["path"], map_location="cpu", weights_only=True)[
                        "state_dict"
                    ]
                    before_states = []
                    for endpoint in row["endpoints"]:
                        path = Path(endpoint["checkpoint"]["path"])
                        assert sha256(path) == endpoint["checkpoint"]["sha256"]
                        state = torch.load(path, map_location="cpu", weights_only=True)[
                            "state_dict"
                        ]
                        assert all(bool(torch.isfinite(v).all()) for v in state.values())
                        assert sum(v.numel() for v in state.values()) == row["parameters"]
                        if endpoint["steps"] == 0:
                            for name, value in state.items():
                                torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
                            before_states.append(True)
                        else:
                            record = read(folder / row["method"] / f"rate{endpoint['rate']}.json")
                            history = record["history"]
                            assert record["all_finite"] and len(history) == 600
                            assert [v["step"] for v in history] == list(range(1, 601))
                            assert all(
                                math.isfinite(v[k]) and v[k] >= 0
                                for v in history
                                for k in (
                                    "loss",
                                    "preclip_norm",
                                    "update_ms",
                                    "forward_ms",
                                    "backward_ms",
                                )
                            )
                            assert [d["step"] for d in record["diagnostics"]] == [0, 50, 150, 600]
                            assert all(
                                a["finite"]
                                for d in record["diagnostics"]
                                for a in d["activations"].values()
                            )
                            first_x = data["x"][data["stream"][0]]
                            first_y = data["y"][data["stream"][0]]
                            predicted = values(initial, first_x, 256)
                            first_loss = (
                                (predicted / data["sy"] - first_y / data["sy"])
                                .square()
                                .mean()
                                .item()
                            )
                            assert abs(first_loss - history[0]["loss"]) <= 1e-7 + 1e-6 * first_loss
                            training.append(
                                {
                                    "label": label,
                                    "method": row["method"],
                                    "rate": endpoint["rate"],
                                    "finite_steps": 600,
                                    "initial_loss_matches": True,
                                }
                            )
                        y = values(state, data["x"][32768:], 257).double()
                        truth = data["y"][32768:].double()
                        mse = (y - truth).square().mean(1) / data["sy"] ** 2
                        jvps = [automatic(state, report_x, v, False, 257)[1] for v in directions]
                        jerror = (
                            torch.stack(jvps).double() - data["target_j"].double()
                        ).square().mean().item() / data["target_j"].double().square().mean().item()
                        for metric, actual in (
                            ("selection_mse", mse[:4096].mean().item()),
                            ("reporting_mse", mse[4096:].mean().item()),
                            ("derivative_relative_mse", jerror),
                        ):
                            difference = abs(actual - endpoint[metric])
                            assert difference <= 2e-7 + 1e-5 * abs(endpoint[metric]), (
                                label,
                                row["method"],
                                endpoint["rate"],
                                metric,
                                difference,
                            )
                            scores.append(
                                {
                                    "label": label,
                                    "method": row["method"],
                                    "rate": endpoint["rate"],
                                    "metric": metric,
                                    "absolute_difference": difference,
                                }
                            )
                    assert before_states == [True]
                    selected = min(
                        range(3), key=lambda i: (row["endpoints"][i]["selection_mse"], i)
                    )
                    assert selected == row["selected_index"]
                    for metric in ("selection_mse", "reporting_mse", "derivative_relative_mse"):
                        assert row[metric] == row["endpoints"][selected][metric]
                print(
                    f"Verified {label}: {len(scores)} metrics, {len(training)} training records",
                    flush=True,
                )
                del data, actual_y, target, directions, oldstats
                gc.collect()
                torch.cuda.empty_cache()
            del captured
        del decoder
    assert len(scores) == 648 and len(training) == 144 and len(datasets) == 18
    write_json(
        ROOT / "audit.json",
        {
            "passed": True,
            "datasets": datasets,
            "scores": scores,
            "training": training,
            "endpoint_states": 216,
            "selected": 72,
            "max_score_difference": max(s["absolute_difference"] for s in scores),
            "audit_neural_updates": 0,
        },
    )
    print("Independent H109 audit passed", flush=True)


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
