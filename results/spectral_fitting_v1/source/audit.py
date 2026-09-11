"""Independent data/moment reconstruction and checkpoint scores for H103/H104."""

import gc
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from results.spectral_discovery_v1.source.study import tensor_sha
from results.spectral_fitting_v1.source.model import FeatureFFN
from src.core.reproducibility import sha256, write_json

ROOT = Path("results/spectral_fitting_v1")


def read(path):
    return json.loads(Path(path).read_text())


def gen(seed):
    return torch.Generator().manual_seed(seed)


def independent_targets(x, basis, rotation, task, train):
    z = x.double() @ basis
    if task == "quadratic":
        features = (z * z - 1) / math.sqrt(2)
    elif task == "cubic":
        features = z * (z * z - 3) / math.sqrt(6)
    elif task == "product":
        features = z * torch.cat((z[:, 1:], z[:, :1]), dim=1)
    else:
        features = (F.relu(z).square() - 0.5 - math.sqrt(2 / math.pi) * z) / math.sqrt(
            1.25 - 2 / math.pi
        )
    output = features @ rotation
    return (output / output[:train].std(0, correction=0)).float()


def independent_matrix(x, y, mode, seed=None):
    e = y.double().square().sum(1)
    e = e / e.mean()
    w = e if mode == "raw" else e / (1 + e)
    w = w - w.mean()
    if seed is not None:
        w = w[torch.randperm(len(w), generator=gen(10700 + seed))]
    w = w - w.mean()
    matrix = torch.zeros(384, 384, dtype=torch.float64, device="cuda")
    # A different summation partition from the experiment.
    for offset in range(0, len(x), 4096):
        batch = x[offset : offset + 4096].cuda().double()
        matrix += batch.T @ (batch * w[offset : offset + 4096, None].cuda())
    return ((matrix + matrix.T) / (2 * len(x))).cpu()


def audit_discovery():
    root = Path("results/spectral_discovery_v1")
    result = read(root / "result.json")
    for path, digest in read(root / "protocol.json")["sources"].items():
        assert sha256(Path(path)) == digest, path
    checked = []
    for seed in (17, 29, 43):
        x = torch.randn(65536, 384, generator=gen(10300 + seed))
        q = torch.linalg.qr(
            torch.randn(384, 384, dtype=torch.float64, generator=gen(10400 + seed))
        ).Q[:, :16]
        out = torch.linalg.qr(
            torch.randn(384, 384, dtype=torch.float64, generator=gen(10500 + seed))
        ).Q[:16]
        for task in ("quadratic", "cubic", "product", "piecewise"):
            y = independent_targets(x, q, out, task, 65536)
            for mode in ("raw", "bounded", "permuted"):
                row = next(
                    r
                    for r in result["records"]
                    if (r["seed"], r["task"], r["mode"]) == (seed, task, mode)
                )
                assert tensor_sha(x) == row["x_sha256"] and tensor_sha(y) == row["y_sha256"]
                saved = torch.load(
                    root / "cases" / f"{task}_{mode}_s{seed}" / "moments.pt", weights_only=True
                )
                actual = independent_matrix(x, y, mode, seed if mode == "permuted" else None)
                torch.testing.assert_close(actual, saved["matrix"], atol=1e-10, rtol=1e-10)
                b = saved["basis"]
                torch.testing.assert_close(
                    b.T @ b, torch.eye(32, dtype=torch.float64), atol=1e-10, rtol=1e-10
                )
                torch.testing.assert_close(
                    saved["matrix"] @ b, b * saved["eigenvalues"][-32:], atol=1e-10, rtol=1e-10
                )
                recovery = torch.linalg.svdvals(q.T @ b).square().mean().item()
                assert abs(recovery - row["recovery"]) < 1e-12
                checked.append(
                    {
                        "seed": seed,
                        "task": task,
                        "mode": mode,
                        "matrix_error": (actual - saved["matrix"]).abs().max().item(),
                        "recovery": recovery,
                    }
                )
            del y
        del x, q, out
        print(f"H103 reconstructed seed {seed}", flush=True)
    write_json(root / "audit.json", {"passed": True, "cases": checked})


def audit_fitting():
    result = read(ROOT / "result.json")
    assert result["runs"] == 408
    for path, digest in read(ROOT / "protocol.json")["sources"].items():
        assert sha256(Path(path)) == digest, path
    checked, alignments = [], []
    for seed in (17, 29, 43):
        data = torch.load(ROOT / "data" / f"s{seed}.pt", weights_only=True)
        x = torch.randn(73728, 384, generator=gen(10800 + seed))
        assert torch.equal(x, data["x"])
        expected_stream = torch.randint(65536, (300, 256), generator=gen(12000 + seed))
        assert torch.equal(expected_stream, data["stream"])
        q = torch.linalg.qr(
            torch.randn(384, 384, dtype=torch.float64, generator=gen(10900 + seed))
        ).Q[:, :16]
        rotation = torch.linalg.qr(
            torch.randn(384, 384, dtype=torch.float64, generator=gen(11000 + seed))
        ).Q[:16]
        assert torch.equal(q, data["teacher"]) and torch.equal(rotation, data["rotation"])
        for task in ("quadratic", "cubic", "product", "piecewise"):
            y = independent_targets(x, q, rotation, task, 65536)
            assert torch.equal(y, data["targets"][task])
            bases = torch.load(
                ROOT / "initializers" / f"{task}_s{seed}" / "bases.pt", weights_only=True
            )
            meta = read(ROOT / "initializers" / f"{task}_s{seed}" / "metrics.json")
            for mode in ("raw", "bounded"):
                assert tensor_sha(bases[mode]) == meta[mode]["basis_sha256"]
                matrix = independent_matrix(x[:65536], y[:65536], mode)
                eigenvalues, vectors = torch.linalg.eigh(matrix)
                inferred = vectors[:, -32:]
                torch.testing.assert_close(
                    inferred @ inferred.T,
                    bases[mode].double() @ bases[mode].double().T,
                    atol=1e-7,
                    rtol=1e-6,
                )
            task_rows = [r for r in result["rows"] if r["seed"] == seed and r["task"] == task]
            for row in task_rows:
                model = FeatureFFN(row["form"], seed, bases)
                assert model.counts()["parameters"] == row["parameters"]
                assert {n: tensor_sha(p) for n, p in model.state_dict().items()} == row[
                    "initial_hashes"
                ]
                assert row["stream_sha256"] == tensor_sha(expected_stream)
                path = Path(row["checkpoint"])
                assert sha256(path) == row["checkpoint_sha256"]
                payload = torch.load(path, weights_only=True)
                assert {n: tensor_sha(p) for n, p in payload["model"].items()} == row[
                    "state_hashes"
                ]
                assert payload["step"] == 300 and row["training_presentations"] == 76800
                model.load_state_dict(payload["model"], strict=True)
                if model.compress is not None:
                    initial_basis = bases[row["form"].split("_")[-1]].double()
                    final_basis = torch.linalg.qr(model.compress.weight.detach().double().T).Q
                    alignments.append(
                        {
                            "label": row["label"],
                            "selected": row["selected"],
                            "initial_recovery": torch.linalg.svdvals(q.T @ initial_basis)
                            .square()
                            .mean()
                            .item(),
                            "final_recovery": torch.linalg.svdvals(q.T @ final_basis)
                            .square()
                            .mean()
                            .item(),
                        }
                    )
                model.cuda().eval()
                scores = []
                with torch.inference_mode():
                    for left, right in ((65536, 69632), (69632, 73728)):
                        total = 0.0
                        for offset in range(left, right, 1024):
                            predicted = model(x[offset : offset + 1024].cuda())
                            total += F.mse_loss(
                                predicted, y[offset : offset + 1024].cuda(), reduction="sum"
                            ).item()
                        scores.append(total / ((right - left) * 384))
                assert scores == [row["selection_mse"], row["reporting_mse"]], (
                    row["label"],
                    scores,
                )
                history_path = path.parent / "history.jsonl"
                assert sha256(history_path) == row["history_sha256"]
                history = [json.loads(line) for line in history_path.read_text().splitlines()]
                assert len(history) == 300 and all(
                    math.isfinite(r["loss"]) and math.isfinite(r["preclip_norm"]) for r in history
                )
                checked.append(
                    {
                        "label": row["label"],
                        "scores": scores,
                        "initialization_exact": True,
                        "data_exact": True,
                    }
                )
                del model, payload, predicted
            for form in {r["form"] for r in task_rows}:
                peers = [r for r in task_rows if r["form"] == form]
                selected = min(peers, key=lambda r: (r["selection_mse"], r["rate"]))
                assert [r["label"] for r in peers if r["selected"]] == [selected["label"]]
            # Factor initializer modes must differ only in A.
            for activation in ("gelu", "cubic"):
                for rate in (0.001, 0.003):
                    peers = [
                        r
                        for r in task_rows
                        if r["form"].startswith(f"factor_{activation}") and r["rate"] == rate
                    ]
                    for name in ("up.weight", "up.bias", "down.weight", "down.bias"):
                        assert len({r["initial_hashes"][name] for r in peers}) == 1
            print(f"H104 audited {task} seed {seed}: {len(task_rows)} checkpoints", flush=True)
        del data, x, y, bases
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        ROOT / "audit.json",
        {
            "passed": True,
            "checkpoint_count": len(checked),
            "score_count": 2 * len(checked),
            "checks": checked,
            "factor_alignment": alignments,
            "rate_selections": 204,
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        audit_discovery()
        audit_fitting()
    except Exception:
        import traceback

        write_json(ROOT / "audit_failure.json", {"traceback": traceback.format_exc()})
        raise
