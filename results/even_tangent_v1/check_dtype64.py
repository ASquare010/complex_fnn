"""Freeze and run the local even-feature diagnostic, including finite-step replay."""

# ruff: noqa: I001
from results.checkpoint_input_offload_v1.source.common import torch
from results.checkpoint_input_offload_v1.source.prepare import hashes, read, sha
from results.ordinary_long_training_v1.io import write_json
from results.readout_learning_v1.model import Model
from results.reversible_softsign_v1.model import inverse
from results.even_tangent_v1.model import features, evaluate
from pathlib import Path
import math

ROOT = Path("results/even_tangent_v1")
assert not (ROOT / "protocol_dtype64.json").exists()
hashes(read("results/readout_learning_v1/receipt.json")["files"])
old = read("results/readout_learning_v1/protocol.json")
for name, path in [("CURRENT_STATE", "research/CURRENT_STATE.md"), ("README", "README.md")]:
    (ROOT / (name + ".before.md")).write_bytes(Path(path).read_bytes())
p = dict(
    study="H153",
    recovery=dict(
        original_protocol_sha256=sha(ROOT / "protocol.json"),
        original_exit=1,
        change="Only scalar fixture coefficients become explicit FP64; equations and tolerances unchanged",
    ),
    seeds=[307, 317, 331],
    probes=[0, 11, 23],
    rates=[1e-6, 0.001, 1.0],
    fractions=[0.01, 0.1, 1.0],
    maintained_files=old["maintained_files"],
    prior_receipt=sha("results/readout_learning_v1/receipt.json"),
    sources={
        **old["sources"],
        **{
            f.as_posix(): sha(f)
            for f in [*ROOT.glob("*.py"), Path("research/even_tangent_plan.md")]
        },
    },
)
write_json(ROOT / "protocol_dtype64.json", p)
z = torch.cat(
    (
        -torch.logspace(-12, 12, 481, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.logspace(-12, 12, 481, dtype=torch.float64),
    )
)
theta, eta = torch.tensor(0.8, dtype=torch.float64), torch.tensor(-1.3, dtype=torch.float64)
coeff = 0.5 * torch.tanh(theta + torch.where(z >= 0, eta, -eta))
y = z + coeff * z / (1 + z.abs())
back = inverse(y, 0.5 * torch.tanh(theta + torch.where(y >= 0, eta, -eta)))
assert ((back - z).abs() / (1 + z.abs())).max().item() <= 1e-12
cases = []
for seed in p["seeds"]:
    s = {k: v.double() for k, v in Model("learned", seed).state_dict().items()}
    x = torch.randn(
        2560, 32, generator=torch.Generator().manual_seed(123000 + seed), dtype=torch.float64
    )
    for probe in p["probes"]:
        fp, fm = features(s, x, probe), features(s, -x, probe)
        parity = {
            name: (
                (fp[name] + fm[name] if name == "theta" else fp[name] - fm[name]).norm()
                / fp[name].norm()
            ).item()
            for name in fp
        }
        assert max(parity.values()) <= 1e-12
        gen = torch.Generator().manual_seed(seed + probe)
        for name in ("theta", "bias", "eta"):
            direction = torch.randn(4, 32, generator=gen, dtype=torch.float64)
            key = {"theta": "theta_delta", "bias": "bias_delta", "eta": "eta"}[name]
            finite = (
                evaluate(s, x[:17], probe, **{key: 1e-6 * direction})
                - evaluate(s, x[:17], probe, **{key: -1e-6 * direction})
            ) / (2e-6)
            analytic = fp[name][:17] @ direction.flatten()
            torch.testing.assert_close(analytic, finite, rtol=1e-4, atol=1e-7)
        raw = {name: (fp[name] + fm[name]) / 2 for name in ("bias", "eta")}
        means = {k: v[:1024].mean(0) for k, v in raw.items()}
        scales = {
            k: (v[:1024] - means[k]).square().mean(0).sqrt().clamp_min(1e-12)
            for k, v in raw.items()
        }
        f = {k: (v - means[k]) / scales[k] for k, v in raw.items()}
        target = x[:, probe] * x[:, (probe + 1) % 32]
        ym = target[:1024].mean()
        designs = {
            "bias": f["bias"],
            "duplicate": torch.cat((f["bias"], f["bias"]), 1) / math.sqrt(2),
            "augmented": torch.cat((f["bias"], f["eta"]), 1) / math.sqrt(2),
        }
        u, sv, _ = torch.linalg.svd(f["bias"][:1024], full_matrices=False)
        rank = int((sv > sv.max() * 1e-9).sum())
        novel = f["eta"][:1024] - u[:, :rank] @ (u[:, :rank].T @ f["eta"][:1024])
        novel_fraction = (novel.square().sum() / f["eta"][:1024].square().sum()).item()
        payload = dict(
            x=x, target=target, ymean=ym, **{f"design_{k}": v for k, v in designs.items()}
        )
        fits = []
        predictions = {}
        for name, F in designs.items():
            u_fit, singular_fit, vh_fit = torch.linalg.svd(F[:1024], full_matrices=False)
            projected_target = u_fit.T @ (target[:1024] - ym)
            for lr in p["rates"]:
                coef = vh_fit.T @ (
                    (singular_fit / (singular_fit.square() + 1024 * lr)) * projected_target
                )
                pred = ym + F @ coef
                predictions[name, lr] = pred
                payload[f"coef_{name}_{lr}"] = coef
                fits.append(
                    dict(
                        arm=name,
                        rate=lr,
                        validation_mse=(pred[1024:1536] - target[1024:1536]).square().mean().item(),
                        reporting_mse=(pred[1536:] - target[1536:]).square().mean().item(),
                        coefficient_norm=coef.norm().item(),
                    )
                )
        duplicate_error = max(
            (predictions["bias", lr] - predictions["duplicate", lr]).abs().max().item()
            for lr in p["rates"]
        )
        assert duplicate_error <= 1e-9
        selected = {
            name: min(
                (v for v in fits if v["arm"] == name),
                key=lambda v: (v["validation_mse"], v["rate"]),
            )
            for name in designs
        }
        replays = []
        for name in ("bias", "augmented"):
            coef = payload[f"coef_{name}_{selected[name]['rate']}"]
            db = coef[:128] / scales["bias"] / (math.sqrt(2) if name == "augmented" else 1)
            de = (
                coef[128:] / scales["eta"] / math.sqrt(2)
                if name == "augmented"
                else torch.zeros(128, dtype=torch.float64)
            )
            intercept = ym - means["bias"] @ db - means["eta"] @ de
            for fraction in p["fractions"]:
                kwargs = dict(
                    bias_delta=fraction * db.reshape(4, 32),
                    eta=fraction * de.reshape(4, 32),
                    output_bias=fraction * intercept,
                )
                exact = 0.5 * (
                    evaluate(s, x[1536:], probe, **kwargs) + evaluate(s, -x[1536:], probe, **kwargs)
                )
                payload[f"replay_{name}_{fraction}"] = exact
                replays.append(
                    dict(
                        arm=name,
                        fraction=fraction,
                        mse=(exact - target[1536:]).square().mean().item(),
                        bias_update_norm=(fraction * db).norm().item(),
                        eta_update_norm=(fraction * de).norm().item(),
                    )
                )
        path = ROOT / f"case{len(cases)}.pt"
        torch.save(payload, path)
        zero = target[1536:].square().mean().item()
        tangent_ratio = selected["augmented"]["reporting_mse"] / selected["bias"]["reporting_mse"]
        replay = next(v for v in replays if v["arm"] == "augmented" and v["fraction"] == 0.1)
        cases.append(
            dict(
                seed=seed,
                probe=probe,
                parity=parity,
                bias_rank=rank,
                eta_novel_energy_fraction=novel_fraction,
                duplicate_error=duplicate_error,
                fits=fits,
                selected=selected,
                replays=replays,
                zero_mse=zero,
                tangent_ratio=tangent_ratio,
                replay_ratio=replay["mse"] / zero,
                passed=tangent_ratio <= 0.95 and replay["mse"] <= 0.95 * zero,
                path=path.as_posix(),
                sha256=sha(path),
            )
        )
        print(
            seed,
            probe,
            "tangent",
            round(tangent_ratio, 4),
            "finite replay",
            round(replay["mse"] / zero, 4),
            flush=True,
        )
assert not torch.cuda.is_initialized()
hashes(p["sources"])
hashes(p["maintained_files"])
write_json(
    ROOT / "result.json",
    dict(
        cases=cases,
        passed=all(v["passed"] for v in cases),
        ridge_solves=81,
        training_updates=0,
        backwards=0,
        cuda_initialized=False,
    ),
)
