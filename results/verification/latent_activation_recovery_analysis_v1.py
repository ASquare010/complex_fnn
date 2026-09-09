"""Independent H083 artifact, selection, held-out and reset verification; no updates."""

import gc
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import equal_payload, read, sha, write_json
from results.latent_activation_recovery_v1.source.adapter import study as s
from src.core.native_recompute_audit import finite_tree

ROOT = Path("results/latent_activation_recovery_v1")
OUT = Path("results/verification/latent_activation_recovery_analysis_v1.json")
TASKS = ("smooth", "oscillatory", "multiplicative", "piecewise")
FORMS = ("plain", "latent_affine", "latent_tanh", "latent_sine", "narrow_swiglu",
         "narrow_gelu", "full_swiglu", "full_gelu")
SEEDS, RATES = (17, 29, 43), (0.001, 0.003)
COUNTS = dict(zip(FORMS, (350208, 350256, 350256, 350256, 350208, 350208, 1179648, 1179648)))


def tensor_sha(t):
    t = t.detach().cpu().contiguous()
    header = json.dumps([str(t.dtype), list(t.shape)]).encode()
    return hashlib.sha256(header + t.numpy().tobytes()).hexdigest()


def load(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, path
    value = torch.load(path, map_location="cpu", weights_only=True)
    assert finite_tree(value), path
    return value


def geom(values):
    assert all(v > 0 and math.isfinite(v) for v in values)
    return math.exp(statistics.mean(math.log(v) for v in values))


@torch.no_grad()
def score(model, x, y):
    sums = [F.mse_loss(model(x[i:i+256]), y[i:i+256], reduction="sum").item()
            for i in range(0, len(x), 256)]
    return sum(sums) / y.numel()


def normalized_observation(x):
    if isinstance(x, dict):
        return {k: normalized_observation(v) for k, v in x.items()
                if k not in ("tensor_file", "tensor_sha256")}
    if isinstance(x, list):
        return [normalized_observation(v) for v in x]
    return x


def verify_prior(before):
    anchors = {
        "h081_final_sha256": "results/verification/blast_learning_screen_final_v1.json",
        "h081_result_sha256": "results/blast_learning_screen_v1/result.json",
        "h081_source_sha256": "results/blast_learning_screen_v1/source.zip",
        "h082_failure_audit_sha256": "results/verification/latent_activation_preflight_failure_v1.json",
        "h082_source_sha256": "results/latent_activation_v1/source.zip",
        "h082_protocol_sha256": "results/latent_activation_v1/protocol.json",
        "h082_preflight_log_sha256": "results/latent_activation_v1/processes/preflight.log",
        "h082_report_sha256": "research/latent_activation_results.md",
    }
    for key, path in anchors.items():
        assert sha(path) == before[key], path
    for name, record in before["h082_files"].items():
        p = Path(name)
        assert sha(p) == record["sha256"] and p.stat().st_size == record["size"], name
        assert p.stat().st_mtime_ns == record["mtime_ns"], name
    assert len(before["h082_files"]) == 27
    assert all(sha(n) == h for n, h in before["prior_plans"].items())


def verify_preflight():
    files = sorted((ROOT / "preflight").glob("*.json"))
    assert len(files) == 8
    assert all(read(n)["status"] == "PASS" and read(n)["optimizer_updates"] == 0 for n in files)
    previous = Path("results/latent_activation_v1/preflight")
    old_files = sorted(previous.glob("*.json"))
    assert len(old_files) == 7
    for old in old_files:
        equal_payload(normalized_observation(read(old)),
                      normalized_observation(read(ROOT / "preflight" / old.name)))
    rows = read(ROOT / "preflight/fidelity.json")["cases"]
    pairs = 0
    for name, row in rows.items():
        path = Path(row["tensor_file"])
        assert sha(path) == row["tensor_sha256"]
        value = load(path)
        equal_payload(value, load(previous / (name + ".pt")))
        equal_payload(value["nonzero_eager"], value["nonzero_checkpoint"])
        a, b = value["identity_plain"], value["identity_curve"]
        assert torch.equal(a["output"], b["output"])
        for n, v in a["gradients"].items():
            assert torch.equal(v, b["gradients"][n])
        if not name.endswith("affine"):
            assert all(torch.count_nonzero(v) == 0 for n, v in b["gradients"].items()
                       if n.endswith("theta_b"))
        assert row["identity_gradient_pairs"] == len(a["gradients"])
        assert row["checkpoint_gradient_pairs"] == len(value["nonzero_eager"]["gradients"])
        pairs += row["identity_gradient_pairs"] + row["checkpoint_gradient_pairs"]
    assert len(rows) == 6 and pairs == 120
    return {"checks": 8, "prior_observations_exact": 7, "prior_tensor_payloads_exact": 6,
            "identity_checkpoint_gradient_pairs_exact": pairs, "optimizer_updates": 0}


def verify_data(manifest):
    assert sha(ROOT / "data.pt") == manifest["data_sha256"]
    data = load(ROOT / "data.pt")
    assert set(data) == {"x", "targets", "train_std", "streams", "feature_permutation",
                         "neighbor_cross_group_fraction"}
    x = (2*torch.rand(73728, 384, generator=torch.Generator().manual_seed(9812))-1)*math.sqrt(3)
    assert torch.equal(x, data["x"]) and tensor_sha(x) == manifest["x_sha256"]
    perm = torch.randperm(384, generator=torch.Generator().manual_seed(9283))
    assert torch.equal(perm, data["feature_permutation"])
    fraction = ((perm//48) != (perm.roll(-1)//48)).float().mean().item()
    assert fraction == data["neighbor_cross_group_fraction"]
    rotation = torch.linalg.qr(torch.randn(384, 384, generator=torch.Generator().manual_seed(9282))).Q
    u = x[:, perm]
    a, b, c, d = [u.roll(-i, dims=-1) for i in range(4)]
    for task in TASKS:
        if task == "smooth":
            raw = torch.sin(2*a)+b.square()+torch.exp(0.5*c)-d
        elif task == "oscillatory":
            raw = torch.sin(3*a)*torch.cos(2*b)+0.5*torch.sin(4*c+d)
        elif task == "multiplicative":
            raw = a*b+2*b*c*d+a.square()*d
        else:
            raw = F.relu(a+b)-0.7*(c-d).abs()+torch.where(a > 0, b, c)
        raw = raw @ rotation
        std = raw[:65536].std(dim=0, correction=0)
        y = raw/std
        assert torch.equal(std, data["train_std"][task])
        assert torch.equal(y, data["targets"][task]) and tensor_sha(y) == manifest["targets"][task]
        print(json.dumps({"regenerated_task": task}), flush=True)
    for seed in SEEDS:
        stream = torch.randint(65536, (600, 256), generator=torch.Generator().manual_seed(20000+seed))
        assert torch.equal(stream, data["streams"][seed])
        assert tensor_sha(stream) == manifest["streams"][str(seed)]
        assert stream.min() >= 0 and stream.max() < 65536
    return data


def expected_scale(form, name):
    if ".curve." in name:
        return 1.0
    if form == "plain" or form.startswith("latent_"):
        return 2048/96 if name == "down.second.weight" else 4.0
    if form.startswith("narrow_") and name == "down.weight":
        return 1536/456 if form == "narrow_gelu" else 1024/304
    return 1.0


def verify_optimizer(cp, model, form, rate, record):
    params = dict(model.named_parameters())
    expected = {}
    for name, parameter in params.items():
        expected.setdefault(expected_scale(form, name), []).append((name, parameter))
    groups = cp["optimizer"]["param_groups"]
    assert len(groups) == len(expected)
    seen, summary = [], []
    for group, (scale, named) in zip(groups, expected.items()):
        assert group["lr"] == rate*scale and group["lr_scale"] == scale
        assert group["weight_decay"] == 0 and group["betas"] == (0.9, 0.95) and group["eps"] == 1e-8
        assert len(group["params"]) == len(named)
        count = 0
        for index, (name, parameter) in zip(group["params"], named):
            seen.append(index)
            state = cp["optimizer"]["state"][index]
            assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
            assert state["step"].item() == 600
            assert state["exp_avg"].shape == state["exp_avg_sq"].shape == parameter.shape
            assert (state["exp_avg_sq"] >= 0).all()
            assert cp["model"][name].shape == parameter.shape
            count += parameter.numel()
        summary.append({"weight_decay": 0, "lr_scale": scale, "parameter_count": count})
    assert len(seen) == len(set(seen)) == len(params)
    assert set(seen) == set(cp["optimizer"]["state"])
    assert record["groups"] == summary


def main():
    assert not OUT.exists()
    assert read(ROOT / "coordinator_status.json")["status"] == "PASS"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol, manifest, result, before = [read(ROOT / n) for n in
                                          ("protocol.json", "data_manifest.json", "result.json", "before.json")]
    assert result["status"] == "complete" and not result["research_goal_achieved"]
    assert protocol["forms"] == list(FORMS) and protocol["counts"] == COUNTS
    assert protocol["tasks"] == list(TASKS) and protocol["seeds"] == list(SEEDS)
    assert protocol["rates"] == list(RATES) and protocol["steps"] == 600 and protocol["batch"] == 256
    assert len(protocol["sources"]) == 128
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    assert sha(s.PLAN) == protocol["plan_sha256"] == before["plan_sha256"]
    assert sha("research/latent_activation_theory.md") == protocol["theory_sha256"]
    assert sha("research/latent_activation_plan.md") == before["original_plan_sha256"]
    with zipfile.ZipFile(ROOT / "source.zip") as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 131
        assert all(hashlib.sha256(z.read(n)).hexdigest() == sha(n) for n in z.namelist())
    verify_prior(before)
    preflight = verify_preflight()
    for phase in ("preflight", "fitting"):
        p = read(ROOT / "processes" / (phase+".json"))
        assert p["status"] == "PASS" and p["returncode"] == 0 and p["source_unchanged"]
        assert sha(ROOT / "processes" / (phase+".log")) == p["log_sha256"]
    assert manifest["protocol_sha256"] == sha(ROOT / "protocol.json") == result["protocol_sha256"]
    data = verify_data(manifest)
    initial = {}
    with s.bound():
        for seed in SEEDS:
            for form in FORMS:
                torch.manual_seed(seed)
                model = s.make_model(form, seed)
                initial[form, seed] = {
                    "hashes": {n: tensor_sha(t) for n, t in model.state_dict().items()},
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.Generator(device="cuda").manual_seed(seed).get_state(),
                }
                assert sum(p.numel() for p in model.parameters()) == COUNTS[form]
                if form.startswith("latent_"):
                    assert {n: h for n, h in initial[form, seed]["hashes"].items() if ".curve." not in n} == initial["plain", seed]["hashes"]
                del model
    records, selected, artifacts, ablations = {}, {}, {}, {}
    selection_rescores, heldout_rescores = 0, 0
    x_select, x_heldout = data["x"][65536:69632].cuda(), data["x"][69632:].cuda()
    with s.bound(), torch.no_grad():
        for task in TASKS:
            y_select = data["targets"][task][65536:69632].cuda()
            y_heldout = data["targets"][task][69632:].cuda()
            zero = y_heldout.double().square().mean().item()
            for seed in SEEDS:
                for form in FORMS:
                    selection_path = ROOT / "selections" / f"{task}_{form}_s{seed}.json"
                    choice = read(selection_path)
                    rates = []
                    for rate in RATES:
                        name = f"{task}_{form}_s{seed}_lr{round(rate*1e6)}"
                        cell = ROOT / "cells" / name
                        record, held = read(cell / "training.json"), read(cell / "heldout.json")
                        cp = load(cell / "checkpoint.pt")
                        assert record["cell"] == cp["cell"] == name and record["base_rate"] == rate
                        assert (record["task"], record["form"], record["seed"]) == (task, form, seed)
                        assert record["checkpoint_sha256"] == held["checkpoint_sha256"] == sha(cell / "checkpoint.pt")
                        assert cp["step"] == record["steps"] == 600 and record["parameters"] == COUNTS[form]
                        assert record["all_final_weights_and_moments_finite"] and finite_tree(record)
                        hashes = {"data_sha256": manifest["data_sha256"],
                                  "sampler_sha256": manifest["streams"][str(seed)],
                                  "protocol_sha256": manifest["protocol_sha256"]}
                        assert all(cp[k] == record[k] == v for k, v in hashes.items())
                        assert record["initial_state_hashes"] == initial[form, seed]["hashes"]
                        assert record["final_state_hashes"] == {n: tensor_sha(t) for n, t in cp["model"].items()}
                        assert torch.equal(cp["cpu_rng"], initial[form, seed]["cpu_rng"])
                        assert torch.equal(cp["cuda_rng"], initial[form, seed]["cuda_rng"])
                        history = record["history"]
                        assert len(history) == 600 and [h["step"] for h in history] == list(range(1, 601))
                        assert all(h["loss"] >= 0 and h["pre_clip_norm"] >= 0 and h["update_ms"] > 0 for h in history)
                        assert record["median_update_ms"] == statistics.median(h["update_ms"] for h in history[50:])
                        assert record["clip_fraction"] == sum(h["pre_clip_norm"] > 1 for h in history)/600
                        assert record["peak_reserved_bytes"] >= record["peak_allocated_bytes"] > 0
                        if form.startswith("latent_"):
                            plain = records[f"{task}_plain_s{seed}_lr{round(rate*1e6)}"]
                            assert history[0]["loss"] == plain["history"][0]["loss"]
                        model = s.make_model(form, seed)
                        verify_optimizer(cp, model, form, rate, record)
                        assert set(record["final_clipped_gradient_norms"]) == set(dict(model.named_parameters()))
                        model.load_state_dict(cp["model"], strict=True)
                        model.cuda().eval()
                        actual_selection = score(model, x_select, y_select)
                        assert actual_selection == record["selection_mse"], (name, "selection", actual_selection, record["selection_mse"])
                        selection_rescores += 1
                        assert held["zero_mse"] == zero and held["mse"] > 0 and math.isfinite(held["mse"])
                        assert (cell / "training.json").stat().st_mtime_ns <= selection_path.stat().st_mtime_ns
                        assert selection_path.stat().st_mtime_ns <= (cell / "heldout.json").stat().st_mtime_ns
                        if choice["selected_cell"] == name:
                            actual = score(model, x_heldout, y_heldout)
                            assert actual == held["mse"], (name, "heldout", actual, held["mse"])
                            heldout_rescores += 1
                            selected[task, form, seed] = {
                                "task": task, "form": form, "seed": seed, "cell": name,
                                "rate": rate, "selection_mse": actual_selection,
                                "heldout_mse": actual, "zero_mse": zero, "finite": True}
                            if form.startswith("latent_"):
                                for projection in (model.up, model.gate, model.down):
                                    projection.curve.theta_a.zero_()
                                    if form == "latent_affine":
                                        projection.curve.theta_b.zero_()
                                reset = score(model, x_heldout, y_heldout)
                                saved_reset = read(cell / "ablation.json")
                                assert saved_reset == {"reset_mse": reset, "original_mse": actual,
                                                       "ratio": reset/actual, "optimizer_updates": 0}
                                assert all(torch.equal(t.cpu(), cp["model"][n]) for n, t in model.state_dict().items()
                                           if ".curve." not in n or (form != "latent_affine" and n.endswith("theta_b")))
                                ablations[name] = saved_reset
                        else:
                            assert not (cell / "ablation.json").exists()
                        records[name] = record
                        rates.append(record)
                        artifacts[name] = {n.name: sha(n) for n in cell.iterdir() if n.is_file()}
                        del model, cp
                    best = min(rates, key=lambda r: (r["selection_mse"], r["base_rate"]))
                    assert choice == {"selected_cell": best["cell"],
                                      "criterion": "final selection MSE, then smaller rate",
                                      "values": {r["cell"]: r["selection_mse"] for r in rates}}
            print(json.dumps({"audited_task": task, "checkpoints": len(records),
                              "heldout_rescores": heldout_rescores, "ablations": len(ablations)}), flush=True)
            del y_select, y_heldout
            gc.collect()
            torch.cuda.empty_cache()
    assert len(records) == len(artifacts) == selection_rescores == 192
    assert len(selected) == heldout_rescores == 96 and len(ablations) == 36
    assert len(list((ROOT / "cells").iterdir())) == 192
    assert len(list((ROOT / "selections").glob("*.json"))) == 96
    expected_rows = [selected[t, f, seed] for t in TASKS for seed in SEEDS for f in FORMS]
    assert result["selected_rows"] == expected_rows

    def ratio(form, reference, tasks=TASKS, seeds=SEEDS):
        return geom([selected[t, form, seed]["heldout_mse"]/selected[t, reference, seed]["heldout_mse"]
                     for t in tasks for seed in seeds])

    def over_zero(form, task):
        return geom([selected[task, form, seed]["heldout_mse"]/selected[task, form, seed]["zero_mse"]
                     for seed in SEEDS])

    positive = sum(over_zero("full_gelu", t) <= 0.98 for t in TASKS) >= 2
    decisions = {}
    for form in ("latent_tanh", "latent_sine"):
        checks = {
            "complete_finite_grid": True,
            "positive_control_assay_passes": positive,
            "two_percent_over_plain": ratio(form, "plain") <= 0.98,
            "every_seed_better_than_plain": all(ratio(form, "plain", seeds=(seed,)) < 1 for seed in SEEDS),
            "one_percent_over_equal_count_affine": ratio(form, "latent_affine") <= 0.99,
            "beats_both_calibrated_narrows": all(ratio(form, ref) < 1 for ref in ("narrow_swiglu", "narrow_gelu")),
            "per_task_regression_cap_vs_plain": all(ratio(form, "plain", tasks=(t,)) <= 1.05 for t in TASKS),
            "per_task_regression_cap_vs_zero": all(over_zero(form, t) <= 1.05 for t in TASKS),
            "seventy_percent_reduction": COUNTS[form] <= 0.3*COUNTS["full_gelu"],
        }
        decisions[form] = {"tests": checks, "earns_full_model_resource_qualification": all(checks.values())}
    ratios = {f: {ref: ratio(f, ref) for ref in FORMS} for f in FORMS}
    task_ratios = {t: {f: {ref: ratio(f, ref, tasks=(t,)) for ref in FORMS} for f in FORMS} for t in TASKS}
    assert decisions == result["gates"] and positive == result["assay_passed"]
    assert ratios == result["ratios"]
    assert {t: {f: over_zero(f, t) for f in FORMS} for t in TASKS} == result["ratios_to_zero"]
    assert {f: {str(seed): ratio(f, "plain", seeds=(seed,)) for seed in SEEDS} for f in FORMS} == result["by_seed_ratios_to_plain"]
    assert {t: {f: statistics.mean(selected[t, f, seed]["heldout_mse"] for seed in SEEDS) for f in FORMS} for t in TASKS} == result["task_mean_mse"]
    verdict = "EARNS_RESOURCE_QUALIFICATION" if any(g["earns_full_model_resource_qualification"] for g in decisions.values()) else "REJECTED_AT_THIS_FITTING_BUDGET" if positive else "INCONCLUSIVE_ASSAY_FAILURE"
    assert verdict == result["scientific_verdict"]
    assert result["optimizer_updates"] == 192*600 == 115200
    assert result["training_example_presentations"] == 115200*256
    assert result["corpus_targets"] == result["full_model_resource_workers"] == 0
    assert result["post_training_reset_ablations"] == 36
    verify_prior(before)
    assert all(sha(n) == h for n, h in protocol["sources"].items())
    resources = {}
    shapes = {}
    for form in FORMS:
        chosen = [records[selected[t, form, seed]["cell"]] for t in TASKS for seed in SEEDS]
        resources[form] = {
            "mean_cell_median_update_ms": statistics.mean(r["median_update_ms"] for r in chosen),
            "max_peak_allocated_mib": max(r["peak_allocated_bytes"] for r in chosen)/2**20,
            "mean_clip_fraction": statistics.mean(r["clip_fraction"] for r in chosen),
            "upper_rate_selected": sum(r["base_rate"] == 0.003 for r in chosen),
        }
        if form.startswith("latent_"):
            curves = [r["final_diagnostics"][proj]["curve"] for r in chosen for proj in ("up", "gate", "down")]
            shapes[form] = {
                "mean_absolute_amplitude": statistics.mean(abs(a) for c in curves for a in c["amplitude"]),
                "sampled_slope_min": min(c["slope_min"] for c in curves),
                "sampled_slope_max": max(c["slope_max"] for c in curves),
                "mean_control_saturation_fraction": statistics.mean(c["control_saturation_fraction"] for c in curves),
                "mean_correction_rms": statistics.mean(c["correction_rms"] for c in curves),
                "reset_to_original_geometric_ratio": geom([ablations[r["cell"]]["ratio"] for r in chosen]),
                "per_task_reset_to_original_geometric_ratio": {
                    t: geom([ablations[selected[t, form, seed]["cell"]]["ratio"] for seed in SEEDS]) for t in TASKS},
            }
    record = {
        "status": "PASS", "scientific_verdict": verdict, "research_goal_achieved": False,
        "checkpoint_records_verified": 192, "optimizer_updates_verified": 115200,
        "selection_rescores_exact": selection_rescores, "selected_heldout_rescores_exact": heldout_rescores,
        "reset_ablations_exact": len(ablations), "analysis_optimizer_updates": 0,
        "analysis_selection_examples": 192*4096, "analysis_heldout_examples": (96+36)*4096,
        "data_regenerated_exact": True, "initial_states_reconstructed": 24,
        "rng_states_exact": 192, "prior_h082_files_unchanged": 27,
        "selection_before_heldout_records_verified": 96, "preflight": preflight,
        "protocol_sha256": sha(ROOT / "protocol.json"), "result_sha256": sha(ROOT / "result.json"),
        "source_archive_sha256": sha(ROOT / "source.zip"), "data_sha256": sha(ROOT / "data.pt"),
        "cell_artifact_hashes": artifacts,
        "selection_artifact_hashes": {n.name: sha(n) for n in (ROOT / "selections").glob("*.json")},
        "ratios": ratios, "task_ratios": task_ratios, "gates": decisions,
        "resources": resources, "learned_shapes": shapes,
        "explicit_preflight_repetitions": 1, "fitting_repetitions": 0,
        "limitations": ["fixed synthetic tasks, one dataset and three optimization seeds",
                        "600 updates do not establish convergence", "standalone resources are not Transformer resources",
                        "reset dependence is not a causal training-trajectory attribution",
                        "selection persistence is verified against source order and recorded filesystem timestamps"],
    }
    write_json(OUT, record)
    print(json.dumps({k: v for k, v in record.items() if k not in
                      ("cell_artifact_hashes", "selection_artifact_hashes", "task_ratios")}), flush=True)


if __name__ == "__main__":
    main()
