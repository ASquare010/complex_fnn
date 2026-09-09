"""Independent H085 audit, including both-rate reporting rescores for the robustness gate."""

import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F

from results.blast_operator_recovery_v1.source.storage import read, sha, write_json
from results.latent_offset_fit_v1.source import study as s
from src.core.native_recompute_audit import finite_tree

ROOT = Path("results/latent_offset_fit_v1")
OUT = Path("results/verification/latent_offset_fit_analysis_v1.json")
TASKS = ("smooth", "oscillatory", "multiplicative", "piecewise")
FORMS = ("plain", "plain_first_lr", "latent_gain", "latent_offset", "latent_offset_first_lr",
         "latent_affine", "narrow_swiglu", "narrow_gelu", "full_swiglu", "full_gelu")
SEEDS, RATES = (17, 29, 43), (0.001, 0.003)
COUNTS = dict(zip(FORMS, (350208, 350208, 350232, 350232, 350232, 350256, 350208, 350208, 1179648, 1179648)))


def load(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, path
    value = torch.load(path, map_location="cpu", weights_only=True)
    assert finite_tree(value), path
    return value


def thash(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(json.dumps([str(t.dtype), list(t.shape)]).encode()+t.numpy().tobytes()).hexdigest()


def geom(v):
    assert all(x > 0 and math.isfinite(x) for x in v)
    return math.exp(statistics.mean(math.log(x) for x in v))


def prior(before):
    assert all(sha(n) == h for n, h in before["anchors"].items())
    assert all(sha(n) == h for n, h in before["prior_plans"].items())
    assert all(sha(n) == h for n, h in before["prior_sources"].items())


def qualification():
    observations = list((ROOT / "preflight").glob("*.json"))
    assert len(observations) == 8
    assert all(read(n)["status"] == "PASS" and read(n)["optimizer_updates"] == 0 for n in observations)
    pairs = 0
    for device in ("cpu", "cuda"):
        for form in ("latent_gain", "latent_offset"):
            row = read(ROOT / "preflight" / (device+"_"+form+".json"))
            path = Path(row["tensor_file"])
            assert sha(path) == row["tensor_sha256"]
            value = load(path)
            reference, eager, checkpoint = (value[k] for k in ("full_affine", "partial_eager", "partial_checkpoint"))
            assert torch.equal(reference["output"], eager["output"]) and torch.equal(eager["output"], checkpoint["output"])
            assert len(eager["gradients"]) == row["retained_gradient_pairs"] == row["checkpoint_gradient_pairs"] == 10
            for name, tensor in eager["gradients"].items():
                assert torch.equal(tensor, reference["gradients"][name]) and torch.equal(tensor, checkpoint["gradients"][name])
            pairs += 20
    assert pairs == 80
    return pairs


def regenerate(manifest):
    assert sha(ROOT / "data.pt") == manifest["data_sha256"]
    data = load(ROOT / "data.pt")
    x = (2*torch.rand(73728, 384, generator=torch.Generator().manual_seed(9813))-1)*math.sqrt(3)
    assert torch.equal(x, data["x"]) and thash(x) == manifest["x_sha256"]
    assert manifest["x_sha256"] != read("results/latent_activation_recovery_v1/data_manifest.json")["x_sha256"]
    permutation = torch.randperm(384, generator=torch.Generator().manual_seed(9283))
    assert torch.equal(permutation, data["feature_permutation"])
    assert ((permutation//48) != (permutation.roll(-1)//48)).float().mean().item() == data["neighbor_cross_group_fraction"]
    q = torch.linalg.qr(torch.randn(384, 384, generator=torch.Generator().manual_seed(9282))).Q
    u = x[:, permutation]
    a, b, c, d = [u.roll(-i, dims=-1) for i in range(4)]
    for task in TASKS:
        if task == "smooth":
            y = torch.sin(2*a)+b.square()+torch.exp(0.5*c)-d
        elif task == "oscillatory":
            y = torch.sin(3*a)*torch.cos(2*b)+0.5*torch.sin(4*c+d)
        elif task == "multiplicative":
            y = a*b+2*b*c*d+a.square()*d
        else:
            y = F.relu(a+b)-0.7*(c-d).abs()+torch.where(a > 0, b, c)
        y = y @ q
        std = y[:65536].std(0, correction=0)
        y = y/std
        assert torch.equal(std, data["train_std"][task]) and torch.equal(y, data["targets"][task])
        assert thash(y) == manifest["targets"][task]
        print(json.dumps({"regenerated_task": task}), flush=True)
    for seed in SEEDS:
        stream = torch.randint(65536, (600, 256), generator=torch.Generator().manual_seed(20000+seed))
        assert torch.equal(stream, data["streams"][seed]) and thash(stream) == manifest["streams"][str(seed)]
    return data


def scale(form, name):
    if ".curve." in name:
        return 1.0
    if form in FORMS[:6]:
        result = 2048/96 if name == "down.second.weight" else 4.0
        return result*(5/8 if form.endswith("_first_lr") and name.endswith("first.weight") else 1)
    if form.startswith("narrow_") and name == "down.weight":
        return 1536/456 if form == "narrow_gelu" else 1024/304
    return 1.0


def optimizer(cp, model, form, rate, record):
    grouped = {}
    for name, p in model.named_parameters():
        grouped.setdefault(scale(form, name), []).append((name, p))
    groups = cp["optimizer"]["param_groups"]
    assert len(groups) == len(grouped)
    seen, summaries = [], []
    for actual, (expected_scale, params) in zip(groups, grouped.items()):
        assert actual["lr_scale"] == expected_scale and actual["lr"] == rate*expected_scale
        assert actual["betas"] == (0.9, 0.95) and actual["eps"] == 1e-8 and actual["weight_decay"] == 0
        assert len(actual["params"]) == len(params)
        count = 0
        for idx, (name, parameter) in zip(actual["params"], params):
            seen.append(idx)
            state = cp["optimizer"]["state"][idx]
            assert set(state) == {"step", "exp_avg", "exp_avg_sq"} and state["step"].item() == 600
            assert state["exp_avg"].shape == state["exp_avg_sq"].shape == parameter.shape == cp["model"][name].shape
            assert (state["exp_avg_sq"] >= 0).all()
            count += parameter.numel()
        summaries.append({"weight_decay": 0, "lr_scale": expected_scale, "parameter_count": count})
    assert len(seen) == len(set(seen)) == len(list(model.parameters()))
    assert set(seen) == set(cp["optimizer"]["state"])
    assert record["groups"] == summaries
    buffers = dict(model.named_buffers())
    if form in ("latent_gain", "latent_offset", "latent_offset_first_lr"):
        fixed = "theta_b" if form == "latent_gain" else "theta_a"
        assert sum(v.numel() for v in buffers.values()) == 24
        assert all(n.endswith(fixed) and not v.requires_grad and torch.count_nonzero(cp["model"][n]) == 0 for n, v in buffers.items())
    else:
        assert not buffers


@torch.no_grad()
def score(model, x, y):
    return sum(F.mse_loss(model(x[i:i+256]), y[i:i+256], reduction="sum").item() for i in range(0, len(x), 256))/y.numel()


assert not OUT.exists() and read(ROOT / "coordinator_status.json")["status"] == "PASS"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
p, r, before, manifest = [read(ROOT / n) for n in ("protocol.json", "result.json", "before.json", "data_manifest.json")]
assert p["forms"] == list(FORMS) and p["counts"] == COUNTS and p["input_seed"] == 9813
assert p["tasks"] == list(TASKS) and p["seeds"] == list(SEEDS) and p["rates"] == list(RATES)
assert p["steps"] == 600 and p["batch"] == 256 and p["first_factor_lr_multiplier"] == 5/8
assert len(p["sources"]) == 135 and all(sha(n) == h for n, h in p["sources"].items())
assert sha(s.PLAN) == p["plan_sha256"] == before["plan_sha256"]
assert manifest["protocol_sha256"] == r["protocol_sha256"] == sha(ROOT / "protocol.json")
assert r["status"] == "complete" and not r["research_goal_achieved"]
prior(before)
with zipfile.ZipFile(ROOT / "source.zip") as z:
    assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 136
    assert all(hashlib.sha256(z.read(n)).hexdigest() == sha(n) for n in z.namelist())
gradient_pairs = qualification()
for phase in ("preflight", "fitting"):
    process = read(ROOT / "processes" / (phase+".json"))
    assert process["status"] == "PASS" and process["returncode"] == 0 and process["source_unchanged"]
    assert sha(ROOT / "processes" / (phase+".log")) == process["log_sha256"]
data = regenerate(manifest)
initial = {}
with s.bound():
    for seed in SEEDS:
        for form in FORMS:
            torch.manual_seed(seed)
            model = s.make_model(form, seed)
            initial[form, seed] = {"hashes": {n: thash(v) for n, v in model.state_dict().items()},
                                   "cpu_rng": torch.get_rng_state(),
                                   "cuda_rng": torch.Generator(device="cuda").manual_seed(seed).get_state()}
            assert sum(v.numel() for v in model.parameters()) == COUNTS[form]
            if form in FORMS[1:6]:
                assert {n: h for n, h in initial[form, seed]["hashes"].items() if ".curve." not in n} == initial["plain", seed]["hashes"]
            del model
records, selected, rated, artifacts, resets = {}, {}, {}, {}, {}
x_select, x_report = data["x"][65536:69632].cuda(), data["x"][69632:].cuda()
with s.bound(), torch.no_grad():
    for task in TASKS:
        ys, yr = data["targets"][task][65536:69632].cuda(), data["targets"][task][69632:].cuda()
        zero = yr.double().square().mean().item()
        for seed in SEEDS:
            for form in FORMS:
                choice_path = ROOT / "selections" / f"{task}_{form}_s{seed}.json"
                choice, candidates = read(choice_path), []
                for rate in RATES:
                    name = f"{task}_{form}_s{seed}_lr{round(rate*1e6)}"
                    path = ROOT / "cells" / name
                    row, held, cp = read(path / "training.json"), read(path / "heldout.json"), load(path / "checkpoint.pt")
                    assert row["cell"] == cp["cell"] == name and row["base_rate"] == rate
                    assert (row["task"], row["form"], row["seed"]) == (task, form, seed)
                    assert cp["step"] == row["steps"] == 600 and row["parameters"] == COUNTS[form]
                    assert sha(path / "checkpoint.pt") == row["checkpoint_sha256"] == held["checkpoint_sha256"]
                    assert finite_tree(row) and row["all_final_weights_and_moments_finite"]
                    assert row["initial_state_hashes"] == initial[form, seed]["hashes"]
                    assert row["final_state_hashes"] == {n: thash(v) for n, v in cp["model"].items()}
                    assert torch.equal(cp["cpu_rng"], initial[form, seed]["cpu_rng"])
                    assert torch.equal(cp["cuda_rng"], initial[form, seed]["cuda_rng"])
                    hashes = {"protocol_sha256": manifest["protocol_sha256"], "data_sha256": manifest["data_sha256"],
                              "sampler_sha256": manifest["streams"][str(seed)]}
                    assert all(cp[k] == row[k] == v for k, v in hashes.items())
                    h = row["history"]
                    assert len(h) == 600 and [v["step"] for v in h] == list(range(1, 601))
                    assert all(v["loss"] >= 0 and v["pre_clip_norm"] >= 0 and v["update_ms"] > 0 for v in h)
                    assert row["clip_fraction"] == sum(v["pre_clip_norm"] > 1 for v in h)/600
                    assert row["median_update_ms"] == statistics.median(v["update_ms"] for v in h[50:])
                    assert row["peak_reserved_bytes"] >= row["peak_allocated_bytes"] > 0
                    if form in FORMS[1:6]:
                        assert h[0]["loss"] == records[f"{task}_plain_s{seed}_lr{round(rate*1e6)}"]["history"][0]["loss"]
                    model = s.make_model(form, seed)
                    optimizer(cp, model, form, rate, row)
                    assert set(row["final_clipped_gradient_norms"]) == set(dict(model.named_parameters()))
                    model.load_state_dict(cp["model"], strict=True)
                    model.cuda().eval()
                    selection_mse, report_mse = score(model, x_select, ys), score(model, x_report, yr)
                    assert selection_mse == row["selection_mse"] and report_mse == held["mse"], name
                    assert zero == held["zero_mse"]
                    assert (path / "training.json").stat().st_mtime_ns <= choice_path.stat().st_mtime_ns <= (path / "heldout.json").stat().st_mtime_ns
                    rated[task, form, seed, rate] = {"task": task, "form": form, "seed": seed, "rate": rate, "cell": name,
                                                   "heldout_mse": report_mse, "zero_mse": zero, "finite": True}
                    if choice["selected_cell"] == name:
                        selected[task, form, seed] = {"task": task, "form": form, "seed": seed, "cell": name, "rate": rate,
                                                     "selection_mse": selection_mse, "heldout_mse": report_mse, "zero_mse": zero, "finite": True}
                        if form.startswith("latent_"):
                            for projection in (model.up, model.gate, model.down):
                                for parameter in projection.curve.parameters():
                                    parameter.zero_()
                            reset = score(model, x_report, yr)
                            saved = read(path / "ablation.json")
                            assert saved == {"reset_mse": reset, "original_mse": report_mse, "ratio": reset/report_mse, "optimizer_updates": 0}
                            assert all(torch.equal(v.cpu(), cp["model"][n]) for n, v in model.state_dict().items() if ".curve." not in n)
                            resets[name] = saved
                    else:
                        assert not (path / "ablation.json").exists()
                    records[name] = row
                    candidates.append(row)
                    artifacts[name] = {n.name: sha(n) for n in path.iterdir() if n.is_file()}
                    del model, cp
                best = min(candidates, key=lambda v: (v["selection_mse"], v["base_rate"]))
                assert choice == {"selected_cell": best["cell"], "criterion": "final selection MSE, then smaller rate",
                                  "values": {v["cell"]: v["selection_mse"] for v in candidates}}
        print(json.dumps({"audited_task": task, "checkpoints": len(records), "selected": len(selected), "resets": len(resets)}), flush=True)
        del ys, yr
assert len(records) == len(rated) == len(artifacts) == 240 and len(selected) == 120 and len(resets) == 48
assert len(list((ROOT / "cells").iterdir())) == 240 and len(list((ROOT / "selections").glob("*.json"))) == 120
assert r["selected_rows"] == [selected[t, f, seed] for t in TASKS for seed in SEEDS for f in FORMS]
assert r["rate_rows"] == [rated[t, f, seed, rate] for t in TASKS for seed in SEEDS for f in FORMS for rate in RATES]


def ratio(f, ref, tasks=TASKS, seeds=SEEDS, rate=None):
    def value(t, form, seed):
        return (selected[t, form, seed] if rate is None else rated[t, form, seed, rate])["heldout_mse"]
    return geom([value(t, f, seed)/value(t, ref, seed) for t in tasks for seed in seeds])


def zero_ratio(f, t):
    return geom([selected[t, f, seed]["heldout_mse"]/selected[t, f, seed]["zero_mse"] for seed in SEEDS])


positive = sum(zero_ratio("full_gelu", t) <= 0.98 for t in TASKS) >= 2
gates, robust = {}, {}
for form in ("latent_offset", "latent_offset_first_lr"):
    reference = "plain_first_lr" if form.endswith("_first_lr") else "plain"
    by_rate = {str(rate): {"one_percent": ratio(form, reference, rate=rate) <= 0.99,
                           "every_seed": all(ratio(form, reference, seeds=(seed,), rate=rate) < 1 for seed in SEEDS)} for rate in RATES}
    robust[form] = {"reference": reference, "per_rate": by_rate, "passes": all(all(v.values()) for v in by_rate.values())}
    tests = {"complete_finite_grid": True, "positive_control_assay_passes": positive,
             "two_percent_over_plain": ratio(form, "plain") <= 0.98,
             "every_seed_better_than_plain": all(ratio(form, "plain", seeds=(seed,)) < 1 for seed in SEEDS),
             "one_percent_over_equal_count_gain": ratio(form, "latent_gain") <= 0.99,
             "one_percent_over_fixed_lr_plain": ratio(form, "plain_first_lr") <= 0.99,
             "beats_both_calibrated_narrows": all(ratio(form, ref) < 1 for ref in ("narrow_swiglu", "narrow_gelu")),
             "within_one_percent_full_affine": ratio(form, "latent_affine") <= 1.01,
             "per_task_regression_cap_vs_plain": all(ratio(form, "plain", tasks=(t,)) <= 1.05 for t in TASKS),
             "per_task_regression_cap_vs_zero": all(zero_ratio(form, t) <= 1.05 for t in TASKS),
             "seventy_percent_reduction": COUNTS[form] <= 0.3*COUNTS["full_gelu"],
             "matched_rate_offset_benefit": robust[form]["passes"]}
    gates[form] = {"tests": tests, "earns_full_model_resource_qualification": all(tests.values())}
assert gates == r["gates"] and robust == r["robustness"] and positive == r["assay_passed"]
ratios = {f: {ref: ratio(f, ref) for ref in FORMS} for f in FORMS}
fixed = {str(rate): {f: {ref: ratio(f, ref, rate=rate) for ref in FORMS} for f in FORMS} for rate in RATES}
assert ratios == r["ratios"] and fixed == r["fixed_rate_ratios"]
assert {str(rate): {f: {str(seed): ratio(f, "plain_first_lr" if f.endswith("_first_lr") else "plain", seeds=(seed,), rate=rate) for seed in SEEDS} for f in FORMS} for rate in RATES} == r["fixed_rate_by_seed_ratios"]
assert {f: {str(seed): ratio(f, "plain", seeds=(seed,)) for seed in SEEDS} for f in FORMS} == r["by_seed_ratios_to_plain"]
assert {t: {f: zero_ratio(f, t) for f in FORMS} for t in TASKS} == r["ratios_to_zero"]
assert {t: {f: statistics.mean(selected[t, f, seed]["heldout_mse"] for seed in SEEDS) for f in FORMS} for t in TASKS} == r["task_mean_mse"]
verdict = "EARNS_RESOURCE_QUALIFICATION" if any(g["earns_full_model_resource_qualification"] for g in gates.values()) else "REJECTED_AT_THIS_FITTING_BUDGET" if positive else "INCONCLUSIVE_ASSAY_FAILURE"
assert verdict == r["scientific_verdict"]
assert r["optimizer_updates"] == 144000 and r["training_example_presentations"] == 36864000
assert r["post_training_reset_ablations"] == 48 and r["corpus_targets"] == r["full_model_resource_workers"] == 0
resources, dependencies = {}, {}
for f in FORMS:
    chosen = [records[selected[t, f, seed]["cell"]] for t in TASKS for seed in SEEDS]
    resources[f] = {"mean_cell_median_update_ms": statistics.mean(v["median_update_ms"] for v in chosen),
                    "max_peak_allocated_mib": max(v["peak_allocated_bytes"] for v in chosen)/2**20,
                    "mean_clip_fraction": statistics.mean(v["clip_fraction"] for v in chosen),
                    "upper_rate_selected": sum(v["base_rate"] == 0.003 for v in chosen)}
    if f.startswith("latent_"):
        dependencies[f] = {"reset_to_original_ratio": geom([resets[v["cell"]]["ratio"] for v in chosen]),
                           "per_task": {t: geom([resets[selected[t, f, seed]["cell"]]["ratio"] for seed in SEEDS]) for t in TASKS}}
prior(before)
assert all(sha(n) == h for n, h in p["sources"].items())
out = {"status": "PASS", "scientific_verdict": verdict, "research_goal_achieved": False,
       "checkpoint_records_verified": 240, "optimizer_updates_verified": 144000,
       "selection_rescores_exact": 240, "all_rate_reporting_rescores_exact": 240,
       "selected_reporting_rescores_exact": 120, "reset_ablations_exact": 48,
       "analysis_optimizer_updates": 0, "analysis_selection_examples": 983040,
       "analysis_reporting_and_reset_examples": 1179648,
       "additional_unselected_reporting_passes": 120,
       "additional_rescore_reason": "Both rates affect the mandatory matched-rate robustness gate.",
       "initial_states_reconstructed": 30, "rng_states_exact": 240, "fresh_data_regenerated_exact": True,
       "qualification_checks": 8, "qualification_exact_gradient_pairs": gradient_pairs,
       "protocol_sha256": sha(ROOT / "protocol.json"), "result_sha256": sha(ROOT / "result.json"),
       "source_archive_sha256": sha(ROOT / "source.zip"), "data_sha256": sha(ROOT / "data.pt"),
       "cell_artifact_hashes": artifacts,
       "selection_artifact_hashes": {n.name: sha(n) for n in (ROOT / "selections").glob("*.json")},
       "ratios": ratios, "fixed_rate_ratios": fixed, "gates": gates, "robustness": robust,
       "resources": resources, "reset_dependencies": dependencies, "scientific_retries": 0}
write_json(OUT, out)
print(json.dumps({k: v for k, v in out.items() if k not in
                  ("cell_artifact_hashes", "selection_artifact_hashes", "ratios", "fixed_rate_ratios")}), flush=True)
