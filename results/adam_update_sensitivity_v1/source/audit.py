"""Independent NumPy FP64 reference; torch is used only to deserialize tensors."""

import itertools
import math

import numpy as np
import torch

from results.adam_update_sensitivity_v1.source.prepare import ROOT, hashes, read, sha
from src.core.reproducibility import write_json


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def arrays(tensors):
    return {k: v.numpy().astype(np.float64) for k, v in tensors.items()}


def norm2(x):
    return float(np.sum(np.square(x), dtype=np.float64))


def error(a, b):
    delta = sum(norm2(a[k] - b[k]) for k in a)
    scale_a = math.sqrt(sum(norm2(v) for v in a.values()))
    scale_b = math.sqrt(sum(norm2(v) for v in b.values()))
    maximum = max(float(np.max(np.abs(a[k] - b[k]))) for k in a)
    return dict(
        distance=math.sqrt(delta) / max((scale_a + scale_b) / 2, 1e-12), max_absolute=maximum
    )


def ideal(g, epsilon):
    return {k: np.divide(v, np.add(np.abs(v), epsilon)) for k, v in g.items()}


def close(a, b, protocol):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=protocol["metric_rtol"], abs_tol=protocol["metric_atol"]), (
        a,
        b,
    )


def step_audit(case, source, initial, protocol):
    for name in ("clipped", "updated"):
        assert sha(case[name]["path"]) == case[name]["sha256"]
    raw = arrays(load(source["probe"]["path"]))
    clipped = arrays(load(case["clipped"]["path"]))
    saved = load(case["updated"]["path"])
    actual = arrays(saved["model"])
    assert raw.keys() == clipped.keys() == actual.keys() == initial.keys()
    assert sum(v.size for v in raw.values()) == case["parameters"] == 9099648
    assert all(
        v.shape == clipped[k].shape == actual[k].shape == initial[k].shape for k, v in raw.items()
    )
    assert all(np.isfinite(v).all() for mapping in (raw, clipped, actual) for v in mapping.values())
    norm = math.sqrt(sum(norm2(v) for v in raw.values()))
    coefficient = min(1.0, 1.0 / (norm + 1e-6))
    clip_error = error(clipped, {k: v * coefficient for k, v in raw.items()})
    expected, actual_moments, expected_moments, covered = {}, [{}, {}], [{}, {}], []
    for group in case["groups"]:
        opts = group["options"]
        assert opts["lr"] == 0.0006 and opts["eps"] == 1e-8 and opts["betas"] == [0.9, 0.95]
        assert opts["foreach"] is None and opts["fused"] is None
        for name, index in zip(group["names"], group["ids"], strict=True):
            covered.append(name)
            decay = 0.1 if initial[name].ndim >= 2 else 0.0
            assert opts["weight_decay"] == decay
            g = clipped[name]
            expected[name] = initial[name] * (1 - opts["lr"] * decay) - opts["lr"] * g / (
                np.abs(g) + opts["eps"]
            )
            state = saved["optimizer"]["state"][index]
            assert state["step"].item() == 1
            for i, (field, beta) in enumerate(
                zip(("exp_avg", "exp_avg_sq"), opts["betas"], strict=True)
            ):
                actual_moments[i][name] = state[field].numpy().astype(np.float64)
                assert np.isfinite(actual_moments[i][name]).all()
                expected_moments[i][name] = (1 - beta) * (g if i == 0 else np.square(g))
    assert len(covered) == len(set(covered)) == len(initial)
    assert set(covered) == initial.keys()
    p_error = error(actual, expected)
    moment_errors = [error(a, b) for a, b in zip(actual_moments, expected_moments, strict=True)]
    update_error = error(
        {k: initial[k] - actual[k] for k in initial}, {k: initial[k] - expected[k] for k in initial}
    )
    passed = (
        p_error["distance"] <= protocol["parameter_relative_tolerance"]
        and p_error["max_absolute"] <= protocol["parameter_absolute_tolerance"]
        and clip_error["distance"] <= protocol["clip_relative_tolerance"]
        and all(e["distance"] <= 1e-6 for e in moment_errors)
    )
    return dict(
        label=case["label"],
        parameter_error=p_error,
        clipped_error=clip_error,
        moment_errors=moment_errors,
        realized_update_error=update_error,
        raw_norm_fp64=norm,
        clip_coefficient_fp64=coefficient,
        passed=passed,
    )


def pair_audit(a, b, epsilon):
    da, db = ideal(a, epsilon), ideal(b, epsilon)
    counts = np.zeros(4, dtype=np.int64)
    errors = np.zeros(4, dtype=np.float64)
    signs = 0
    for key in a:
        magnitude = np.maximum(np.abs(a[key]), np.abs(b[key]))
        masks = [
            magnitude <= epsilon,
            (magnitude > epsilon) & (magnitude <= 10 * epsilon),
            (magnitude > 10 * epsilon) & (magnitude <= 100 * epsilon),
            magnitude > 100 * epsilon,
        ]
        delta = np.square(da[key] - db[key])
        signs += int(
            np.count_nonzero(((a[key] < 0) & (b[key] > 0)) | ((a[key] > 0) & (b[key] < 0)))
        )
        for i, mask in enumerate(masks):
            counts[i] += np.count_nonzero(mask)
            errors[i] += np.sum(delta, where=mask, dtype=np.float64)
    total = float(np.sum(errors))
    d, g = error(da, db)["distance"], error(a, b)["distance"]
    return dict(
        direction_distance=d,
        clipped_distance=g,
        amplification=d / g if g else None,
        bin_counts=counts.tolist(),
        bin_squared_errors=errors.tolist(),
        bin_error_shares=(errors / total).tolist() if total else [0.0] * 4,
        near_zero_error_share=float(np.sum(errors[:2]) / total) if total else 0.0,
        opposite_nonzero_signs=signs,
    )


def run():
    assert not (ROOT / "audit.json").exists() and not torch.cuda.is_initialized()
    p, result, analysis = [
        read(ROOT / n) for n in ("protocol.json", "result.json", "analysis.json")
    ]
    for mapping in [
        p[k] for k in ("sources", "maintained_files", "input_hashes", "library_hashes")
    ]:
        hashes(mapping)
    hashes(read(ROOT / "audit_protocol.json")["files"])
    initial = arrays(load(p["initial_path"])["model"])
    rows = [
        step_audit(c, next(s for s in p["cases"] if s["label"] == c["input_label"]), initial, p)
        for c in result["cases"]
    ]
    gpu = [c for c in result["cases"] if c["device"] == "cuda"]
    clips = {c["input_label"]: arrays(load(c["clipped"]["path"])) for c in gpu}
    independently_strong, alternative_ratios = True, {1e-7: [], 1e-6: []}
    for item in analysis["pairs"]:
        left, right, eps = item["left"], item["right"], item["epsilon"]
        checked = pair_audit(clips[left], clips[right], eps)
        for key, value in checked.items():
            if key in ("bin_counts", "opposite_nonzero_signs") or value is None:
                assert item[key] == value
            elif isinstance(value, list):
                for x, y in zip(value, item[key], strict=True):
                    close(x, y, p)
            else:
                close(value, item[key], p)
        if not item["same_policy"]:
            if eps == 1e-8:
                independently_strong &= (
                    checked["amplification"] is not None
                    and checked["amplification"] >= 10
                    and checked["near_zero_error_share"] >= 0.5
                )
            else:
                base = pair_audit(clips[left], clips[right], 1e-8)
                alternative_ratios[eps].append(
                    checked["direction_distance"] / base["direction_distance"]
                )
    checked_distortions = {1e-7: [], 1e-6: []}
    for item in analysis["distortions"]:
        c, eps = clips[item["label"]], item["epsilon"]
        d = error(ideal(c, eps), ideal(c, 1e-8))["distance"]
        close(d, item["distance"], p)
        checked_distortions[eps].append(d)
    assert analysis["strong_near_zero_amplification"] == independently_strong
    for item in analysis["conditioning"]:
        eps = item["epsilon"]
        ratio, distortion = max(alternative_ratios[eps]), max(checked_distortions[eps])
        close(ratio, item["max_error_ratio"], p)
        close(distortion, item["max_direction_distortion"], p)
        assert item["passed"] == (ratio <= 0.5 and distortion <= 0.01)
    params = {c["input_label"]: arrays(load(c["updated"]["path"])["model"]) for c in gpu}
    raw = {c["label"]: arrays(load(c["probe"]["path"])) for c in p["cases"]}
    updates = {
        label: {k: initial[k] - v for k, v in state.items()} for label, state in params.items()
    }
    for item in analysis["native_pairs"]:
        a, b = item["left"], item["right"]
        for key, mapping in (
            ("raw_gradient_distance", raw),
            ("parameter_distance", params),
            ("realized_update_distance", updates),
        ):
            close(error(mapping[a], mapping[b])["distance"], item[key], p)
    for item in analysis["device_pairs"]:
        label = item["label"]
        c = next(c for c in result["cases"] if c["device"] == "cpu" and c["input_label"] == label)
        cp = arrays(load(c["updated"]["path"])["model"])
        cu = {k: initial[k] - v for k, v in cp.items()}
        close(error(cp, params[label])["distance"], item["parameter_distance"], p)
        close(error(cu, updates[label])["distance"], item["realized_update_distance"], p)
        close(
            error(arrays(load(c["clipped"]["path"])), clips[label])["distance"],
            item["clipped_distance"],
            p,
        )
    for eps in p["epsilons"]:
        assert ideal({"g": np.array([0.0, eps])}, eps)["g"].tolist() == [0.0, 0.5]
        a, b = 2 * eps, 3 * eps
        assert math.isclose(
            abs(a / (a + eps) - b / (b + eps)),
            eps * abs(a - b) / ((a + eps) * (b + eps)),
            rel_tol=1e-12,
        )
    assert len(result["boundaries"]) == 7
    assert all(b == dict(allocated=0, reserved=0) for b in result["boundaries"])
    assert len(analysis["pairs"]) == 45 and len(analysis["distortions"]) == 12
    assert set((x["left"], x["right"]) for x in analysis["pairs"]) == set(
        itertools.combinations([c["input_label"] for c in gpu], 2)
    )
    assert not torch.cuda.is_initialized()
    passed = all(r["passed"] for r in rows)
    write_json(
        ROOT / "audit.json",
        dict(
            passed=passed,
            rows=rows,
            tensor_artifacts_verified=24,
            direction_pairs_checked=45,
            distortions_checked=12,
            native_pairs_checked=15,
            device_pairs_checked=6,
            optimizer_steps=0,
            forwards=0,
            backwards=0,
            cuda_initialized=False,
            broad_goal_achieved=False,
        ),
    )
    print("Independent NumPy audit passed:", passed, flush=True)


if __name__ == "__main__":
    run()
