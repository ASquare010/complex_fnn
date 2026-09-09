from pathlib import Path

old = Path("research/archive/retired/src/core/overcomplete_screen.py").read_text()
s = old.replace('"""H056 bounded quality screen; computation uses the existing frozen worker."""', '"""H061 additive language screen using preserved controls and the frozen trainer."""')
s = s.replace('import math\n', 'import math\nimport subprocess\nimport sys\nimport time\n')
s = s.replace('results/overcomplete_screen_v1', 'results/additive_screen_v1')
s = s.replace('research/overcomplete_screen_plan.md', 'research/additive_block_lowrank_screen_plan.md')
s = s.replace('RECIPE = "overcomplete_headwise"', 'RECIPE = "additive_block_lowrank"')
s = s.replace('("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle", "headwise")', '("full_swiglu", "full_gelu", "calibrated_narrow", "blockshuffle")')
s = s.replace('    from src.core.wikitext_screen import promotion\n\n', '')
s = s.replace('    from src.core.wikitext_screen import select_best\n', '')
helper = '''
def select_best(rows):
    if not rows or any(not math.isfinite(row["validation_loss"]) for row in rows):
        raise ValueError("Selection requires finite completed trials")
    return min(rows, key=lambda row: (row["validation_loss"], row["training"]["learning_rate"]))


def promotion(selected):
    candidate = selected["blockshuffle"]
    loss = candidate["validation_loss"]
    gates = {
        "at_least_70_percent_fewer_ffn_weights": candidate["ffn_reduction_percent"] >= 70,
        "beats_calibrated_narrow": loss < selected["calibrated_narrow"]["validation_loss"],
    }
    for recipe in ("full_swiglu", "full_gelu"):
        gates[f"within_one_percent_{recipe}"] = loss <= 1.01 * selected[recipe]["validation_loss"]
        gates[f"memory_within_ten_percent_{recipe}"] = candidate["peak_allocated_vram_bytes"] <= 1.1 * selected[recipe]["peak_allocated_vram_bytes"]
    return gates


def nodes(source, kind):
    return {node.name: ast.dump(node) for node in ast.parse(source).body if isinstance(node, kind)}


def run_reference_probe(request, output, cwd, label):
    command = [sys.executable, "-X", "faulthandler", str(Path("results/verification/additive_control_probe_v1.py").resolve()), str(request.resolve()), str(output.resolve())]
    log_path = ROOT / f"reference_{label}.log"
    record_path = ROOT / f"reference_{label}.json"
    assert not log_path.exists() and not record_path.exists()
    write(record_path, {"status": "RUNNING", "command": command, "cwd": str(cwd)})
    start = time.perf_counter()
    with log_path.open("w") as handle:
        process = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    write(record_path, {"status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode, "seconds": time.perf_counter() - start, "command": command, "cwd": str(cwd), "log_sha256": sha(log_path)})
    print(json.dumps({"reference_probe": label, "returncode": process.returncode}), flush=True)
    assert process.returncode == 0, log_path.read_text()

'''
s = s.replace('\ndef decision(candidate, references):', helper+'\ndef decision(candidate, references):')
a=s.index('        for n in (\n',s.index('def archive_check'))
b=s.index('        # Compare actual training updates',a)
s=s[:a]+'''        assert archive.read("src/core/data.py") == Path("src/core/data.py").read_bytes()
        for old_path, new_path, names in (
            ("src/dense_ffn/__init__.py", "src/dense_ffn/__init__.py", ("DenseFFN",)),
            ("src/grouped_ffn/__init__.py", "src/core/structured_linear.py", ("GroupedLinear",)),
            ("src/blockshuffle_ffn/__init__.py", "src/blockshuffle_ffn/__init__.py", ("BlockShuffleLinear", "BlockShuffleFFN", "RecomputedSwiGLU")),
        ):
            left = nodes(archive.read(old_path), ast.ClassDef)
            right = nodes(Path(new_path).read_bytes(), ast.ClassDef)
            assert all(left[name] == right[name] for name in names), old_path
        left = nodes(archive.read("src/core/trainer.py"), ast.FunctionDef)
        right = nodes(Path("src/core/trainer.py").read_bytes(), ast.FunctionDef)
        assert all(left[name] == right[name] for name in ("learning_rate", "resolve_precision"))
'''+s[b:]
start=s.index('def preflight():')
end=s.index('\n\ndef finish():',start)
preflight='''def preflight():
    import torch

    from src.core.config import ModelConfig, TrainConfig
    from src.core.data import TokenData, load_manifest
    from src.core.optimization import group_summary, initialize_dense_width, parameter_groups
    from src.core.reproducibility import environment, provenance
    from src.core.transformer import Transformer

    assert sha(PLAN) == "86d3833d3d7fc99e976baf227893193df6715668d7e5ea3a4b2edc5664904763"
    assert not ROOT.exists()
    torch.set_num_threads(4)
    final = read("results/verification/additive_qualification_final_v1.json")
    assert final["status"] == "PASS" and final["quality_screen_earned"]
    assert sha("results/additive_qualification_v1/result.json") == final["result_sha256"]
    assert sha("results/additive_qualification_v1/source.zip") == final["source_archive_sha256"]
    tested = read("results/additive_qualification_v1/process_integration_tests.json")
    assert tested["returncode"] == 0 and tested["source_unchanged"]
    assert all(sha(name) == digest for name, digest in tested["source_hashes"].items())
    prior = read("results/overcomplete_screen_v1/preflight.json")
    rows = [row for row in prior["controls"] if row["recipe"] in CONTROLS]
    assert len(rows) == 12
    assert {(row["recipe"], row["rate"]) for row in rows} == {(recipe, rate) for recipe in CONTROLS for rate in RATES}
    manifest = load_manifest(CACHE)
    assert manifest["files"] == prior["data_hashes"]
    data = TokenData(CACHE, "cpu", 10017)
    batches = list(data.validation(16, 128, 158))
    assert len(batches) == 158 and batches[-1][0].shape == (9, 128)
    assert torch.equal(torch.cat([y.flatten() for _, y in batches]), data.valid[1:322689])
    ROOT.mkdir()
    groups, common, snapshots = {}, None, {}
    for row in rows:
        path = Path("results/runs") / row["run"]
        assert sha(path / "metrics.json") == row["metrics_sha256"]
        assert sha(path / "checkpoint.pt") == row["checkpoint_sha256"]
        assert sha(path / "source.zip") == row["source_archive_sha256"]
        metrics = read(path / "metrics.json")
        mc, tc = ModelConfig(**metrics["model"]), TrainConfig(**metrics["training"])
        mc.validate()
        tc.validate()
        assert (mc.width, mc.layers, mc.heads, mc.context, mc.vocab_size) == (384, 8, 6, 128, 4096)
        assert mc.ffn_width == {"full_swiglu": 1024, "full_gelu": 1536, "calibrated_narrow": 304, "blockshuffle": 2048}[row["recipe"]]
        assert (tc.steps, tc.batch_size, tc.eval_batches, tc.log_every, tc.seed, tc.precision) == (200, 16, 158, 50, 17, "bf16")
        assert tc.learning_rate == row["rate"]
        assert metrics["data"]["files"] == manifest["files"] and metrics["environment"] == environment()
        assert metrics["training_tokens"] == 409600 and metrics["validation_tokens"] == 322688
        archive_check(path, metrics, compare_old=True)
        checkpoint_check(path, metrics, prior["sampling_rng_sha256"])
        model = Transformer(mc, 17)
        initialize_dense_width(model, tc)
        actual = group_summary(parameter_groups(model, tc))
        assert actual == metrics["optimizer_parameter_groups"]
        groups[row["run"]] = actual
        shared = {name: sha_tensor(p) for name, p in model.named_parameters() if ".ffn." not in name}
        if common is None:
            common = shared
        assert shared == common
        key = metrics["provenance"]["source_hash"]
        entry = snapshots.setdefault(key, {"path": path, "metrics": metrics, "controls": []})
        entry["controls"].append({"run": row["run"], "model": metrics["model"], "training": metrics["training"]})
        del model

    # Execute original src trees in fresh interpreters, never importing them into the live core.
    compatibility = {}
    for index, (key, snapshot) in enumerate(snapshots.items()):
        stage = (ROOT / "legacy_sources" / key).resolve()
        stage.mkdir(parents=True)
        with zipfile.ZipFile(snapshot["path"] / "source.zip") as archive:
            for name in snapshot["metrics"]["provenance"]["source_files"]:
                assert (stage / name).resolve().is_relative_to(stage)
                archive.extract(name, stage)
        request = ROOT / f"reference_request_{index}.json"
        write(request, {"train_path": str((CACHE / "train.npy").resolve()), "controls": snapshot["controls"]})
        old_output, new_output = ROOT / f"reference_old_{index}.json", ROOT / f"reference_current_{index}.json"
        run_reference_probe(request, old_output, stage, f"old_{index}")
        run_reference_probe(request, new_output, Path.cwd(), f"current_{index}")
        assert read(old_output) == read(new_output), f"Full-size compatibility mismatch: {key}"
        compatibility[key] = {"controls": [cell["run"] for cell in snapshot["controls"]], "original_result_sha256": sha(old_output), "current_result_sha256": sha(new_output)}

    gpu_data = TokenData(CACHE, "cuda", 10017)
    for _ in range(200):
        gpu_data.batch(16, 128)
    sampler = sha_tensor(gpu_data.generator.get_state())
    assert sampler == prior["sampling_rng_sha256"]
    del gpu_data
    torch.cuda.empty_cache()
    mc, tc = configuration(RATES[0])
    model = Transformer(mc, 17)
    assert mc.unique_ffn_parameters == 2801664 and mc.total_parameters == 9099648
    assert {name: sha_tensor(p) for name, p in model.named_parameters() if ".ffn." not in name} == common
    actual = parameter_groups(model, tc)
    scales = {id(p): group["lr_scale"] for group in actual for p in group["params"]}
    named_scales = {name: scales[id(p)] for name, p in model.named_parameters()}
    assert len(scales) == sum(len(group["params"]) for group in actual) == len(list(model.parameters()))
    configuration_records, cells = {}, {}
    for rate in RATES:
        c, t = configuration(rate)
        path = output_path(rate)
        assert not path.exists()
        configuration_records[path.name] = {"model": asdict(c), "training": asdict(t)}
        cells[path.name] = path.as_posix()
    prov = provenance()
    # The reference helper is part of preflight evidence even though the training worker does not import it.
    record = {"status": "PASS", "h060_existing_sources_unchanged": True, "retained_exact_variant_count": 6, "full_size_control_probe_count": 12, "full_size_control_source_snapshots": compatibility, "reference_helper_sha256": sha("results/verification/additive_control_probe_v1.py"), "controls": rows, "data_hashes": manifest["files"], "validation_targets": 322688, "validation_stream_exact": True, "sampling_rng_sha256": sampler, "environment": environment(), "common_non_ffn_initial_weights_exact": True, "control_optimizer_groups": groups, "candidate_optimizer_groups": group_summary(actual), "candidate_lr_scales": named_scales}
    protocol = {"plan_sha256": sha(PLAN), "configurations": configuration_records, "provenance": prov, "new_training_tokens": 1228800, "rates": RATES}
    write(ROOT / "preflight.json", record)
    write(ROOT / "protocol.json", protocol)
    with zipfile.ZipFile(ROOT / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in prov["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
        archive.write("results/verification/additive_control_probe_v1.py", "results/verification/additive_control_probe_v1.py")
    write(ROOT / "worker_qualification.json", {"status": "PASS", "protocol_sha256": sha(ROOT / "protocol.json"), "plan_path": PLAN.as_posix(), "plan_sha256": sha(PLAN), "critical_source_hashes": prov["source_files"], "cells": cells})
    print(json.dumps({"status": "PASS", "retained_controls_verified": len(rows), "full_size_control_probes_exact": 12, "candidate_optimizer_groups": record["candidate_optimizer_groups"]}), flush=True)


def sha_tensor(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
'''
s=s[:start]+preflight+s[end:]
Path('src/core/additive_screen.py').write_text(s,encoding='utf-8')
print('Prepared H061 driver with original control gates, isolated source probes and unchanged worker.')