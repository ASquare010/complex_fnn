"""Verify H113 and freeze the H114 protocol before scientific execution."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path("results/fp32_classifier_profile_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


assert not (ROOT / "protocol.json").exists()
prior_path = Path("results/verification/classifier_precision_final_v1.json")
receipt = read(prior_path)
assert receipt["status"] == "PASS"
for name, digest in receipt["files"].items():
    assert sha(name) == digest, name
for name, digest in receipt["tensor_hashes"].items():
    assert sha(name) == digest, name
old = read("results/classifier_precision_v1/protocol.json")
for name, digest in old["sources"].items():
    assert sha(name) == digest, name
for dataset in old["datasets"].values():
    path = Path(dataset["path"])
    assert sha(path / "manifest.json") == dataset["manifest_sha256"]
    for name, digest in dataset["manifest"]["files"].items():
        assert sha(path / name) == digest
fixtures, checkpoint_hashes, extra_provenance = [], {}, {}
for row in read("results/whole_job_memory_workspace_v1/result.json")["cases"]:
    state = next(c for c in row["checkpoints"] if c["step"] == 800)
    assert sha(state["path"]) == state["sha256"]
    checkpoint_hashes[state["path"]] = state["sha256"]
    fixtures.append(
        dict(
            label=f"{row['dataset']}_s{row['seed']}_{row['policy']}",
            scope=f"narrow_{row['dataset']}",
            dataset=row["dataset"],
            batch=8,
            context=512,
            seed=row["seed"],
            source_policy=row["policy"],
            source_step=800,
            checkpoint=state["path"],
            model_config=row["model_config"],
        )
    )
for variant in ("gelu", "swiglu"):
    folder = Path(f"results/ungated_duration_v1/runs/full_{variant}_seed17")
    qualification, config = read(folder / "qualification.json"), read(folder / "config.json")
    path = folder / "checkpoint.pt"
    assert qualification["status"] == "PASS" and sha(path) == qualification["checkpoint_sha256"]
    checkpoint_hashes[path.as_posix()] = sha(path)
    for name in ("qualification.json", "config.json", "metrics.json", "source.zip"):
        extra_provenance[(folder / name).as_posix()] = sha(folder / name)
    fixtures.append(
        dict(
            label=f"full_{variant}_s17",
            scope=f"full_{variant}",
            dataset="wikitext2",
            batch=16,
            context=128,
            seed=17,
            source_policy="block",
            source_step=3200,
            checkpoint=path.as_posix(),
            model_config=config["model"],
        )
    )
assert len(fixtures) == len(checkpoint_hashes) == 14
for index, fixture in enumerate(fixtures):
    policies = ["block_bf16", "chunks_bf16", "block_fp32", "chunks_fp32"]
    policies = policies[index % 4 :] + policies[: index % 4]
    fixture["policy_order"] = list(reversed(policies)) if (index // 4) % 2 else policies
before = ROOT / "before_documents"
before.mkdir(exist_ok=False)
documents = {}
for name in (*old["before_documents"], ".gitignore"):
    target = before / (name.replace("/", "__") if name != ".gitignore" else "gitignore")
    target.write_bytes(Path(name).read_bytes())
    documents[name] = dict(sha256=sha(name), preserved_path=target.as_posix())
new = [*ROOT.glob("source/*.py"), Path("research/fp32_classifier_profile_plan.md")]
sources = {**old["sources"], **{p.as_posix(): sha(p) for p in new}}
free = shutil.disk_usage(Path.cwd()).free
assert free > 15 * 2**30, free
result = dict(
    study="H114",
    sources=sources,
    maintained_files=old["maintained_files"],
    datasets=old["datasets"],
    fixtures=fixtures,
    checkpoint_hashes=checkpoint_hashes,
    extra_source_provenance=extra_provenance,
    previous_receipt_sha256=sha(prior_path),
    before_documents=documents,
    cases=56,
    optimizer_updates=2800,
    profile_backward_passes=2856,
    warmup=20,
    measured_updates=30,
    steps=50,
    learning_rate=0.0006,
    chunk_size=512,
    free_disk_bytes=free,
    previous_goal_turn="progress",
    broad_goal_achieved=False,
)
(ROOT / "protocol.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
(ROOT / "runs").mkdir(exist_ok=False)
print(
    f"Frozen {len(sources)} sources, {len(fixtures)} starting states and 56 cases; prior receipt verified."
)
