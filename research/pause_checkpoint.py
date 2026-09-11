"""Preserve selected completed models and exact pause progress; no training."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path("results/checkpoint_fp16_long_v1")
out = Path("src/experimental/checkpoints")


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


selected = []
prior = read("results/batch_scale_long_v1/protocol.json")
f = next(
    r
    for r in prior["schedule"]
    if r["arm"] == "buffer4"
    and r["fixture"]["dataset"] == "wikitext2"
    and r["fixture"]["seed"] == 101
)
c = read(Path("results/batch_scale_long_v1") / f"case{f['index']:02d}" / "case.json")["measurement"]
selected.append(
    (
        "exact_offload_wikitext2_s101_step800",
        c["checkpoint"]["path"],
        "H156 qualified exact helper",
        c["checkpoint"]["sha256"],
    )
)
completed = []
for folder in sorted(root.glob("case[0-9][0-9]")):
    if not (folder / "case.json").exists():
        continue
    row = read(folder / "case.json")
    completed.append(
        dict(
            index=row["index"],
            arm=row["arm"],
            dataset=row["fixture"]["dataset"],
            seed=row["fixture"]["seed"],
            updates=800,
        )
    )
    if row["arm"] == "fp16":
        c = row["measurement"]
        selected.append(
            (
                f"fp16_{row['fixture']['dataset']}_s{row['fixture']['seed']}_step800",
                c["checkpoint"]["path"],
                "H160 completed run; independent long-run audit pending",
                c["checkpoint"]["sha256"],
            )
        )
manifest = []
for name, source, status, digest in selected:
    assert sha(source) == digest
    dest = out / (name + ".pt")
    assert not dest.exists()
    shutil.copyfile(source, dest)
    assert sha(dest) == digest
    manifest.append(
        dict(
            name=name,
            path=dest.as_posix(),
            source=source,
            sha256=digest,
            status=status,
            git_tracked=False,
            contains="model, optimizer, sampler and configuration",
        )
    )
partial = []
for folder in sorted(root.glob("case[0-9][0-9]")):
    if (folder / "case.json").exists():
        continue
    history = list(folder.rglob("history.jsonl"))
    if not history:
        continue
    records = []
    for line in history[0].read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    checkpoints = [dict(path=p.as_posix(), sha256=sha(p)) for p in sorted(folder.rglob("step*.pt"))]
    partial.append(
        dict(
            case=folder.name,
            last_recorded_update=records[-1]["step"] if records else 0,
            checkpoints=checkpoints,
            interrupted=True,
        )
    )
pause = dict(
    status="PAUSED_BY_USER",
    study="H160",
    completed_runs=completed,
    partial_runs=partial,
    completed_run_updates=800 * len(completed),
    recorded_partial_updates=sum(v["last_recorded_update"] for v in partial),
    restart_automatically=False,
    checkpoint_manifest="src/experimental/checkpoints/manifest.json",
)
(root / "pause.json").write_text(json.dumps(pause, indent=2) + "\n")
(out / "manifest.json").write_text(json.dumps(dict(models=manifest, pause=pause), indent=2) + "\n")
assert (
    Path("src/experimental/checkpoint_compression.py").read_bytes()
    == Path("results/checkpoint_fp16_v1/codec.py").read_bytes()
)
print(json.dumps(dict(models=len(manifest), completed_runs=len(completed), partial=partial)))
