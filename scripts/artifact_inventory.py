"""Inventory local scientific evidence without copying it into Git."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def inventory(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts, total = 0, 0
    started = time.perf_counter()
    with destination.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "bytes", "sha256"])
        for root in (Path("results"), Path("data"), Path("research/archive")):
            for base, directories, names in os.walk(root):
                directories[:] = sorted(d for d in directories if d != "__pycache__")
                for name in sorted(names):
                    path = Path(base) / name
                    if path.suffix in (".pyc", ".pyo"):
                        continue
                    before = path.stat()
                    with path.open("rb") as artifact:
                        digest = hashlib.file_digest(artifact, "sha256").hexdigest()
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError(f"Artifact changed during inventory: {path}")
                    writer.writerow([path.as_posix(), before.st_size, digest])
                    counts += 1
                    total += before.st_size
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"files": counts, "bytes": total, "seconds": time.perf_counter() - started,
                      "manifest": destination.as_posix()}, indent=2))


def check_index() -> None:
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    files = [Path(path) for path in paths if path]
    prohibited = {".pt", ".pth", ".safetensors", ".pickle", ".npy", ".npz", ".pyc"}
    problems = []
    total = 0
    largest = []
    for path in files:
        size = path.stat().st_size
        total += size
        largest.append((size, path.as_posix()))
        if path.suffix in prohibited or path.parts[0] in ("data", ".venv", ".cache"):
            problems.append(f"Generated/binary artifact in index: {path}")
        if path.suffix == ".zip" and not path.as_posix().startswith("research/archive/"):
            problems.append(f"Repeated source archive in index: {path}")
        if size > 5 * 1024**2:
            problems.append(f"File exceeds 5 MiB review limit: {path}")
    if total > 48 * 1024**2:
        problems.append(f"Tracked bytes exceed 48 MiB compact release budget: {total}")
    print(json.dumps({"files": len(files), "bytes": total, "largest": sorted(largest, reverse=True)[:8],
                      "problems": problems}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-index", action="store_true")
    args = parser.parse_args()
    if args.check_index:
        check_index()
    elif args.output:
        inventory(args.output)
    else:
        parser.error("Choose --output PATH or --check-index")
