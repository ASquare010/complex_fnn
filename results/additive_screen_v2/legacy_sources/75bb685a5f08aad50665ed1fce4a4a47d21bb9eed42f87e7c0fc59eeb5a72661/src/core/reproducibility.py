"""Record the actual working source, not merely a potentially stale git commit."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else 0,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda else False,
        "platform": platform.platform(),
    }


def provenance() -> dict:
    paths = sorted(
        [
            *Path("src").rglob("*.py"),
            *Path("tests").rglob("*.py"),
            *Path("configs").rglob("*.json"),
            Path("main.py"),
            Path("pyproject.toml"),
            Path("uv.lock"),
        ]
    )
    hashes = {p.as_posix(): sha256(p) for p in paths if p.exists()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit, dirty = None, "unavailable"
    return {
        "git_commit": commit,
        "git_status": dirty,
        "source_files": hashes,
        "source_hash": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }
