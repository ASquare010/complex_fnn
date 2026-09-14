"""Verify archived evidence; add --apply to restore missing original files."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "research/reinstall_evidence_manifest.json").read_text())
    missing = 0
    for record in manifest["files"]:
        original = (root / record["path"]).resolve()
        archive = (root / record["archive"]).resolve()
        if not original.is_relative_to(root) or not archive.is_relative_to(root):
            raise ValueError("Manifest path leaves the repository")
        packed = archive.read_bytes()
        if hashlib.sha256(packed).hexdigest() != record["archive_sha256"]:
            raise ValueError(f"Archive hash mismatch: {archive}")
        raw = gzip.decompress(packed)
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Original hash mismatch: {original}")
        if original.exists():
            if original.read_bytes() != raw:
                raise ValueError(f"Refusing to overwrite changed evidence: {original}")
        else:
            missing += 1
            if args.apply:
                original.parent.mkdir(parents=True, exist_ok=True)
                with original.open("xb") as stream:
                    stream.write(raw)
    print(f"Verified {len(manifest['files'])} archives; {missing} originals "
          f"{'restored' if args.apply else 'missing (use --apply to restore)' }.")


if __name__ == "__main__":
    main()
