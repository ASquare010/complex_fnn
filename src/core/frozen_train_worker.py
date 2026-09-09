"""Train an explicitly qualified frozen cell without importing report/plot modules."""

import argparse
import hashlib
import json
from pathlib import Path


def read_cell(protocol_path, qualification_path, cell):
    protocol_bytes = protocol_path.read_bytes()
    qualification = json.loads(qualification_path.read_text())
    assert hashlib.sha256(protocol_bytes).hexdigest() == qualification["protocol_sha256"]
    protocol = json.loads(protocol_bytes)
    assert (
        hashlib.sha256(Path(qualification["plan_path"]).read_bytes()).hexdigest()
        == protocol["plan_sha256"]
        == qualification["plan_sha256"]
    )
    assert cell in qualification["cells"]
    for name, expected in qualification["critical_source_hashes"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    return protocol["configurations"][cell], Path(qualification["cells"][cell])


def worker(protocol_path, qualification_path, cell):
    raw, output = read_cell(protocol_path, qualification_path, cell)
    assert not output.exists(), f"Never overwrite a frozen cell: {output}"
    # Import only the original computation. No experiment/report driver imports.
    from src.core.config import ModelConfig, TrainConfig
    from src.core.trainer import train

    train(
        ModelConfig(**raw["model"]),
        TrainConfig(**raw["training"]),
        Path("data/wikitext2_v1"),
        output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    args = parser.parse_args()
    worker(args.protocol, args.qualification, args.cell)
