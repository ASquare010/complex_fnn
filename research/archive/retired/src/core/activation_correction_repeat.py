"""One qualified full-training repeat for correction-only rational compilation."""

import argparse
import json
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

from src.core.activation_screen import CACHE
from src.core.activation_training_repeat import configuration, source_check, verify
from src.core.reproducibility import environment, provenance, sha256, write_json
from src.core.trainer import train

CONFIG = Path("configs/wikitext2_blockshuffle_rational_correction_compiled.json")
RUN = Path("results/runs/wikitext2_blockshuffle_rational_correction_compiled_lr1200_s17_200")
PLAN = Path("research/activation_correction_repeat_plan.md")


def audit(output):
    assert json.loads(Path("results/activation_correction_execution_v1/result.json").read_text())[
        "earns_training_repeat"
    ]
    assert not RUN.exists()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "plan_sha256": sha256(PLAN),
        "config_sha256": sha256(CONFIG),
        "provenance": provenance(),
        "environment": environment(),
        "worker_timeout_seconds": 900,
        **source_check(),
    }
    write_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["provenance"]["source_files"]:
            archive.write(name, name)
        archive.write(PLAN, PLAN.as_posix())
    try:
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "src.core.activation_correction_repeat",
                    "worker",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=True,
            )
        result = verify(RUN, CONFIG)
        write_json(output / "result.json", result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write_json(
            output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "audit"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        mc, tc = configuration(CONFIG)
        train(mc, tc, CACHE, RUN)
    else:
        if args.output is None:
            parser.error("audit requires output")
        audit(args.output)
