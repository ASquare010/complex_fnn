"""Fresh sequential processes for the frozen H102 subset; never overwrite a run."""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("results/token_memory_replication_v1")
PLAN = Path("research/token_memory_followup_plan.md")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=False)
    original = json.loads(Path("results/token_memory_v1/protocol.json").read_text())
    for name, digest in original["sources"].items():
        assert sha(name) == digest, f"Frozen H101 source changed: {name}"
    sources = {p.as_posix():sha(p) for p in Path("results/token_memory_v1/source").glob("*.py")}
    protocol = {"started_utc":datetime.now(timezone.utc).isoformat(),
                "plan_sha256":sha(PLAN),"sources":sources,
                "h101_protocol_sha256":sha("results/token_memory_v1/protocol.json"),
                "variants":["gelu_narrow","swiglu"],"seeds":[17,29,43],
                "contexts":[128,512],"policies":["native","block","loss_chunks"]}
    (ROOT/"protocol.json").write_text(json.dumps(protocol,indent=2))
    start = time.perf_counter()
    receipts = []
    for context in (128,512):
        for i, seed in enumerate((17,29,43)):
            policies = protocol["policies"][i:]+protocol["policies"][:i]
            for variant in protocol["variants"]:
                for policy in policies:
                    key = f"{variant}_t{context}_s{seed}_{policy}"
                    command = [sys.executable,"-X","faulthandler","-m",
                               "results.token_memory_v1.source.diagnose",policy,
                               "--seed",str(seed),"--context",str(context),
                               "--variant",variant,"--root",str(ROOT/key)]
                    with (ROOT/f"{key}.log").open("x") as log:
                        process = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                    receipt = {"key":key,"command":command,"returncode":process.returncode}
                    receipts.append(receipt)
                    (ROOT/"processes.json").write_text(json.dumps(receipts,indent=2))
                    if process.returncode:
                        raise RuntimeError(f"Worker failed; preserved {key}.log")
                    print((ROOT/f"{key}.log").read_text().strip(),flush=True)
    (ROOT/"completion.json").write_text(json.dumps({"status":"COMPLETE","cases":len(receipts),
               "elapsed_seconds":time.perf_counter()-start},indent=2))
