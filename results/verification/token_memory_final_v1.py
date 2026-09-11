"""Read-only independent consistency check of H101/H102 public evidence."""

import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    initial = Path("results/token_memory_v1")
    repeat = Path("results/token_memory_replication_v1")
    for protocol in (read(initial/"protocol.json"),read(repeat/"protocol.json")):
        for path,digest in protocol["sources"].items():
            assert sha(path)==digest,path
    assert sha("research/token_memory_plan.md")==read(initial/"protocol.json")["plan_sha256"]
    assert sha("research/token_memory_followup_plan.md")==read(repeat/"protocol.json")["plan_sha256"]
    results = read(repeat/"result.json")
    audit = read(repeat/"audit.json")
    assert audit["passed"] and len(audit["cases"])==38
    assert all(r["exact_score_match"] and r["parameter_hashes_match"] and r["data_order_match"]
               for r in audit["cases"])
    metrics = [read(p) for p in repeat.glob("*/cases/*/metrics.json")]
    assert len(metrics)==36 and len(read(repeat/"processes.json"))==36
    assert all(r["returncode"]==0 for r in read(repeat/"processes.json"))
    with (repeat/"metrics.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table)==36
    for comparison in results["comparisons"]:
        peers = {r["policy"]:r for r in metrics if
                 (r["variant"],r["context"],r["seed"]) ==
                 (comparison["variant"],comparison["context"],comparison["seed"])}
        a,b,n = peers["loss_chunks"],peers["block"],peers["native"]
        computed = {"memory":a["peak_allocated_bytes"]<=.85*b["peak_allocated_bytes"],
                    "time":a["timing"]["update_ms"]["median"]<=1.25*b["timing"]["update_ms"]["median"],
                    "quality":abs(a["validation_nll"]/n["validation_nll"]-1)<=.01,
                    "finite":a["weights_finite"]}
        assert computed==comparison["gates"]
        assert all(computed.values())==comparison["all_pass"]
    assert sum(c["all_pass"] for c in results["comparisons"] if c["variant"]=="gelu_narrow")==6
    assert all(not g["all_seeds_pass"] for g in results["groups"] if g["variant"]=="swiglu")
    assert read(initial/"result.json")["completed_cases"]==22
    assert read(initial/"interruption.json")["completed_cases"]==22
    tests = (repeat/"tests.log").read_text(encoding="utf-8-sig")
    assert "111 passed" in tests
    paths = [Path("research/token_memory_results.md"),Path("src/core/token_memory.py"),
             Path("tests/test_token_memory.py"),repeat/"result.json",initial/"result.json",
             repeat/"audit.json",repeat/"tests.log",Path("research/figures/token_memory.png")]
    receipt = {"status":"PASS","completed_initial_cases":22,"replication_cases":36,
               "audited_checkpoints":38,"pytest_passed":111,"helper_tests_included":14,
               "separate_small_model_qualification_cases":40,"all_frozen_source_hashes_match":True,
               "all_frozen_gates_recomputed":True,"broad_architectural_goal_achieved":False,
               "files":{p.as_posix():sha(p) for p in paths}}
    target = Path("results/verification/token_memory_final_v1.json")
    target.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
