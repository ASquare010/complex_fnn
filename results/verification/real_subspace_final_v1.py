"""Check H105 public accounting against independently reconstructed evidence."""

import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT=Path("results/real_subspace_v1")


def read(path):
    data=Path(path).read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe",b"\xfe\xff")) else "utf-8-sig")


def js(path):
    return json.loads(read(path))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    sources=js(ROOT/"protocol.json")["sources"]
    for name,digest in sources.items():
        assert sha(name)==digest,name
    assert js(ROOT/"qualification.json")["passed"]
    assert len(js(ROOT/"qualification.json")["checks"])==19
    result,summary,audit= (js(ROOT/name) for name in ("result.json","summary.json","audit.json"))
    assert result["status"]=="COMPLETE" and len(result["rows"])==378
    assert audit["passed"] and audit["score_count"]==378
    assert len(audit["recaptured_pairs"])==18 and all(r["all_rows_exact"] for r in audit["recaptured_pairs"])
    assert len(result["collections"])==6 and result["neural_updates"]==0
    for collection in result["collections"]:
        ids=collection["window_ids"]
        assert len(ids)==len(set(ids))==320
        assert set(ids[:256]).isdisjoint(ids[256:])
    by_key={tuple(r[k] for k in ("teacher","layer","seed","rank","method")):r for r in result["rows"]}
    for checked in audit["rows"]:
        row=by_key[tuple(checked[k] for k in ("teacher","layer","seed","rank","method"))]
        assert checked["passed"] and abs(checked["independent_mse"]-row["mse"])<=1e-7+1e-5*row["mse"]
    rank=js(ROOT/"rank_audit.json")
    assert rank["passed"] and rank["cases"]==54 and rank["rank64_all_above_five_percent"]
    for activation,methods in summary["groups"].items():
        for method,ranks in methods.items():
            for rank_text,metrics in ranks.items():
                chosen=[r for r in result["rows"] if (r["teacher"],r["method"],str(r["rank"]))==(activation,method,rank_text)]
                assert len(chosen)==9
                for key,record in metrics.items():
                    values=[r[key] for r in chosen]
                    assert statistics.mean(values)==record["mean"]
                    assert statistics.median(values)==record["median"]
                    assert statistics.variance(values)==record["variance"]
    for method,decision in summary["decisions"].items():
        assert decision["verdict"]=="REJECTED_FIXED_PROJECTION_RECIPE"
        for activation,judgement in decision["teachers"].items():
            rows=[r for r in result["rows"] if (r["teacher"],r["method"],r["rank"])==(activation,method,64)]
            for ref,ratio in judgement["ratios"].items():
                actual=math.prod(r["normalized_mse"] / by_key[activation,r["layer"],r["seed"],64,ref]["normalized_mse"] for r in rows)**(1/9)
                assert abs(actual-ratio)<1e-12
            assert not judgement["gates"]["mean_error_at_most_five_percent"]
            assert not judgement["gates"]["five_percent_better_than_pca"]
            assert not judgement["gates"]["five_percent_better_than_linear_aware"]
    with gzip.open(ROOT/"metrics.csv.gz","rt",newline="") as handle:
        table=list(csv.DictReader(handle))
    assert len(table)==378
    for exported,actual in zip(table,result["rows"],strict=True):
        for key in ("parameters","normalized_mse","inference_ms","pipeline_peak_cuda_bytes"):
            assert float(exported[key])==actual[key]
    assert "116 passed" in read(ROOT/"tests.log") and read(ROOT/"tests_exit.txt").strip()=="0"
    for name in ("study","audit","diagnose","analyze","plot","audit_rank"):
        assert read(ROOT/f"{name}_exit.txt").strip()=="0",name
    packed=ROOT/"result.json.gz"
    packed.write_bytes(gzip.compress((ROOT/"result.json").read_bytes(),mtime=0))
    assert gzip.decompress(packed.read_bytes())==(ROOT/"result.json").read_bytes()
    for name in ("study","audit","tests","audit_rank"):
        path=ROOT/f"{name}.log"
        path.with_suffix(".log.gz").write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    report=Path("research/real_subspace_results.md")
    for link in re.findall(r"\]\(([^)]+)\)",read(report)):
        if not link.startswith("https:"):
            assert (report.parent/link).exists(),link
    files=[report,Path("research/figures/real_subspace.png"),ROOT/"summary.json",ROOT/"audit.json",
           ROOT/"rank_diagnosis.json",ROOT/"rank_audit.json",ROOT/"result.json",packed,
           Path("src/core/ffn_capture.py"),Path("tests/test_ffn_capture.py"),*sorted((ROOT/"source").glob("*.py"))]
    receipt={"status":"PASS","frozen_source_files":len(sources),"frozen_source_hashes_match":True,
        "comparison_scores":378,"full_decoder_recaptured_pair_sets":18,"rank_bound_checks":54,
        "qualification_checks":19,"pytest_passed":116,"neural_updates":0,"all_primary_recipes_rejected":True,
        "rank64_error_floor_exceeds_threshold_all_cases":True,
        "largest_score_audit_absolute_error":max(r["absolute_error"] for r in audit["rows"]),
        "largest_rank_eigen_svd_difference":max(r["eigen_vs_svd_error"] for r in rank["checks"]),
        "report_links_exist":True,"broad_goal_achieved":False,
        "files":{str(p):sha(p) for p in files}}
    Path("results/verification/real_subspace_final_v1.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    run()
