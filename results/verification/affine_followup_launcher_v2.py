"""Finish bounded post-screen work sequentially after the affine cohort closes."""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.core.reproducibility import provenance, write_json

root=Path("results/verification")
status=root/"affine_followup_launcher_v2.json"
assert not status.exists()
records=[]

def run(name, command, timeout):
    print("Starting "+name,flush=True)
    with (root/(name+"_continuation_v2.log")).open("x",encoding="utf-8") as log:
        start=time.perf_counter()
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    records.append({"name":name,"command":command,"returncode":result.returncode,"seconds":time.perf_counter()-start})
    write_json(status,{"status":"running","completed":records})
    assert result.returncode==0,(name,result.returncode)
    print("Completed "+name,flush=True)

try:
    start=time.perf_counter()
    while True:
        p=Path("results/affine_activation_screen_v1/progress.json")
        if p.exists() and json.loads(p.read_text())["status"]=="complete":
            break
        assert not Path("results/affine_activation_screen_v1/failure.json").exists()
        assert time.perf_counter()-start<1800
        time.sleep(5)
    assert Path("results/activation_data_fit_v1/result.json").exists()
    assert not Path("results/affine_activation_shapes_v1").exists()
    # Previous process failed during torch import, before creating this output.
    run("affine_activation_shapes_v1",[sys.executable,"-X","faulthandler","-m","src.core.activation_shapes","audit","--selection","results/affine_activation_screen_v1/result.json","--plan","research/affine_activation_plan.md","--output","results/affine_activation_shapes_v1"],900)
    run("affine_activation_report_v1",[sys.executable,"-m","src.core.affine_activation_report"],90)
    run("compiled_correction_tests_v1",[sys.executable,"-m","pytest","tests/test_compiled_correction.py","-q"],240)
    run("activation_correction_execution_v1",[sys.executable,"-X","faulthandler","-m","src.core.activation_correction_execution","--output","results/activation_correction_execution_v1"],1000)
    write_json(status,{"status":"complete","completed":records,"provenance":provenance()})
except BaseException as exc:
    write_json(status,{"status":"failed","completed":records,"error":repr(exc)})
    raise
