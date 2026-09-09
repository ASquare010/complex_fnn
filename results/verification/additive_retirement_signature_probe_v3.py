import json
import runpy
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
out=Path("results/verification/additive_retirement_signature_values_v3.json")
assert not out.exists()
fn=runpy.run_path("results/verification/cleanup_signatures_v1.py")["signatures"]
actual=fn()
expected=json.loads(Path("results/verification/cleanup_before_signatures_v1.json").read_text())
assert actual==expected
out.write_text(json.dumps(actual,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":"PASS","exact_cpu_gpu_cases":len(actual)}))
