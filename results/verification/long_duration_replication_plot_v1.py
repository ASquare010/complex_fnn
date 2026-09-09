"""Regenerate only H051 plots from verified records, without importing Torch."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seeds=(17,29,43)
recipes=("full_swiglu","calibrated_narrow","blockshuffle","full_gelu")
labels=("Full SwiGLU","Calibrated narrow","Plain BlockShuffle","Full GELU")
root=Path("results/verification")
assert json.loads((root/"long_duration_replication_report_v1.json").read_text())["returncode"]==0
assert json.loads((root/"long_duration_replication_tests_v1.json").read_text())["returncode"]==0
trials=json.loads(Path("results/long_duration_v1/result.json").read_text())["trials"]
trials=[{**r,"seed":17} for r in trials]+json.loads(Path("results/long_duration_replication_v1/result.json").read_text())["trials"]
assert len(trials)==12 and all(r["status"]=="COMPLETE" for r in trials)
inputs={}
fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
for ax,seed in zip(axes,seeds):
    for color,(recipe,label) in enumerate(zip(recipes,labels)):
        trial=next(t for t in trials if t["seed"]==seed and t["recipe"]==recipe)
        p=Path("results/runs")/trial["run"]
        assert hashlib.sha256((p/"metrics.json").read_bytes()).hexdigest()==trial["metrics_sha256"]
        m=json.loads((p/"metrics.json").read_text())
        h=[json.loads(s) for s in (p/"history.jsonl").read_text().splitlines()]
        assert [r["step"] for r in h]==[1,800,1600,2400,3200]
        assert h[-1]["validation_loss"]==m["validation_loss"]==trial["nll"]
        inputs[trial["run"]]={"metrics_sha256":trial["metrics_sha256"],"history_sha256":hashlib.sha256((p/"history.jsonl").read_bytes()).hexdigest()}
        h=[r for r in h if r["step"]>=800]
        ax.plot([r["step"] for r in h],[r["validation_loss"] for r in h],"o-",label=label,color=f"C{color}")
    ax.set(title=f"Seed {seed}: 3,200-step schedule",xlabel="Optimizer step",ylabel="Full-validation NLL")
    ax.grid(alpha=.2);ax.legend(fontsize=8)
outputs={}
for ext in ("png","svg"):
    path=Path(f"results/plots/long_duration_replication.{ext}")
    backup=root/f"long_duration_replication_before_palette.{ext}"
    assert not backup.exists();backup.write_bytes(path.read_bytes())
    fig.savefig(path,dpi=150)
    outputs[ext]={"before_sha256":hashlib.sha256(backup.read_bytes()).hexdigest(),"after_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
plt.close(fig)
record={"status":"PASS","scope":"Plot-only regeneration after full report and 233 tests passed. Same recorded data, fixed recipe colors/order, no Torch/report verifier import or new score.","inputs":inputs,"outputs":outputs,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/"long_duration_replication_plot_verification_v1.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps({"status":"PASS","records":len(inputs),"outputs":outputs}))
