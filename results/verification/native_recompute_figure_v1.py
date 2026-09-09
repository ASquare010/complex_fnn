"""Plot only recorded H063 resource measurements; no model execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root=Path("results/native_recompute_v1")
result=json.loads((root/"result.json").read_text())
recipes=("full_swiglu","full_gelu","calibrated_narrow","blockshuffle","rational")
scopes=("none","ffn","block")
labels=("Full\nSwiGLU","Full\nGELU","Calibrated\nnarrow","Plain\nBlockShuffle","Rational\nBlockShuffle")
colors=("#7b8794","#007f9b","#c56820")
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
x=np.arange(len(recipes))
for i,(scope,color) in enumerate(zip(scopes,colors)):
    rows=[result["results"][f"{r}_{scope}"] for r in recipes]
    axes[0].bar(x+(i-1)*.24,[r["peak_allocated_bytes"]/2**20 for r in rows],.24,color=color,label={"none":"Existing execution","ffn":"Whole FFN","block":"Whole block"}[scope])
    axes[1].bar(x+(i-1)*.24,[r["median_step_ms"] for r in rows],.24,color=color)
for ax in axes:
    ax.set_xticks(x,labels,fontsize=9)
    ax.grid(axis="y",alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set(ylabel="Peak allocated GPU memory (MiB)",title="Equivalent checkpoint options for every recipe")
axes[1].set(ylabel="Median training update (ms)",title="10 timed updates after 10 warmup updates")
axes[0].legend(fontsize=8)
fig.suptitle("Native recomputation audit | trained seed-17 checkpoints | synthetic continuations",fontsize=12)
for suffix in ("svg","png"):
    path=Path("research/figures")/f"native_recompute.{suffix}"
    assert not path.exists()
    fig.savefig(path,dpi=160)
print(json.dumps({"status":"PASS","resource_cells":15,"corpus_scoring":False}))