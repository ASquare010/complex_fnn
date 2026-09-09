"""Plot recorded H066 measurements only; no model execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

r=json.loads(Path("results/staged_resource_v1/result.json").read_text())
recipes=("full_swiglu","full_gelu","rational")
labels=("Full SwiGLU","Full GELU","Rational BlockShuffle")
fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout="constrained")
x=np.arange(3)
for index,(legend,color) in enumerate((("Whole block + original inner policy","#758492"),("Additional internal checkpoint regions","#d4802b"))):
    suffixes=("native",)*3 if index==0 else ("inner","inner","staged")
    rows=[r["results"][f"{recipe}_{suffix}"] for recipe,suffix in zip(recipes,suffixes)]
    for ax,key,factor in ((axes[0],"peak_allocated_bytes",2**20),(axes[1],"median_step_ms",1)):
        values=[row[key]/factor for row in rows]
        bars=ax.bar(x+(index-.5)*.34,values,.34,color=color,label=legend)
        ax.bar_label(bars,fmt="%.1f",fontsize=8,padding=3)
limit=min(r["dense_memory_limits_bytes"].values())/2**20
axes[0].axhline(limit,color="#a23939",linestyle="--",linewidth=1)
axes[0].text(-.5,limit+6,f"Stricter full-control cap: {limit:.3f} MiB",fontsize=8,color="#a23939")
axes[0].set(ylabel="Peak allocated GPU memory (MiB)",ylim=(0,520),title="Staged rational saves 32 MiB; memory gate fails")
axes[1].set(ylabel="Median update time (ms)",ylim=(0,270),title="10 timed updates after 10 warmup updates")
for ax in axes:
    ax.set_xticks(x,labels,fontsize=8)
    ax.grid(axis="y",alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
axes[1].legend(loc="upper left",fontsize=7,frameon=False)
fig.suptitle("Full-model staged recomputation | exact measured numerical comparisons | synthetic updates",fontsize=11)
for suffix in ("png","svg"):
    out=Path(f"research/figures/staged_resource.{suffix}")
    assert not out.exists()
    fig.savefig(out,dpi=160)
print(json.dumps({"status":"PASS","recorded_resource_cells":6,"new_model_execution":False}))
