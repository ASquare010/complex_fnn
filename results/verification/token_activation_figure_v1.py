"""Illustrate H067's defined scalar branch; curves are not fitted or learned."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

v=np.linspace(-6,6,801)
sigmoid=1/(1+np.exp(-v))
fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout="constrained")
for route,color in ((-3,"#287ca0"),(0,"#6d747a"),(3,"#d58331")):
    alpha=.25*np.tanh(route)
    q=v+alpha*v*sigmoid
    derivative=1+alpha*(sigmoid+v*sigmoid*(1-sigmoid))
    label=f"Router preactivation {route:+d}"
    axes[0].plot(v,q,color=color,label=label)
    axes[1].plot(v,derivative,color=color,label=label)
bound=1-.25*(1+1/np.e)
axes[1].axhline(bound,color="#a44949",linestyle="--",linewidth=1,label=f"Conservative partial bound {bound:.3f}")
axes[0].set(title="Value shape changes with the token's router",xlabel="Projected value v",ylabel="v + 0.25 tanh(r) SiLU(v)")
axes[1].set(title="Derivative holding router fixed",xlabel="Projected value v",ylabel="Partial derivative with respect to v",ylim=(.6,1.35))
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=7,loc="upper left")
fig.suptitle("Defined activation family | illustrative settings, not learned curves or a full-network gradient bound",fontsize=10)
for suffix in ("png","svg"):
    out=Path(f"research/figures/token_activation.{suffix}")
    assert not out.exists()
    fig.savefig(out,dpi=160)
print(json.dumps({"status":"PASS","illustration_only":True,"training_data_used":False}))
