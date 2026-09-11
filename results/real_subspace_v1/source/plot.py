"""Plot audited JSON in a separate process without importing Torch."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run():
    s=json.loads(Path("results/real_subspace_v1/summary.json").read_text())
    assert s["audited_scores"]==378
    styles=[("pca","Input PCA","#4d749c"),
            ("linear_aware","Linear-response aware","#13856c"),
            ("energy","Output energy","#d18b33"),
            ("white_energy","Whitened energy","#a65289"),
            ("white_residual","Whitened residual energy","#8b69c4"),
            ("white_permuted","Shuffled energy","#9a9fa6")]
    fig,axes=plt.subplots(1,2,figsize=(11,5.8))
    fig.subplots_adjust(left=.085,right=.98,bottom=.32,top=.82,wspace=.25)
    for ax,teacher in zip(axes,("gelu","swiglu"),strict=True):
        for method,label,color in styles:
            ax.plot([32,64,128],[s["groups"][teacher][method][str(r)]["normalized_mse"]["mean"] for r in (32,64,128)],
                    marker="o",label=label,color=color,linewidth=1.8)
        ax.plot([32,64,128],[s["rank_error_floors"][teacher][str(r)]["mean"] for r in (32,64,128)],
                color="#28313c",linestyle=":",label="Output-rank error floor",linewidth=2)
        ax.axhline(.05,color="#3b414b",linewidth=1,linestyle="--",label="5% allocation threshold")
        ax.set_xscale("log",base=2)
        ax.set_yscale("log")
        ax.set_xticks([32,64,128],["32\n~89% fewer weights","64\n~79% fewer weights","128\n~58% fewer weights"],fontsize=8)
        ax.set_xlim(26,158)
        ax.set_ylim(.025,1.6)
        ax.set_title(teacher.upper(),loc="left",fontweight="bold")
        ax.set_ylabel("Normalized FFN output MSE")
        ax.set_xlabel("Shared input and output rank")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.18)
    fig.suptitle("Energy-based feature discovery loses on real FFN inputs",fontsize=15,y=.96)
    fig.text(.5,.885,"Means across three layers and three calibration samples per pretrained teacher.",ha="center",fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.02,"378 local comparisons; zero SGD updates. Output-rank floor is a post-screen mathematical diagnostic.",ha="center",fontsize=9)
    fig.savefig("research/figures/real_subspace.png",dpi=160)
    plt.close(fig)
    print("H105 figure written")


if __name__=="__main__":
    run()
