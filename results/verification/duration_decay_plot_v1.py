"""Plot verified H052 validation histories; no Torch/report imports."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
root = Path('results/duration_decay_v1')
result, pre = read(root/'result.json'), read(root/'preflight.json')
rows = []
styles = {'full_swiglu':('Full SwiGLU','C0','-'), 'full_gelu':('Full GELU','C1','-'),
          'calibrated_narrow':('Calibrated narrow','C4','-'), 'blockshuffle':('BlockShuffle: parameter decay','0.4','--'),
          'product_decay':('BlockShuffle: product decay','C2','-')}
for recipe in styles:
    record = result if recipe == 'product_decay' else pre['references'][recipe]
    path = Path('results/runs')/record['run']/'history.jsonl'
    assert sha(path) == record['files']['history.jsonl']
    h = [json.loads(s) for s in path.read_text().splitlines()]
    rows.append((recipe,h,path))
fig, ax = plt.subplots(figsize=(9,4.8),layout='constrained')
for recipe,h,path in rows:
    label,color,style = styles[recipe]
    late = [r for r in h if r['step'] >= 800]
    ax.plot([r['step'] for r in late],[r['validation_loss'] for r in late],marker='o',
            label=label,color=color,linestyle=style,linewidth=1.8)
ax.set(xlabel='Optimizer steps',ylabel='Validation NLL (lower is better)',
       title='WikiText, seed 17: only BlockShuffle factor decay changes',xticks=[800,1600,2400,3200])
ax.grid(alpha=.2)
ax.legend(fontsize=8)
fig.text(.5,-.03,'Full 322,688-target validation. Early losses are retained in the report. One seed; convergence unproven.',ha='center',fontsize=8)
paths=[]
for ext in ('png','svg'):
    p=Path(f'results/plots/duration_decay.{ext}')
    assert not p.exists()
    fig.savefig(p,dpi=180,bbox_inches='tight')
    paths.append(p)
plt.close(fig)
record={'status':'PASS','script_sha256':sha(Path(__file__)),'histories':{r:sha(p) for r,h,p in rows},
        'figures':{p.as_posix():sha(p) for p in paths},'plotted_steps':[800,1600,2400,3200]}
Path('results/verification/duration_decay_plot_verification_v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
