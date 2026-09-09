"""Report only the complete, hash-verified H052 endpoint; no model imports."""
import hashlib
import json
from pathlib import Path

root = Path('results/duration_decay_v1')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
result, pre = read(root/'result.json'), read(root/'preflight.json')
assert result['status'] == 'complete'
assert result['plan_sha256'] == sha('research/duration_decay_plan.md')
paths = {r: Path('results/runs')/v['run'] for r,v in pre['references'].items()}
paths['product_decay'] = Path('results/runs')/result['run']
for r,p in paths.items():
    files = result['files'] if r == 'product_decay' else pre['references'][r]['files']
    assert all(sha(p/n) == h for n,h in files.items())
metrics = {r:read(p/'metrics.json') for r,p in paths.items()}
histories = {r:[json.loads(s) for s in (p/'history.jsonl').read_text().splitlines()] for r,p in paths.items()}
names = {'full_swiglu':'Full SwiGLU', 'full_gelu':'Full GELU', 'calibrated_narrow':'Calibrated narrow',
         'blockshuffle':'BlockShuffle: parameter decay', 'product_decay':'BlockShuffle: product decay'}
order = ('full_swiglu','full_gelu','calibrated_narrow','blockshuffle','product_decay')
d = result['decision']
rows = ['| Recipe | Final validation NLL | FFN weights | Peak MiB | Clipped steps |', '|---|---:|---:|---:|---:|']
for r in order:
    m = metrics[r]
    rows.append(f'| {names[r]} | {m["validation_loss"]:.6f} | {m["ffn_parameters"]:,} | {m["peak_allocated_vram_bytes"]/2**20:.2f} | {100*m["clipped_step_fraction"]:.2f}% |')
relative = ['| Product-decay comparison | Relative NLL cost |','|---|---:|']
relative += [f'| vs {names[r]} | {v:+.4f}% |' for r,v in d['relative_nll_percent'].items()]
trajectory = ['| Step | Parameter-decay NLL | Product-decay NLL |','|---|---:|---:|']
for a,b in zip(histories['blockshuffle'],histories['product_decay'],strict=True):
    assert a['step'] == b['step']
    trajectory.append(f'| {a["step"]} | {a["validation_loss"]:.6f} | {b["validation_loss"]:.6f} |')
verdict = 'PASSES the frozen local and material-gain gates; earns a separately frozen seed-29/43 replication' if d['earns_replication'] else 'FAILS the frozen promotion rule; no further product-decay tuning is earned by this cell'
text = f'''# H052: Longer-duration factor-decay control

**Product decay {verdict}.** This is one seed, not a robust improvement or a
new optimizer. The earlier three-seed parameter-decay failure remains.

## Controlled endpoint

The [frozen plan](duration_decay_plan.md) changes only the existing
TrainConfig.ffn_decay_mode from parameter to product. The model remains plain
SwiGLU BlockShuffle: d=384, eight layers, hidden 2048, G=8, context 128, batch 16,
peak LR 0.0012, 320-step warmup/cosine, native BF16 and gate recomputation.
All initial weights, LR multipliers and non-FFN treatments are unchanged.

{chr(10).join(rows)}

Each row uses seed 17 and 3,200 steps / 6,553,600 sampled tokens. The four H050
references are retained; only the product-decay row adds training tokens.
Every endpoint scores all 322,688 validation targets; the official test remains
unscored. FFN reduction is 70.3125%; both BlockShuffle models have 9,099,648 total
parameters. Inference work and activation are unchanged. Cross-session training
throughput is recorded in raw metrics but does not support a paired speed claim.

{chr(10).join(relative)}

Positive NLL cost is worse. Local quality/memory gates: **{'PASS' if d['local_gates_pass'] else 'FAIL'}**.
The separately required >=0.2% NLL improvement over parameter-decay BlockShuffle:
**{'PASS' if d['material_gain_pass'] else 'FAIL'}**. Both conditions are required before more seeds.
The raw [decision](../results/duration_decay_v1/result.json) retains each gate.

## Trajectory and scope

{chr(10).join(trajectory)}

Product decay's final 800-step NLL change is
{result['relative_last_800_step_nll_change_percent']:+.4f}%; its late-plateau diagnostic
**{'PASSES' if result['late_plateau_screen_passes'] else 'FAILS'}**. This operational check cannot prove convergence.
Initialization NLL and the first pre-update training loss match the old run within
1e-7. Step-1 validation follows one update, so it need not match after decay changes.
All completed diagnostics and checkpoint weights are finite; no numerical failure
occurred. A one-seed trajectory cannot establish a general mechanism.

The zero-gradient/zero-Adam-moment decay identity explains the intervention,
not actual trained norms or complete updates. Up/gate factors change decay .1
to .0125; the down factors change to .0125/.00234375. LR multipliers stay (4,4)
and (4,64/3). Non-FFN parameters and full/narrow reference treatments are equal.
The [CPU diagnosis](duration_optimizer_geometry.md) did not observe Frobenius
collapse in the old model. H013's smaller TinyStories product-decay negative
result is retained in the [earlier plan](product_decay_plan.md).

## Reproduction and retained evidence

Plan SHA256: `{result['plan_sha256']}`.
Result SHA256: `{sha(root/'result.json')}`.
The [preflight](../results/duration_decay_v1/preflight.json),
[protocol](../results/duration_decay_v1/protocol.json) and
[worker qualification](../results/duration_decay_v1/qualification.json) preserve
all configurations, actual optimizer groups, source and data hashes. Run archives,
checkpoint, full history, diagnostics and phase logs are retained. The verifier
checks every reference artifact, initial loss, schedule, parameter/FLOP count,
final CUDA sampler state and the complete source archive. The original trainer,
optimizer implementation and all 31 models are unchanged.

The durable launcher uses UV with compile/data extras, runs one GPU worker and
stops on any unclassified process failure. Exact commands are in
[the launcher](../results/verification/duration_decay_launcher_v1.py).
No failed or partial trial is assigned an endpoint NLL.
'''
Path('research/duration_decay_results.md').write_text(text,encoding='utf-8')
summary = {'status':'PASS','result_sha256':sha(root/'result.json'),'decision':d,
           'reference_and_trial_artifacts_exact':True,'report_sha256':sha('research/duration_decay_results.md'),
           'script_sha256':sha(Path(__file__)),'metrics_sha256':{r:sha(p/'metrics.json') for r,p in paths.items()}}
Path('results/duration_decay_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary),flush=True)
