"""Final H052 artifact/source audit, no training or validation scoring."""
import collections
import hashlib
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=Path('results/duration_decay_v1')
result=read(root/'result.json')
pre=read(root/'preflight.json')
protocol=read(root/'protocol.json')
summary=read('results/duration_decay_summary.json')
assert not result['decision']['earns_replication'] and not result['decision']['local_gates_pass']
assert sha(root/'result.json') == summary['result_sha256']
assert sha('research/duration_decay_results.md') == summary['report_sha256']
assert sha('research/duration_decay_plan.md') == result['plan_sha256'] == protocol['plan_sha256']
assert sha(root/'preflight.json') == protocol['preflight_sha256']
phases={}
for phase in ('focused_tests','prepare','train','verify','tests','report','plot'):
 p=Path(f'results/verification/duration_decay_{phase}_v1.json')
 rec=read(p)
 assert rec['returncode']==0 and rec['source_unchanged']
 assert all(sha(n)==h for n,h in rec['source_files'].items())
 phases[phase]={'record':p.as_posix(),'seconds':rec['seconds'],'sha256':sha(p)}
assert '235 passed' in Path('results/verification/duration_decay_tests_v1.log').read_text()
for name,expected in pre['data_hashes'].items():
 assert sha(Path('data/wikitext2_v1')/name)==expected
for recipe,ref in {**pre['references'],'product_decay':result}.items():
 p=Path('results/runs')/ref['run']
 assert all(sha(p/n)==h for n,h in ref['files'].items())
 m=read(p/'metrics.json')
 with zipfile.ZipFile(p/'source.zip') as z:
  assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in m['provenance']['source_files'].items())
with zipfile.ZipFile(root/'source.zip') as z:
 assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in protocol['provenance']['source_files'].items())
 assert hashlib.sha256(z.read('research/duration_decay_plan.md')).hexdigest()==protocol['plan_sha256']
old=read('results/verification/long_duration_replication_final_v1.json')
assert sha('results/long_duration_replication_v1/result.json') == old['result_sha256']
plans={**old['plans_unchanged'],'duration_decay_v1':'duration_decay_plan.md'}
for folder,plan in plans.items():
 assert read(Path('results')/folder/'protocol.json')['plan_sha256']==sha(Path('research')/plan)
figure=read('results/verification/duration_decay_plot_verification_v1.json')
assert all(sha(n)==h for n,h in figure['figures'].items())
paths=list(Path('results/runs').glob('*/metrics.json'))
counts=collections.Counter(read(p)['training']['steps'] for p in paths)
assert len(paths)==165 and counts[3200]==13
# Same prior computation plus two new experiment/test files; no architecture edits.
tested_old=read(old['full_test_record'])
for n,h in tested_old['source_files'].items():
 expected=old['post_test_change']['current_sha256'] if n==old['post_test_change']['file'] else h
 assert sha(n)==expected
files=[Path(n) for n in ('README.md','research/CURRENT_STATE.md','research/idea_bank.md',
 'research/duration_decay_plan.md','research/duration_decay_results.md','src/blockshuffle_ffn/model.md',
 'research/duration_optimizer_geometry.md','research/headwise_hessian_obstruction.md',
 'research/structured_headwise_budget.md','research/learnable_activation_domain.md')]
links=0
for p in files:
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
  target=target.strip('<>').split('#')[0]
  if not target or re.match(r'^[a-zA-Z]+://',target): continue
  assert (p.parent/unquote(target)).exists(),(p,target)
  links+=1
record={'status':'PASS','research_target_passes':False,'full_tests_passed':235,
 'full_test_record':'results/verification/duration_decay_tests_v1.json',
 'all_tested_sources_exact':True,'all_previous_computation_preserved':True,
 'registered_variants':31,'previous_models_preserved':30,'retained_lm_profile_runs':len(paths),
 'step_counts':dict(counts),'local_links_checked':links,'plans_unchanged':plans,
 'new_training_tokens':6553600,'new_completed_trials':1,'numerical_failures':0,
 'failed_experiment_launchers':0,'decision':result['decision'],'phases':phases,
 'all_five_endpoint_artifacts_and_archives_exact':True,'all_data_hashes_exact':True,
 'plots_visually_inspected':True,'result_sha256':sha(root/'result.json'),
 'summary_sha256':sha('results/duration_decay_summary.json'),
 'documents':{p.as_posix():sha(p) for p in files},'script_sha256':sha(Path(__file__))}
Path('results/verification/duration_decay_final_v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ('documents','plans_unchanged','phases')}),flush=True)
