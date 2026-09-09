import ast,hashlib,json,re,zipfile
from pathlib import Path
from urllib.parse import unquote

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
root=Path('results/overcomplete_qualification_v1')
protocol,cpu,gpu=read(root/'protocol.json'),read(root/'cpu.json'),read(root/'cuda.json')
assert cpu['status']==gpu['status']=='PASS'
assert sha(root/'protocol.json')==cpu['protocol_sha256']
assert sha(root/'cpu.json')==gpu['cpu_result_sha256']
assert sha('research/overcomplete_qualification_plan.md')==cpu['plan_sha256']==gpu['plan_sha256']==protocol['plan_sha256']
assert sha(root/'witness_weights.pt')==cpu['witness']['weights_sha256']
style=read(root/'style_equivalence.json')
assert style['status']=='PASS' and style['all_other_ast_exact']
assert sha(style['file'])==style['current_sha256']
assert sha(root/'qualification_before_style.py')==style['prior_sha256']
with zipfile.ZipFile(root/'source.zip') as z:
 for n,h in protocol['provenance']['source_files'].items():
  assert hashlib.sha256(z.read(n)).hexdigest()==h
  assert sha(n)==(style['current_sha256'] if n==style['file'] else h)
 assert hashlib.sha256(z.read('research/overcomplete_qualification_plan.md')).hexdigest()==protocol['plan_sha256']
phases={}
for label in ('focused_tests','cpu','cuda','tests'):
 p=Path(f'results/verification/overcomplete_{label}_v1.json')
 r=read(p)
 assert r['returncode']==0 and r['source_unchanged']
 assert all(sha(n)==(style['current_sha256'] if n==style['file'] else h) for n,h in r['source_files'].items())
 phases[label]={'sha256':sha(p),'seconds':r['seconds']}
assert '238 passed' in Path('results/verification/overcomplete_tests_v1.log').read_text()
old=read('results/verification/duration_decay_final_v1.json')
assert sha('results/duration_decay_v1/result.json')==old['result_sha256']
assert not read('results/duration_decay_v1/result.json')['decision']['earns_replication']
old_sources=read(old['full_test_record'])['source_files']
assert all(sha(n)==h for n,h in old_sources.items())
plans={**old['plans_unchanged'],'overcomplete_qualification_v1':'overcomplete_qualification_plan.md'}
for folder,plan in plans.items():
 assert read(Path('results')/folder/'protocol.json')['plan_sha256']==sha(Path('research')/plan)
for n,h in read('results/duration_decay_v1/preflight.json')['data_hashes'].items():
 assert sha(Path('data/wikitext2_v1')/n)==h
config_ast=ast.parse(Path('src/core/config.py').read_text())
variants=next(ast.literal_eval(n.value) for n in config_ast.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VARIANTS' for t in n.targets))
assert len(variants)==31 and not any('overcomplete' in v for v in variants)
assert len(list(Path('results/runs').glob('*/metrics.json')))==165
files=[Path(n) for n in ('README.md','research/CURRENT_STATE.md','research/idea_bank.md',
 'research/overcomplete_qualification_plan.md','research/overcomplete_qualification_results.md',
 'src/multihead_ffn/overcomplete.md','research/headwise_hessian_obstruction.md','research/structured_headwise_budget.md')]
links=0
for p in files:
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
  target=target.strip('<>').split('#')[0]
  if not target or re.match(r'^[a-zA-Z]+://',target): continue
  assert (p.parent/unquote(target)).exists(),(p,target)
  links+=1
peaks=gpu['isolated_peaks']
record={'status':'PASS','research_target_passes':False,'full_tests_passed':238,
 'full_test_record':'results/verification/overcomplete_tests_v1.json',
 'all_prior_235_tested_sources_exact':True,'post_test_style_equivalence':style,
 'registered_variants':31,'standalone_unregistered_candidates_added':1,'retained_lm_profile_runs':165,
 'new_language_training_tokens':0,'new_validation_targets_scored':0,'synthetic_cuda_updates':10,
 'new_cpu_gaussian_inputs':12288,'phases':phases,'plans_unchanged':plans,'local_links_checked':links,
 'all_immutable_data_hashes_exact':True,'cpu_result_sha256':sha(root/'cpu.json'),
 'cuda_result_sha256':sha(root/'cuda.json'),'source_archive_sha256':sha(root/'source.zip'),
 'witness_weights_sha256':sha(root/'witness_weights.pt'),'ffn_weights_per_layer':350208,
 'isolated_memory_cost_percent':100*(peaks['overcomplete']['peak_allocated_bytes']/peaks['full_swiglu']['peak_allocated_bytes']-1),
 'earns_separate_integration_and_memory_qualification':True,'earns_language_training':False,
 'documents':{p.as_posix():sha(p) for p in files},'numerical_failures':0,'process_failures':0}
Path('results/verification/overcomplete_final_v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ('documents','plans_unchanged','phases','post_test_style_equivalence')}),flush=True)
