"""H054 final artifact audit; no additional training or scoring."""
import ast,hashlib,json,re,zipfile
from pathlib import Path
from urllib.parse import unquote

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=Path('results/overcomplete_integration_v1')
p=read(root/'protocol.json');r=read(root/'result.json');pre=read(root/'preflight.json')
assert r['status']=='complete' and not r['earns_separate_quality_screen']
assert not r['gates']['memory_within_ten_percent_gelu']
assert sha('research/overcomplete_integration_plan.md')==p['plan_sha256']==r['plan_sha256']
assert sha(root/'preflight.json')==p['preflight_sha256'] and pre['old_variants_exact']==31
assert sha('results/verification/overcomplete_before_v1.pt')==pre['snapshot_sha256']
with zipfile.ZipFile(root/'source.zip') as z:
 for n,h in p['provenance']['source_files'].items():
  assert sha(n)==h and hashlib.sha256(z.read(n)).hexdigest()==h
 assert hashlib.sha256(z.read('research/overcomplete_integration_plan.md')).hexdigest()==p['plan_sha256']
phases={}
for label in ('focused_tests','preflight','swiglu','overcomplete_headwise_swiglu','gelu','finish','tests'):
 path=Path(f'results/verification/overcomplete_integration_{label}_v1.json')
 rec=read(path)
 assert rec['returncode']==0 and rec['source_unchanged']
 assert all(sha(n)==h for n,h in rec['source_files'].items())
 phases[label]={'record':path.as_posix(),'sha256':sha(path),'seconds':rec['seconds']}
assert '241 passed' in Path('results/verification/overcomplete_integration_tests_v1.log').read_text()
for variant,t in r['trials'].items():
 path=root/variant
 assert sha(path/'result.json')==t['result_sha256']
 cell=read(path/'result.json')
 assert cell['status']=='PASS' and cell['provenance']['source_files']==p['provenance']['source_files']
 assert all(sha(path/n)==h for n,h in cell['files'].items())
 assert cell['checkpoint_logits_roundtrip_exact'] and cell['all_weights_gradients_moments_and_layer_diagnostics_finite']
for n,h in pre['data_hashes'].items():assert sha(Path('data/wikitext2_v1')/n)==h
previous=read('results/verification/overcomplete_final_v1.json')
assert sha('results/overcomplete_qualification_v1/cpu.json')==previous['cpu_result_sha256']
assert sha('results/overcomplete_qualification_v1/cuda.json')==previous['cuda_result_sha256']
plans={**previous['plans_unchanged'],'overcomplete_integration_v1':'overcomplete_integration_plan.md'}
for folder,plan in plans.items():assert read(Path('results')/folder/'protocol.json')['plan_sha256']==sha(Path('research')/plan)
config=ast.parse(Path('src/core/config.py').read_text())
variants=next(ast.literal_eval(n.value) for n in config.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VARIANTS' for t in n.targets))
assert len(variants)==32
assert len(list(Path('results/runs').glob('*/metrics.json')))==165
files=[Path(n) for n in ('README.md','research/CURRENT_STATE.md','research/idea_bank.md','research/overcomplete_integration_plan.md','research/overcomplete_integration_results.md','src/multihead_ffn/overcomplete.md')]
links=0
for path in files:
 for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
  target=target.strip('<>').split('#')[0]
  if not target or re.match(r'^[a-zA-Z]+://',target):continue
  assert (path.parent/unquote(target)).exists(),(path,target)
  links+=1
record={'status':'PASS','research_target_passes':False,'full_tests_passed':241,'full_test_record':'results/verification/overcomplete_integration_tests_v1.json',
        'all_tested_sources_exact':True,'old_variants_exact':31,'registered_variants':32,'retained_lm_profile_runs':165,
        'training_token_exposures':122880,'validation_targets_scored':0,'numerical_failures':0,'process_failures':0,
        'gates':r['gates'],'earns_quality_screen':False,'plans_unchanged':plans,'phases':phases,'local_links_checked':links,
        'all_data_and_checkpoint_artifact_hashes_exact':True,'result_sha256':sha(root/'result.json'),
        'source_archive_sha256':sha(root/'source.zip'),'documents':{path.as_posix():sha(path) for path in files}}
Path('results/verification/overcomplete_integration_final_v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ('documents','plans_unchanged','phases')}),flush=True)
