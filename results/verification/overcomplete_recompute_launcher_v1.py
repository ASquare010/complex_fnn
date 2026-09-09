"""Durable H055 launcher; one fixed recomputed candidate, no automatic retries."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path('results/verification')
for label,args,timeout in (
 ('focused_tests',['-m','pytest','-p','no:anyio','-q','tests/test_overcomplete_recompute.py','tests/test_overcomplete_integration.py','tests/test_overcomplete_ffn.py'],180),
 ('preflight',['-m','src.core.overcomplete_recompute','preflight'],180),
 ('gpu',['-m','src.core.overcomplete_recompute','gpu'],300),
 ('finish',['-m','src.core.overcomplete_recompute','finish'],60),
 ('tests',['-m','pytest','-p','no:anyio','-q'],240),
):
 p=root/f'overcomplete_recompute_{label}_v1.json'
 assert not p.exists()
 sources={f.as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for base in ('src','tests') for f in Path(base).rglob('*.py')}
 command=[sys.executable,'-X','faulthandler',*args]
 record={'status':'running','command':command,'source_files':sources,'started_unix':time.time(),'plan_sha256':hashlib.sha256(Path('research/overcomplete_recompute_plan.md').read_bytes()).hexdigest()}
 p.write_text(json.dumps(record,indent=2),encoding='utf-8')
 start=time.perf_counter()
 try:
  with (root/f'overcomplete_recompute_{label}_v1.log').open('w',encoding='utf-8') as log:
   code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
  record.update(status='complete',returncode=code)
 except BaseException as exc:
  record.update(status='failed',returncode=None,exception=repr(exc))
 record.update(seconds=time.perf_counter()-start,source_unchanged=all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in sources.items()))
 p.write_text(json.dumps(record,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in record.items() if k!='source_files'}),flush=True)
 if record['returncode']!=0:sys.exit(1)
