"""Durable one-shot H054 launcher; one GPU worker at a time, no retries."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
root=Path('results/verification')
phases=[('focused_tests',['-m','pytest','-p','no:anyio','-q','tests/test_overcomplete_integration.py','tests/test_overcomplete_ffn.py','tests/test_baselines.py'],180),
        ('preflight',['-m','src.core.overcomplete_integration','preflight'],180)]
phases.extend((v,['-m','src.core.overcomplete_integration','gpu','--variant',v],300) for v in ('swiglu','overcomplete_headwise_swiglu','gelu'))
phases.extend([('finish',['-m','src.core.overcomplete_integration','finish'],60),('tests',['-m','pytest','-p','no:anyio','-q'],240)])
for label,args,timeout in phases:
 p=root/f'overcomplete_integration_{label}_v1.json'
 assert not p.exists()
 sources={f.as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for base in ('src','tests') for f in Path(base).rglob('*.py')}
 command=[sys.executable,'-X','faulthandler',*args]
 record={'status':'running','command':command,'source_files':sources,'started_unix':time.time(),
         'plan_sha256':hashlib.sha256(Path('research/overcomplete_integration_plan.md').read_bytes()).hexdigest()}
 p.write_text(json.dumps(record,indent=2),encoding='utf-8')
 start=time.perf_counter()
 try:
  with (root/f'overcomplete_integration_{label}_v1.log').open('w',encoding='utf-8') as log:
   code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
  record.update(status='complete',returncode=code)
 except BaseException as exc:
  record.update(status='failed',returncode=None,exception=repr(exc))
 record.update(seconds=time.perf_counter()-start,source_unchanged=all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in sources.items()))
 p.write_text(json.dumps(record,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in record.items() if k!='source_files'}),flush=True)
 if record['returncode'] != 0:
  sys.exit(1)
