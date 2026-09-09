import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

root=Path('results/verification')
assert json.loads((root/'duration_decay_verify_v1.json').read_text())['returncode'] == 0
for label,args,timeout in (
 ('tests',['-m','pytest','-p','no:anyio','-q'],180),
 ('report',['results/verification/duration_decay_report_v1.py'],60),
 ('plot',['results/verification/duration_decay_plot_v1.py'],90),
):
 p=root/f'duration_decay_{label}_v1.json'
 assert not p.exists()
 sources={f.as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for base in ('src','tests') for f in Path(base).rglob('*.py')}
 command=[sys.executable,'-X','faulthandler',*args]
 record={'status':'running','command':command,'source_files':sources,'started_unix':time.time()}
 p.write_text(json.dumps(record,indent=2),encoding='utf-8')
 start=time.perf_counter()
 try:
  with (root/f'duration_decay_{label}_v1.log').open('w',encoding='utf-8') as log:
   code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
  record.update(status='complete',returncode=code)
 except BaseException as exc:
  record.update(status='failed',returncode=None,exception=repr(exc))
 record.update(seconds=time.perf_counter()-start,source_unchanged=all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in sources.items()))
 p.write_text(json.dumps(record,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in record.items() if k!='source_files'}),flush=True)
 if record['returncode'] != 0:
  sys.exit(1)
