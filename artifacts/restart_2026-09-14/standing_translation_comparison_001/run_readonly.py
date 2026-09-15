"""Send a bounded stdlib program through SSH stdin; only local outputs are written."""
from pathlib import Path
import json,subprocess,hashlib,datetime

root=Path(__file__).resolve().parent
parts=[root/'numeric_evidence.py',root/'compare_remote.py',root/'INPUT_PINS.json']
pins=json.loads(parts[2].read_text())
script=parts[0].read_text()+'\nINPUT_PINS='+repr(pins)+'\n'+parts[1].read_text()
command=['ssh','-o','BatchMode=yes','orionh@100.82.166.9','/usr/bin/python3','-B','-S']
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
result=subprocess.run(command,input=script,text=True,capture_output=True,timeout=180)
(root/'ANALYSIS.json').write_text(result.stdout)
(root/'ANALYSIS.stderr.txt').write_text(result.stderr)
record={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'command':command,'returncode':result.returncode,
        'local_sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in parts},
        'stdin_sha256':hashlib.sha256(script.encode()).hexdigest(),
        'remote_files_written':False,'native_or_gpu_calls':False}
(root/'EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(result.returncode)
