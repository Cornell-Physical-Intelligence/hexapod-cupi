"""One tools-disabled MAX partner consultation; final response only."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
import uuid

root=Path(__file__).resolve().parent
session=str(uuid.uuid4())
cmd=['/Users/andreboufama/.local/bin/claude','--print','--model','claude-fable-5-1',
     '--effort','max','--tools','','--strict-mcp-config','--mcp-config','{"mcpServers":{}}',
     '--disable-slash-commands','--no-session-persistence','--safe-mode',
     '--session-id',session,'--output-format','json']
if (root/'FABLE_LAUNCH.json').exists():raise SystemExit('Preserve prior consultation; use a new version for another request')
(root/'FABLE_LAUNCH.json').write_text(json.dumps({'argv':cmd,'session_id':session,'started_unix':time.time(),
    'prompt_sha256':hashlib.sha256((root/'fable_prompt.md').read_bytes()).hexdigest(),
    'model':'claude-fable-5-1','effort':'max','tools_disabled':True,'MCP_disabled':True,
    'scope':'Final response JSON only; no JSONL events or internal thinking'},indent=2)+'\n')
with (root/'fable_prompt.md').open() as i, (root/'fable_final.json').open('w') as o, (root/'fable_stderr.txt').open('w') as e:
    result=subprocess.run(cmd,stdin=i,stdout=o,stderr=e,cwd=root)
(root/'FABLE_EXIT.json').write_text(json.dumps({'returncode':result.returncode,'finished_unix':time.time()},indent=2)+'\n')
raise SystemExit(result.returncode)
