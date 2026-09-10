from pathlib import Path
import hashlib,json,subprocess,time
p=Path(__file__).resolve().parent
cmd=['claude','--print','--model','claude-fable-5-1','--effort','max','--tools','','--strict-mcp-config','--mcp-config','{"mcpServers":{}}','--disable-slash-commands','--no-session-persistence','--output-format','json']
(p/'LAUNCH.json').write_text(json.dumps({'argv':cmd,'started_unix':time.time(),'prompt_sha256':hashlib.sha256((p/'prompt.md').read_bytes()).hexdigest(),'tools_disabled':True},indent=2)+'\n')
with (p/'prompt.md').open() as i,(p/'result.json').open('w') as o,(p/'stderr.log').open('w') as e:
 r=subprocess.run(cmd,stdin=i,stdout=o,stderr=e)
(p/'EXIT.json').write_text(json.dumps({'returncode':r.returncode,'finished_unix':time.time()},indent=2)+'\n')
raise SystemExit(r.returncode)
