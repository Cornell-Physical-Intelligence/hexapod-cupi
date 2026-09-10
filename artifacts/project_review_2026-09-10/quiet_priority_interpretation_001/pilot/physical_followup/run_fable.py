"""Run the requested partner with supplied evidence and no tools/session log."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

root = Path(__file__).resolve().parent
cmd = ['claude', '--print', '--model', 'claude-fable-5-1', '--effort', 'max',
       '--tools', '', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
       '--disable-slash-commands', '--no-session-persistence', '--safe-mode',
       '--output-format', 'json']
(root / 'FABLE_LAUNCH.json').write_text(json.dumps({
    'argv': cmd, 'started_unix': time.time(),
    'prompt_sha256': hashlib.sha256((root / 'fable_prompt.md').read_bytes()).hexdigest(),
    'tools_disabled': True, 'session_persistence': False,
    'output_scope': 'Final response/provenance JSON only; no streaming or internal thinking logs',
}, indent=2) + '\n')
with (root / 'fable_prompt.md').open() as i, (root / 'fable_final.json').open('w') as o, (root / 'fable_stderr.log').open('w') as e:
    result = subprocess.run(cmd, stdin=i, stdout=o, stderr=e)
(root / 'FABLE_EXIT.json').write_text(json.dumps({'returncode': result.returncode, 'finished_unix': time.time()}, indent=2) + '\n')
raise SystemExit(result.returncode)
