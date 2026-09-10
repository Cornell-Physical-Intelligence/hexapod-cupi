"""Independently replay unchanged quiet criteria and explicit stop timing."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

p = Path(__file__).resolve().parent
run = p.parent / 'evidence/run'
source = p / 'frozen_omni_quiet_review.py'
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if (isinstance(n, ast.Assign) and any(
    isinstance(t, ast.Name) and t.id == 'QUIET_GATES' for t in n.targets))
    or (isinstance(n, ast.FunctionDef) and n.name == 'quiet_metrics')]
assert len(nodes) == 2
ns = {'np': np}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
with np.load(run / 'standing/trace.npz') as raw:
    data = {k: raw[k] for k in raw.files}
    standing = [ns['quiet_metrics'](data, i, 200, data['joint_names'].tolist(), .02) for i in range(32)]
references = json.loads((run / 'wave/reference_states.json').read_text())
last = references[-1]['result']['state']
quiet_start = int(np.ceil(last['reference_quiet_time_s'] / .02)) + 100
with np.load(run / 'wave/trace.npz') as raw:
    data = {k: raw[k] for k in raw.files}
    assert not np.any(data['requested_command'][quiet_start:])
    stop = ns['quiet_metrics'](data, 0, quiet_start, data['joint_names'].tolist(), .02)
report = dict(scoring_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              standing_trace_sha256=hashlib.sha256((run / 'standing/trace.npz').read_bytes()).hexdigest(),
              wave_trace_sha256=hashlib.sha256((run / 'wave/trace.npz').read_bytes()).hexdigest(),
              bounds=ns['QUIET_GATES'], standing=standing, final_stop=stop,
              stop_requested_s=last['stop_requested_time_s'], reference_quiet_s=last['reference_quiet_time_s'],
              reference_stop_latency_s=last['reference_quiet_time_s'] - last['stop_requested_time_s'],
              excluded_post_reference_settling_s=2.0, quiet_start_step=quiet_start,
              stage2_complete=False, scope='Reference only; not a prompt-stop or learned-policy qualification')
output = p / 'report.json'
if output.exists():
    assert json.loads(output.read_text()) == report
else:
    output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(standing_passed=sum(r['pass'] for r in standing), stop_passed=stop['pass'],
                     scored_quiet_s=stop['window_duration_s'], reference_stop_latency_s=report['reference_stop_latency_s'])))
