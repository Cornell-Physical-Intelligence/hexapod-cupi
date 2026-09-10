"""Replay the unchanged frozen quiet scorer against the existing standing trace."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
p=Path(__file__).resolve().parent
expected=json.loads((p/'report.json').read_text())
source=p/'frozen_omni_quiet_review.py'
trace=p.parent/'reference_physics_002/results/run/standing/trace.npz'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected['scoring_source_sha256']
assert hashlib.sha256(trace.read_bytes()).hexdigest()==expected['trace_sha256']
tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='QUIET_GATES' for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='quiet_metrics')]
assert len(nodes)==2
namespace={'np':np}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
raw=np.load(trace);data={k:raw[k] for k in raw.files}
rows=[namespace['quiet_metrics'](data,i,200,data['joint_names'].tolist(),.02) for i in range(32)]
assert rows==expected['replicas']
assert namespace['QUIET_GATES']==expected['gate_bounds']
print(json.dumps({'all_32_rows_exactly_reproduced':True,'passed':sum(r['pass'] for r in rows),'stage2_complete':False}))
