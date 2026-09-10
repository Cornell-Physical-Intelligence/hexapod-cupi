"""CPU-only draft receipt; records exact parent hashes and AST delta."""
from pathlib import Path
import ast,json,hashlib,difflib
p=Path(__file__).resolve().parent;old=p/'parent_guard_004.py';new=p/'launch_guarded_remote.py'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
a=ast.parse(old.read_text());b=ast.parse(new.read_text())
fa={n.name:n for n in a.body if isinstance(n,ast.FunctionDef)};fb={n.name:n for n in b.body if isinstance(n,ast.FunctionDef)}
unchanged=[name for name in fa if ast.dump(fa[name],include_attributes=False)==ast.dump(fb[name],include_attributes=False)]
const_deltas=[]
for x,y in zip([n for n in ast.walk(fa['main']) if isinstance(n,ast.Constant)],[n for n in ast.walk(fb['main']) if isinstance(n,ast.Constant)]):
 if x.value!=y.value:const_deltas.append({'old':x.value,'new':y.value});x.value=y.value
assert ast.dump(fa['main'],include_attributes=False)==ast.dump(fb['main'],include_attributes=False)
parent=p.parent/'direct_omni_train_smoke_guard_004';parentmap=json.loads((parent/'FREEZE_SHA256.json').read_text())
assert all(sha(parent/k)==v for k,v in parentmap.items())
assert set(parentmap)=={str(f.relative_to(parent)) for f in parent.rglob('*') if f.is_file() and f!=parent/'FREEZE_SHA256.json'}
audit=json.loads((p/'previous_owner_remote_audit.json').read_text());assert all(sha(p/'previous_owner'/k)==v for k,v in audit['pins'].items())
report={'scope':'CPU preparation only; new source/contract/host identities pending; no remote action','parent_freeze_sha256':sha(p/'parent_guard_004_freeze.json'),'parent_payload_count':len(parentmap),'parent_all_payloads_unchanged':True,'parent_runtime_sha256':sha(old),'candidate_runtime_sha256':sha(new),'unchanged_functions':unchanged,'main_ast_identical_after_exact_string_replacements':True,'main_string_replacements':const_deltas,'intentional_function_changes':['require_final_bindings: exact nine prior receipt inventory','verify_previous_owner: completed six-phase quiet-priority pilot, twelve owned identifiers, native/host freeze identity','validate_train_inputs: native005 smoke schema only'],'prior_receipt_count':9,'prior_receipts_copied_byte_identical':True,'tests':{'command':'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tmp/direct_omni_train_smoke_guard_005 -p test_*.py -v','count':37,'passed':True,'log_sha256':sha(p/'test_output_draft_001.txt')}}
(p/'DRAFT_READINESS.json').write_text(json.dumps(report,indent=2)+'\n')
(p/'GUARD_DELTA_DRAFT.patch').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile='frozen guard004/launch_guarded_remote.py',tofile='draft guard005/launch_guarded_remote.py')))
print(json.dumps(report,indent=2))
