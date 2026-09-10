from pathlib import Path
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
r=Path(__file__).resolve().parent
expected=json.loads((r/'BUNDLE_SHA256.json').read_text())
actual={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in r.rglob('*')if p.is_file()and p!=r/'BUNDLE_SHA256.json'and '__pycache__'not in p.parts}
assert actual==expected,'Published payload mismatch'
for name in ('source','host','guard','auditor'):
 p=r/name;e=json.loads((p/'FREEZE_SHA256.json').read_text())
 a={q.relative_to(p).as_posix():hashlib.sha256(q.read_bytes()).hexdigest()for q in p.rglob('*')if q.is_file()and q!=p/'FREEZE_SHA256.json'and '__pycache__'not in q.parts}
 assert a==e,name
raw=json.loads((r/'terminal/RAW_SHA256.json').read_text())
for name,digest in raw.items():assert hashlib.sha256((r/'terminal'/name).read_bytes()).hexdigest()==digest,name
a=json.loads((r/'terminal/audit.json').read_text());assert a['audit_verified']and a['query_completed']and a['terminal_outcome']=='authentic_completed_query'
spec=importlib.util.spec_from_file_location('_portable_query_contract',r/'source/query_contract.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=json.loads((r/'terminal/run/query/state.json').read_text());receipt=m.validate_result(r/'terminal/run/query',s['identity'])
assert receipt==a['native_validation']['receipt']
assert receipt['declared_semantics_supported'] is False and receipt['geometry_accuracy_within_proposed_bounds'] is False
i=json.loads((r/'interpretation/RESULT.json').read_text());assert i['changed_bounds'] is False and i['changed_native_bytes'] is False
assert i['decoded_diagnostic']['declared_semantics_supported'] is True and i['decoded_diagnostic']['geometry_accuracy_within_proposed_bounds'] is True
assert i['decoded_diagnostic']['bounds']==i['original_replay']['bounds']
print('PASS',len(actual),'payloads; original query flags preserved and separate explicit decode recorded')
