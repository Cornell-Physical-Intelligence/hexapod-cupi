from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
m=json.loads((r/'FREEZE_SHA256.json').read_text())
a={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in r.rglob('*')if p.is_file()and p!=r/'FREEZE_SHA256.json'and '__pycache__'not in p.parts}
assert a==m,'Review payload changed'
s=json.loads((r/'RESULT.json').read_text())
assert s['parity']['actual_rows_bitexact_candidate_vs_parent']==8000
assert s['negative_tests_passed']==7 and not s['physical_thresholds_changed']
assert not s['native_or_GPU_execution_by_review']
print('PASS',len(a),'CPU review payloads; candidate not natively admitted')
