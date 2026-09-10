from pathlib import Path
import hashlib,json,runpy
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(p,name,bound=None):
 if bound is not None:assert sha(p/name)==bound
 m=json.loads((p/name).read_text());assert not any(f.is_symlink() for f in p.rglob('*'))
 assert {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f!=p/name}==m
 return len(m)
n=verify(root,'BUNDLE_SHA256.json')
verify(root/'guard','FREEZE_SHA256.json','2bb6fbbf9b404c05bfff625a715b19c43f60eead679e57ff8786a024188c1d80')
verify(root/'preparation','BUNDLE_SHA256.json','fcd31c767fd29dcc2aaaaf1355223b48991309380d2fd5e81e847427fc612f9a')
runpy.run_path(str(root/'preparation/verify_payload.py'),run_name='__main__')
print(json.dumps({'outer_payloads_verified':n,'scope':'preparation and dispatch only'},indent=2))
