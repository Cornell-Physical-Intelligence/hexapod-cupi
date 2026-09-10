"""Portable, read-only outer and nested SHA-256 verification. No third-party imports."""
from pathlib import Path, PurePosixPath
import hashlib,json

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def safe(root,relative):
    path=PurePosixPath(relative)
    if not relative or path.is_absolute() or '..' in path.parts or str(path)!=relative:
        raise ValueError('Unsafe manifest path: '+relative)
    return root/relative

def verify_tree(root,manifest_name,expected=None):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Symbolic link in bundle: '+str(root))
    manifest=root/manifest_name
    if expected is not None and digest(manifest)!=expected:
        raise ValueError('Nested manifest changed: '+str(manifest))
    mapping=json.loads(manifest.read_text())
    if not isinstance(mapping,dict):raise ValueError('Manifest must be an object')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=manifest}
    if actual!=set(mapping):raise ValueError('Missing or unlisted payload: '+str(root))
    for relative,bound in mapping.items():
        path=safe(root,relative)
        if not isinstance(bound,str) or len(bound)!=64 or digest(path)!=bound:
            raise ValueError('Payload changed: '+str(path))
    return len(mapping)

def main():
    count=verify_tree(ROOT,'BUNDLE_SHA256.json')
    nested=json.loads((ROOT/'NESTED_INPUTS.json').read_text())
    if len(nested)!=4 or len({x['directory'] for x in nested})!=4:
        raise ValueError('Expected four distinct preserved input bundles')
    total=0
    for item in nested:
        root=safe(ROOT,item['directory'])
        size=verify_tree(root,item['manifest'],item['manifest_sha256'])
        if size!=item['payload_count']:raise ValueError('Nested payload count differs')
        total+=size
    print(json.dumps({'verified':True,'outer_payloads':count,'nested_bundles':len(nested),'original_nested_payloads':total,'scope':'Local bytes and inventories only; no execution or quality admission'},indent=2))

if __name__=='__main__':main()
