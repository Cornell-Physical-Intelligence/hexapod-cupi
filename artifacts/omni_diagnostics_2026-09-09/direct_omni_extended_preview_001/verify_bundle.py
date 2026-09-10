from pathlib import Path
import hashlib, json, subprocess, sys
sys.dont_write_bytecode = True
R = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root, name):
    m = json.loads((root/name).read_text())
    assert not any(p.is_symlink() for p in root.rglob('*'))
    assert {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p != root/name} == m
    return len(m)
n = verify(R, 'BUNDLE_SHA256.json')
for name, bound in [('terminal','6b30b8cc4e1699350e66601d8d1032850f14b5f206006a47b88d70daf3a6a29b'), ('guard','4454f8083ad0983f352b9d1e78a7ea537c21835dd53deba4dfe7ec8b2c13c413'), ('setup_review','68aa27278050a8318548ceef8087b51d8ce310635c97e6848efb27df5c83cd17')]:
    assert sha(R/name/'FREEZE_SHA256.json') == bound
    verify(R/name, 'FREEZE_SHA256.json')
subprocess.run([sys.executable, '-B', '-S', str(R/'terminal/verify_payload.py')], check=True)
assert sha(R/'terminal/run/recording/rollout.mp4') == '1cd2dab2ac5d4c981f4528017febfed0287ac0cdb9e19e2d34cc52704e7f4745'
print(json.dumps({'passed':True, 'payloads':n, 'historical_C_study':True, 'Stage2_complete':False}, indent=2))
