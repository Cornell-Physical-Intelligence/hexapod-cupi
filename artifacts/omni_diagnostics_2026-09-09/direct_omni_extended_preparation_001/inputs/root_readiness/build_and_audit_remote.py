import hashlib, json, pathlib, subprocess

base = pathlib.Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
prep = base / 'direct_omni_train_preparation_004'
source = base / 'direct_omni_train_source_004'
cold = base / 'direct_omni_cold_source_001'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory(p, exclude):
    assert not any(x.is_symlink() for x in p.rglob('*'))
    return {x.relative_to(p).as_posix(): sha(x) for x in p.rglob('*') if x.is_file() and x.name != exclude}
freeze = prep / 'FREEZE_SHA256.json'
assert sha(freeze) == '0bdd2abf3811e68ac59e4fbc325eb5d2bb96afd2183eb84c4d37faeb86ff9e9f'
assert inventory(prep, freeze.name) == json.loads(freeze.read_text())
assert not source.exists(), 'Fresh source required; do not rebuild'
result = subprocess.run(['python3', '-B', str(prep/'build_source.py'), '--source', str(cold), '--output', str(source)], capture_output=True, text=True)
print(json.dumps({'build_returncode':result.returncode, 'build_stdout':result.stdout, 'build_stderr':result.stderr}), flush=True)
assert result.returncode == 0
pins = json.loads(result.stdout)
manifest = json.loads((source/'campaign_source_hashes.json').read_text())
assert inventory(source, 'campaign_source_hashes.json') == manifest
parent = json.loads((cold/'campaign_source_hashes.json').read_text())
assert inventory(cold, 'campaign_source_hashes.json') == parent
changed = {k: {'before': parent.get(k), 'after':v} for k,v in manifest.items() if parent.get(k) != v}
removed = sorted(set(parent)-set(manifest))
expected = {'tools/train_length_study.py', 'robot/hexapod_mkii_length_study/training_plan.json', 'source_origin.json'}
origin = json.loads((source/'source_origin.json').read_text())
expected.update('tools/'+k for k in origin['runtime_overlays'])
assert set(changed) == expected, (set(changed), expected)
assert not removed
assert sha(source/'campaign_source_hashes.json') == pins['source_manifest_sha256']
print(json.dumps({'verified':True, 'pins':pins, 'native_freeze_sha256':sha(freeze), 'native_payloads':28, 'parent_files':len(parent), 'source_files':len(manifest), 'changed':changed, 'removed':removed}, indent=2), flush=True)
