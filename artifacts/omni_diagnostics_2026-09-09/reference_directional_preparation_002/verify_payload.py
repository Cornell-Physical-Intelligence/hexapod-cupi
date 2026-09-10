"""Portable preparation verifier. No simulator imports or remote actions."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root, manifest, expected=None):
    path = root / manifest
    if expected is not None: assert sha(path) == expected
    assert not any(p.is_symlink() for p in root.rglob('*'))
    files = {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p != path}
    assert files == json.loads(path.read_text())
    return len(files)
def main():
    total = verify(ROOT, 'BUNDLE_SHA256.json')
    assert verify(ROOT/'owner_preparation', 'BUNDLE_SHA256.json', 'b659db677718dbfe2dbef1ae500d0c493c4c28393c377d27b48039adac72d19c') == 22
    assert verify(ROOT/'owner_preparation/owner', 'FREEZE_SHA256.json', '9c50cffd4676481344d91066ef198baa634f73bddbbde342c316903db6280040') == 13
    assert verify(ROOT/'independent_review', 'FREEZE_SHA256.json', '5d0c5509d1981e0b1b263ec03d1e8176473bc033ea804485dac0be909b00c91c') == 3
    assert verify(ROOT/'guard', 'FREEZE_SHA256.json', '379862121a2776693a75c7e224df341d8e6199ae3749a329746c81073e08d1be') == 2
    p = ROOT/'owner_preparation/source_overlay/campaign_source_hashes.json'
    assert sha(p) == '7c64602cae91478aebd4b03adcc79f56084db7529fd1c54c0439d061075a1c58'
    source = json.loads(p.read_text()); assert len(source) == 930
    spec = json.loads((ROOT/'owner_preparation/RECONSTRUCTION.json').read_text())
    for name, value in spec['copy_exact_parent_then_overwrite'].items():
        assert source[name] == value == sha(ROOT/'owner_preparation/source_overlay'/name)
    assert len(spec['copy_exact_parent_then_overwrite']) == 5
    previous = json.loads((ROOT/'previous_owner_exit_preflight.json').read_text())
    assert previous['status'] == 'passed' and len(previous['previous_owned_absence']) == 4
    assert 'ActiveState=inactive' in previous['previous_unit'] and 'ExecMainStatus=0' in previous['previous_unit']
    for job in previous['previous_owned_absence']:
        assert {row['key'] for row in job['checks']} == {'container_id', 'container_name'}
        for row in job['checks']:
            assert row['returncode'] != 0 and ('no such object' in row['stderr'].lower() or 'no such container' in row['stderr'].lower())
    dispatch = json.loads((ROOT/'dispatch.json').read_text())
    assert '--unit=hexapod-reference-directional-002-20260910.service' in dispatch['command']
    assert dispatch['launched_unix'] > previous['pause043_restoration']['restored_unix']
    print(json.dumps(dict(payloads=total, source_overlay_files=5, source_files=930,
        previous_owned_names_and_IDs_absent=4, preparation_verified=True,
        physical_result_included=False, GPU_launches=0, Stage2_complete=False), indent=2))
if __name__ == '__main__': main()
