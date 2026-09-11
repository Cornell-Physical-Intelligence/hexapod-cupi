"""Verify the handoff archive without running stored research or operator code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'BUNDLE_SHA256.json').read_text())
    assert manifest['schema'] == 'immutable_public_payload_map_v1'
    pins = manifest['files']
    files = {str(p.relative_to(ROOT)): p for p in ROOT.rglob('*') if p.is_file()}
    assert set(files) == set(pins) | {'BUNDLE_SHA256.json'}
    assert not any(p.is_symlink() for p in ROOT.rglob('*'))
    for name, expected in pins.items():
        assert digest(files[name]) == expected, name

    inventory = json.loads((ROOT / 'PREPARATION_INVENTORY.json').read_text())
    assert inventory['research_paused'] and not inventory['runtime_adopted']
    count = 0
    for item in inventory['bundles']:
        folder = ROOT / item['snapshot_path']
        actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
        assert actual == set(item['files'])
        for name, record in item['files'].items():
            path = folder / name
            assert path.stat().st_size == record['size_bytes']
            assert digest(path) == record['sha256']
            count += 1
        if item['original_freeze_sha256']:
            freeze = folder / 'FREEZE_SHA256.json'
            assert digest(freeze) == item['original_freeze_sha256']
            raw = json.loads(freeze.read_text())
            original = raw['files'] if 'schema' in raw else raw
            assert actual == set(original) | {'FREEZE_SHA256.json'}
            for name, expected in original.items():
                assert digest(folder / name) == expected

    automation = json.loads((ROOT / 'operations/continuation_paused.json').read_text())
    assert automation['status'] == 'PAUSED'
    pause = json.loads((ROOT / 'operations/spark_pause_verified.json').read_text())
    assert pause['marker']['research_paused']
    assert pause['marker']['external_automation_reservation_released'] is False
    final = json.loads((ROOT / 'operations/spark_final.json').read_text())
    assert not final['gpu_compute_processes']
    assert not final['active_research_units']
    assert not final['hexapod_timers']
    assert final['reservation']['queue_lock_held']
    assert final['reservation']['masked_user_units'] == 31
    print(json.dumps({'verified': True, 'preparation_directories': len(inventory['bundles']),
                      'preparation_files': count, 'public_payloads': len(pins),
                      'research_paused': True, 'native_calls': 0,
                      'runtime_adopted': False, 'remote_calls': 0}))


if __name__ == '__main__':
    main()
