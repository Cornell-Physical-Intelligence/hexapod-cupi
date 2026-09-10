"""Portable hash/inventory and preparation-binding verification; no runtime imports."""
from pathlib import Path, PurePosixPath
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SOURCE = 'ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62'
PLAN = 'eee25089a65ec112f5eeed9c70f6310ef019ca5f077618c736fcfaa859739fc2'
ORIGIN = '3b89ac6451e85020381994dce073b2b61af9caa08c79c12c887aff0cb41fb0c2'
CHECKPOINT = '1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
BUNDLES = {
    'native_004': ('1ab678beeba6bf74978c210c4e38f13655e25728fe66e3007e3b1db5e19b6b20', 43),
    'host_003': ('df815208f340a9c8c0c2ce8c9fcfad7fd28791f745961ab714f96c0508454efc', 20),
    'guard_004': ('d7ecf21be4f9d38d9d5ba50d1d83b844b51ae82656c8e46efe51956d0630becb', 31),
    'root_readiness': ('6b5059bbf823383704a2e1900ef6509e2247f8f584b51fc654ccfae864786e9c', 10),
    'dispatch': ('e41205ab0012bd356943a531fdf703b654c871cc95a4307b15ee645bf22208fb', 6),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tree(path, expected=None, count=None):
    manifest = path / 'FREEZE_SHA256.json'
    if expected is not None:
        require(sha(manifest) == expected, f'Changed manifest: {path}')
    payloads = read(manifest)
    require(isinstance(payloads, dict), 'Manifest must be a path/hash map')
    if count is not None:
        require(len(payloads) == count, f'Payload count mismatch: {path}')
    require(not path.is_symlink() and not any(p.is_symlink() for p in path.rglob('*')),
            f'Symlink in bundle: {path}')
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*')
              if p.is_file() and p != manifest}
    require(actual == set(payloads), f'Extra/missing payloads: {path}')
    for name, bound in payloads.items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts,
                f'Invalid relative path: {name}')
        require(sha(path / name) == bound, f'Changed payload: {path / name}')
    return payloads


def constants(path):
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values


def main():
    outer = verify_tree(ROOT)
    inner = {name: verify_tree(ROOT / name, digest, n)
             for name, (digest, n) in BUNDLES.items()}
    selection = read(ROOT / 'PUBLICATION_SELECTION.json')
    require(selection['within_bundle_exclusions'] == [], 'Unexpected exclusions')
    require(selection['copied_files_including_five_inner_manifests'] == 115,
            'Copied-file count mismatch')
    for row in selection['bundles']:
        require((row['manifest_sha256'], row['payloads']) == BUNDLES[row['destination']],
                'Selection identity mismatch')
    require(len(selection['bundles']) == len(BUNDLES), 'Selection count mismatch')
    audit = read(ROOT / 'root_readiness/remote_source_audit.json')
    for key, bundle in [('native', 'native_004'), ('host', 'host_003'), ('guard', 'guard_004')]:
        require(audit[key]['inventory'] == inner[bundle], f'Remote inventory differs: {key}')
        require((audit[key]['manifest_sha256'], audit[key]['files']) == BUNDLES[bundle],
                f'Remote binding differs: {key}')
    require(audit['source']['manifest_sha256'] == SOURCE and audit['source']['files'] == 599,
            'Wrong source identity')
    inventory = audit['source']['inventory']
    require(len(inventory) == 599, 'Wrong remote inventory size')
    require(inventory['robot/hexapod_mkii_length_study/training_plan.json'] == PLAN,
            'Wrong source plan')
    require(inventory['source_origin.json'] == ORIGIN, 'Wrong source origin')
    require(audit['source_build_verified'] == {
        'source_manifest_sha256': SOURCE, 'files': 599, 'plan_sha256': PLAN,
        'origin_sha256': ORIGIN}, 'Independent build verification differs')
    require(audit['cold_parent_sha256'] ==
            '4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b',
            'Wrong cold parent')
    require(len(audit['added']) == 10 and set(audit['changed']) == {
        'robot/hexapod_mkii_length_study/training_plan.json', 'source_origin.json',
        'tools/train_length_study.py'}, 'Unexpected source delta')
    for name in audit['added']:
        require(inventory[name] == inner['native_004'][name.removeprefix('tools/')],
                f'Native/source overlay differs: {name}')
    host = constants(ROOT / 'host_003/launch_train_spark.py')
    guard = constants(ROOT / 'guard_004/launch_guarded_remote.py')
    require(host['SOURCE_MAP'] == SOURCE and host['CONTRACT_FREEZE'] == BUNDLES['native_004'][0]
            and host['ORIGINAL_CHECKPOINT'] == CHECKPOINT, 'Wrong host pins')
    require(guard['SOURCE_SHA256'] == SOURCE
            and guard['CONTRACT_SHA256'] == BUNDLES['native_004'][0]
            and guard['HOST_FREEZE_SHA256'] == BUNDLES['host_003'][0]
            and guard['HOST_SHA256'] == sha(ROOT / 'host_003/launch_train_spark.py'),
            'Wrong guard pins')
    receipt = read(ROOT / 'root_readiness/host_preflight_001.json')
    require(receipt['exit_code'] == 0 and receipt['stderr'] == '', 'Preflight failed')
    parsed = json.loads(receipt['stdout'])
    identity = parsed['identity']
    require(parsed['no_GPU'] is True and parsed['host_freeze_sha256'] == BUNDLES['host_003'][0],
            'Wrong preflight host/scope')
    require(identity['source_manifest_sha256'] == SOURCE and identity['plan_sha256'] == PLAN
            and identity['checkpoint_sha256'] == CHECKPOINT
            and identity['Stage2_complete'] is False, 'Wrong preflight identity')
    selected = identity['selection']
    require(selected['schema'] == 'direct315_quiet_priority_native_v3'
            and selected['allocation'] == 'smoke' and selected['branch'] == 'quiet_priority'
            and (selected['replicas'], selected['controls_per_update'], selected['updates']) == (32, 24, 2)
            and selected['caps']['quiet_temporal_weight'] == 1.0, 'Wrong explicit selection')
    require(read(ROOT / 'native_004/CPU_READINESS.json')['tests']['count'] == 23,
            'Wrong native test receipt')
    require(read(ROOT / 'host_003/CPU_READINESS_FINAL.json')['tests_passed'] == 28,
            'Wrong host test receipt')
    require(read(ROOT / 'guard_004/CPU_READINESS_FINAL.json')['tests_passed'] == 31,
            'Wrong guard test receipt')
    require(not list(ROOT.rglob('*.jsonl')), 'Unexpected event stream')
    print(json.dumps({'result': 'PASS', 'outer_payloads': len(outer),
                      'outer_manifest_sha256': sha(ROOT / 'FREEZE_SHA256.json'),
                      'inner_payloads': {k: len(v) for k, v in inner.items()},
                      'remote_source_payloads': len(inventory),
                      'scope': 'Preparation integrity and bounded dispatch snapshot; no terminal/performance verdict'},
                     indent=2))


if __name__ == '__main__':
    main()
