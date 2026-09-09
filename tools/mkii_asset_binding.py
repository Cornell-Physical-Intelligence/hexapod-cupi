"""Bind CPU/Kit audits and live runtime to the same portable physical bundle."""
from pathlib import PurePosixPath
import json


def verify_asset_binding(bundle, cpu, kit, runtime, source_contract):
    errors = []
    files = source_contract.get('files', {})
    root = PurePosixPath(bundle['usd_path_relative'])
    expected_files = bundle['bundle_files_sha256']
    for relative, sha in expected_files.items():
        # Bundle records use paths relative to the repository.
        if files.get(relative) != sha:
            errors.append(f'Source identity does not cover selected bundle file: {relative}')
    for label, audit in (('CPU', cpu), ('Kit', kit)):
        if audit.get('pass') is not True or audit.get('errors') != []:
            errors.append(f'{label} asset audit did not pass')
        for field, expected in (('usd_root_sha256', bundle['usd_root_sha256']),
                                ('kinematic_contract_sha256', bundle['kinematics_sha256']),
                                ('closure_constraint_variant', bundle['closure_constraint_variant'])):
            if audit.get(field) != expected:
                errors.append(f'{label} asset {field} differs')
        deps = audit.get('dependencies')
        if not isinstance(deps, list) or not deps:
            errors.append(f'{label} USD dependencies missing')
            continue
        observed = {}
        for dep in deps:
            path = PurePosixPath(dep.get('path', ''))
            if path.is_absolute() or '..' in path.parts or not path.parts:
                errors.append(f'{label} unsafe USD dependency')
                continue
            relative = str(root.parent / path)
            if relative in observed:
                errors.append(f'{label} duplicate USD dependency')
            observed[relative] = dep.get('sha256')
            if expected_files.get(relative) != dep.get('sha256') or relative not in expected_files:
                errors.append(f'{label} USD dependency differs: {relative}')
        expected_deps = {path: sha for path, sha in expected_files.items()
                         if PurePosixPath(path).suffix in ('.usd', '.usda', '.usdc')}
        if observed != expected_deps:
            errors.append(f'{label} incomplete USD dependency coverage')
    if runtime.get('asset_bundle') != bundle:
        errors.append('Actual runtime asset bundle differs from audited bundle')
    for field in ('model_id', 'usd_root_sha256', 'kinematics_sha256', 'closure_constraint_variant'):
        if runtime.get(field) != bundle[field]:
            errors.append(f'Actual runtime {field} differs from audited bundle')
    if errors:
        raise ValueError('; '.join(errors))
    return {'pass': True, 'model_id': bundle['model_id'], 'usd_root_sha256': bundle['usd_root_sha256'],
            'closure_constraint_variant': bundle['closure_constraint_variant']}


def solver_runtime_equivalent(nominal, refined):
    """Only the separately validated position-iteration refinement may differ."""
    copies = []
    for runtime in (nominal, refined):
        if not isinstance(runtime, dict) or not isinstance(runtime.get('resolved_simulation'), dict):
            return False
        copied = json.loads(json.dumps(runtime, sort_keys=True, allow_nan=False))
        copied['resolved_simulation'].pop('solver_position_iterations', None)
        copies.append(json.dumps(copied, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return copies[0] == copies[1]
