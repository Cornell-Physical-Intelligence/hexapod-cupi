"""Check checkpoint declarations, freeze native probe ranges and read the compatibility receipt.

A compatibility receipt binds one recorded checkpoint to the current package class
paths. Its native probe evidence must lie inside the ranges of three baseline replays.
"""
import hashlib
import json
import math
from pathlib import Path

from contracts.release import COMPATIBILITY_KEYS

LEARNER_DEFAULTS = {'learner': 'ppo', 'networks': 'mlp', 'motion_prior': False}
LEARNER_KEYS = ('learner', 'networks', 'motion_prior', 'learner_source_files', 'amp_dataset')
# The receipt binds one recorded checkpoint and its declaration by hash.
SCHEMA = 'hexapod_checkpoint_compatibility_v1'
CHECKPOINT_SHA = 'bda143d766cc0a2b35cd7f85d5c11871325287a7a3b08484fd9f6ac64dace6a9'
DECLARATION_SHA = '8959def5b9c758f192e8447126620d7a65d2e72a4ac9daf87bbc66cbb26158c3'
# Class paths in the recorded learner configuration, mapped to the current package classes.
CLASS_NAMES = {
    'locomotion.rate_schedule:CappedRatePPO': 'locomotion.rl.ppo:CappedRatePPO',
    'locomotion.rate_schedule:CommandBootstrapPPO': 'locomotion.rl.ppo:CommandBootstrapPPO',
    'locomotion.action_distribution:BoundedMeanGaussian': 'locomotion.rl.ppo:BoundedMeanGaussian',
    'locomotion.action_distribution:CorrelatedBoundedMeanGaussian': 'locomotion.rl.ppo:CorrelatedBoundedMeanGaussian',
}
SOURCE_KEYS = {'schema', 'physics_source_files', 'adapter_sha256', 'entry_sha256'}


def physics_config(cfg, metadata):
    """The physics identity that standing admission, training, evaluation and native replay all declare."""
    return {'physics_dt': cfg.physics_dt, 'decimation': cfg.decimation,
            'spacing_m': cfg.spacing_m, 'target_slew_rad': cfg.target_slew_rad, 'action_scale_rad': cfg.action_scale_rad,
            'solver_position_iterations': 32, 'solver_velocity_iterations': 0,
            'floor': cfg.floor, 'floor_contact_offset_m': cfg.floor_contact_offset_m,
            'floor_rest_offset_m': cfg.floor_rest_offset_m, 'filter_collisions': True, 'material_friction': [1., 1.],
            'restitution': 0., 'external_forces_every_iteration': True,
            'neutral_joint_position_rad': metadata['nominal_joint_position_rad'],
            'reset_root_height_m': metadata['reset_root_height_m']}


def checkpoint_reward_version(record):
    """The reward a checkpoint was trained with; checkpoints from before the flag trained version 1."""
    version = record['identity'].get('reward_version', '1')
    if version not in ('1', '2', '3', '4'):
        raise ValueError('Unknown checkpoint reward version: '+str(version))
    return version


def json_shape(value):
    """The value as a checkpoint record stores it: JSON turns tuples into lists."""
    return json.loads(json.dumps(value, allow_nan=False))


def require_learner_identity(record, identity):
    """Reject a checkpoint whose learner, networks or demonstration bank differ; older records mean stock PPO."""
    recorded = record['identity']
    for key in LEARNER_KEYS:
        if recorded.get(key, LEARNER_DEFAULTS.get(key)) != json_shape(identity.get(key, LEARNER_DEFAULTS.get(key))):
            raise ValueError('Checkpoint learner, networks or demonstration bank differs: '+key)


def require_learner_configuration(record, identity, config):
    """Reject a checkpoint trained with other RSL-RL sources or another learner configuration."""
    recorded = record['identity']
    if (recorded['upstream_source_files'] != identity['upstream_source_files']
            or recorded['ppo_config'] != json_shape(config)):
        raise ValueError('Checkpoint learner dependency or configuration differs')


def require_embedded_declaration(infos, record):
    """The declaration pickled inside the checkpoint must equal its JSON record."""
    if json_shape(infos) != {'identity': record['identity'], 'updates': record['updates']}:
        raise ValueError('Embedded checkpoint declaration differs')


def require_checkpoint_identity(record, identity):
    recorded = record['identity']
    if any(recorded[key] != identity[key] for key in COMPATIBILITY_KEYS):
        raise ValueError('Checkpoint model, physics, seed or implementation differs')
    if identity['schema'] == 'hexapod_locomotion_ppo_v2':
        if recorded.get('runtime_sources') != identity.get('runtime_sources'):
            raise ValueError('Checkpoint runtime dependencies differ')
    require_learner_identity(record, identity)


def configuration(value):
    if isinstance(value, dict):
        return {key: CLASS_NAMES.get(item, item) if key == 'class_name' and isinstance(item, str)
                else configuration(item) for key, item in value.items()}
    if isinstance(value, list):
        return [configuration(item) for item in value]
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(value, path=''):
    if isinstance(value, dict):
        return {key: item for name, child in value.items() for key, item in numbers(child, path+'/'+name).items()}
    if isinstance(value, list):
        return {key: item for i, child in enumerate(value) for key, item in numbers(child, path+'/'+str(i)).items()}
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError('Nonfinite native metric: '+path)
        return {path: value}
    return {}


def measurements(summary):
    rows = summary['results']
    if (summary['diagnostic_screens_passed'] is not True or summary['missing_probe_cases']
            or summary['failed_probe_cases'] or len(rows) != 13
            or len({row['case_id'] for row in rows}) != 13 or not all(row['pass'] for row in rows)):
        raise ValueError('Require 13/13 unchanged native probes')
    values, gates = {}, {}
    for row in rows:
        name = row['case_id']
        values.update(numbers(row['metrics'], name+'/metrics'))
        for key, check in row['checks'].items():
            values.update(numbers(check.get('value'), name+'/checks/'+key))
            gates[name+'/'+key] = {field: value for field, value in check.items() if field != 'value'}
    values.update(numbers(summary.get('loads', {}), 'loads'))
    return values, gates


def freeze(summaries):
    if len(summaries) != 3:
        raise ValueError('Freeze exactly three baseline replays')
    measured = [measurements(summary) for summary in summaries]
    values, gates = measured[0]
    identity = {key: summaries[0]['allocation'][key] for key in ('checkpoint_sha256', 'evaluator_sha256')}
    if any(set(other) != set(values) or other_gates != gates for other, other_gates in measured[1:]):
        raise ValueError('Baseline metric schema or gate definitions differ')
    if any(any(summary['allocation'][key] != value for key, value in identity.items()) for summary in summaries):
        raise ValueError('Baseline weights or scorer differ')
    return {'schema': 'hexapod_refactor_ranges_v1', 'identity': identity, 'gates': gates,
            'ranges': {key: [min(row[0][key] for row in measured), max(row[0][key] for row in measured)]
                       for key in sorted(values)}, 'baseline_count': 3,
            'scope': 'Observed baseline ranges; no statistical coverage claim or changed qualification gate.'}


def compare(limits, summary):
    failures = []
    try:
        if limits['schema'] != 'hexapod_refactor_ranges_v1' or limits['baseline_count'] != 3:
            raise ValueError('Invalid frozen comparison ranges')
        values, gates = measurements(summary)
        if gates != limits['gates']:
            failures.append('Gate definitions differ')
        if any(summary['allocation'][key] != value for key, value in limits['identity'].items()):
            failures.append('Weights or scorer differ')
        if set(values) != set(limits['ranges']):
            failures.append('Metric schema differs')
        for key, bounds in limits['ranges'].items():
            if key in values and not bounds[0] <= values[key] <= bounds[1]:
                failures.append({'metric': key, 'candidate': values[key], 'baseline_range': bounds})
    except (ValueError, KeyError) as error:
        failures.append(str(error))
    return {'schema': 'hexapod_refactor_comparison_v1', 'pass': not failures, 'failures': failures,
            'stage2_complete': False}


def read_compatibility_receipt(path, digest):
    """Verify the receipt bytes, its two evidence files and the native parity they record."""
    path = Path(path)
    if sha(path) != digest:
        raise ValueError('Compatibility receipt hash differs')
    receipt = json.loads(path.read_text())
    if (receipt.get('schema') != SCHEMA or receipt.get('mode') != 'evaluate' or receipt.get('eval_scope') != 'probes'
            or receipt.get('checkpoint_sha256') != CHECKPOINT_SHA
            or receipt.get('declaration_sha256') != DECLARATION_SHA):
        raise ValueError('Unsupported checkpoint compatibility receipt')
    if set(receipt.get('files', {})) != {'ranges.json', 'summary.json'}:
        raise ValueError('Compatibility evidence files differ')
    if 'certified_physics_config' in receipt and not isinstance(receipt['certified_physics_config'], dict):
        raise ValueError('Compatibility certified physics must be a configuration')
    evidence = {}
    for name, expected in receipt['files'].items():
        entry = path.parent/name
        if entry.is_symlink() or sha(entry) != expected:
            raise ValueError('Compatibility evidence hash differs: ' + name)
        evidence[name] = json.loads(entry.read_text())
    if evidence['ranges.json']['identity']['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Compatibility evidence checkpoint differs')
    summary = evidence['summary.json']
    if (len(summary.get('loads', {})) != 13
            or any(row.get('status') != 'available' for row in summary['loads'].values())
            or len(evidence['ranges.json'].get('baseline_provenance', [])) != 3
            or summary['allocation']['evaluator_sha256'] != receipt['candidate_runtime_sources']['files'].get('locomotion/evaluation/scoring.py')):
        raise ValueError('Compatibility provenance or load evidence differs')
    if not compare(evidence['ranges.json'], evidence['summary.json'])['pass']:
        raise ValueError('Compatibility native parity failed')
    return receipt


def require_compatibility_identity(receipt, record, identity):
    """Accept the candidate runtime of the receipt when the model, physics, inputs, seed and learner match.

    A receipt that certifies a physics configuration binds its native parity evidence to that configuration;
    the loader then requires that configuration instead of the checkpoint's recorded one.
    """
    if record['identity'] != receipt['original_identity']:
        raise ValueError('Compatibility original identity differs')
    if record.get('checkpoint_sha256') != CHECKPOINT_SHA:
        raise ValueError('Compatibility checkpoint differs')
    if identity.get('runtime_sources') != receipt['candidate_runtime_sources']:
        raise ValueError('Compatibility candidate runtime differs')
    certified = receipt.get('certified_physics_config', record['identity']['physics_config'])
    if identity['physics_config'] != certified:
        raise ValueError('Compatibility certified physics differs')
    if any(record['identity'][key] != identity[key] for key in COMPATIBILITY_KEYS if key not in SOURCE_KEYS | {'physics_config'}):
        raise ValueError('Compatibility model, physics, input or seed differs')
    require_learner_identity(record, identity)


def require_compatibility_configuration(record, identity, config):
    """Compare the learner configuration after mapping recorded class paths to current classes."""
    recorded = record['identity']
    if (recorded['upstream_source_files'] != identity['upstream_source_files']
            or configuration(recorded['ppo_config']) != json_shape(config)):
        raise ValueError('Compatibility learner dependency or configuration differs')
