"""Validate evaluation receipts without changing historical checkpoints."""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'hexapod_checkpoint_compatibility_v1'
CHECKPOINT_SHA = 'bda143d766cc0a2b35cd7f85d5c11871325287a7a3b08484fd9f6ac64dace6a9'
DECLARATION_SHA = '8959def5b9c758f192e8447126620d7a65d2e72a4ac9daf87bbc66cbb26158c3'
CLASS_NAMES = {
    'locomotion.rate_schedule:CappedRatePPO': 'locomotion.learning.rate_schedule:CappedRatePPO',
    'locomotion.rate_schedule:CommandBootstrapPPO': 'locomotion.learning.rate_schedule:CommandBootstrapPPO',
    'locomotion.action_distribution:BoundedMeanGaussian': 'locomotion.learning.action_distribution:BoundedMeanGaussian',
    'locomotion.action_distribution:CorrelatedBoundedMeanGaussian': 'locomotion.learning.action_distribution:CorrelatedBoundedMeanGaussian',
}
SOURCE_KEYS = {'schema', 'physics_source_files', 'adapter_sha256', 'entry_sha256'}


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


def read_receipt(path, digest):
    path = Path(path)
    if sha(path) != digest:
        raise ValueError('Compatibility receipt hash differs')
    receipt = json.loads(path.read_text())
    if (receipt.get('schema') != SCHEMA or receipt.get('mode') != 'evaluate'
            or receipt.get('checkpoint_sha256') != CHECKPOINT_SHA
            or receipt.get('declaration_sha256') != DECLARATION_SHA):
        raise ValueError('Unsupported checkpoint compatibility receipt')
    if set(receipt.get('files', {})) != {'ranges.json', 'summary.json'}:
        raise ValueError('Compatibility evidence files differ')
    evidence = {}
    for name, expected in receipt['files'].items():
        entry = path.parent/name
        if entry.is_symlink() or sha(entry) != expected:
            raise ValueError('Compatibility evidence hash differs: ' + name)
        evidence[name] = json.loads(entry.read_text())
    if evidence['ranges.json']['identity']['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Compatibility evidence checkpoint differs')
    if not compare(evidence['ranges.json'], evidence['summary.json'])['pass']:
        raise ValueError('Compatibility native parity failed')
    return receipt


def require_identity(receipt, record, identity):
    from contracts.release import COMPATIBILITY_KEYS
    from .identity import require_learner_identity
    if record['identity'] != receipt['original_identity']:
        raise ValueError('Compatibility original identity differs')
    if record.get('checkpoint_sha256') != CHECKPOINT_SHA:
        raise ValueError('Compatibility checkpoint differs')
    if identity.get('runtime_sources') != receipt['candidate_runtime_sources']:
        raise ValueError('Compatibility candidate runtime differs')
    if any(record['identity'][key] != identity[key] for key in COMPATIBILITY_KEYS if key not in SOURCE_KEYS):
        raise ValueError('Compatibility model, physics, input or seed differs')
    require_learner_identity(record, identity)


def require_configuration(record, identity, config):
    from .identity import json_shape
    recorded = record['identity']
    if (recorded['upstream_source_files'] != identity['upstream_source_files']
            or configuration(recorded['ppo_config']) != json_shape(config)):
        raise ValueError('Compatibility learner dependency or configuration differs')
