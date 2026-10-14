"""Freeze observed native ranges before reading candidate measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path


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


def read_run(root, role):
    root = Path(root)
    state = json.loads((root/'state.json').read_text())
    receipt = json.loads((root/'parity_receipt.json').read_text())
    job = json.loads((root.parent/'jobs/standing.json').read_text())
    if (state['status'] != 'completed' or state['errors'] or receipt['role'] != role
            or receipt['exit_code'] != 0 or job['status'] != 'completed' or not job['cleanup_checked']):
        raise ValueError('Native attempt or cleanup did not complete')
    summary = json.loads((root/'evaluation/summary.json').read_text())
    loads = sorted((root/'evaluation').glob('batch_*/force_metrics.json'))
    if len(loads) != 13:
        raise ValueError('Require contact and motor loads from all 13 probes')
    summary['loads'] = {}
    for path in loads:
        value = json.loads(path.read_text())
        if value['status'] != 'available':
            raise ValueError('Incomplete load evidence')
        summary['loads'][path.parent.name] = value
    summary['provenance'] = {'run': str(root), 'state_sha256': sha(root/'state.json'),
                             'summary_sha256': sha(root/'evaluation/summary.json'),
                             'receipt_sha256': sha(root/'parity_receipt.json')}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'compare'))
    parser.add_argument('--baseline', type=Path, action='append', default=[])
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--ranges', type=Path)
    parser.add_argument('--ranges-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        baseline = [read_run(path, 'baseline') for path in args.baseline]
        value = freeze(baseline)
        value['baseline_provenance'] = [row['provenance'] for row in baseline]
    else:
        if sha(args.ranges) != args.ranges_sha256:
            raise ValueError('Frozen comparison ranges changed')
        candidate = read_run(args.candidate, 'candidate')
        value = compare(json.loads(args.ranges.read_text()), candidate)
        value.update(ranges_sha256=args.ranges_sha256, candidate_provenance=candidate['provenance'])
    with args.output.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'pass': value.get('pass')}))
    return 1 if value.get('pass') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
