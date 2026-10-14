"""Freeze observed native ranges before reading candidate measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from locomotion.runs.compat import numbers, measurements, freeze, compare


def read_run(root, role):
    from locomotion.runs.reservation import verify_tree
    from locomotion.tests.native_parity import CHECKPOINT_SHA, DECLARATION_SHA
    root = Path(root)
    pack = root.parents[1]
    read = lambda path: json.loads(path.read_text())
    state = json.loads((root/'state.json').read_text())
    proof = read(pack/'source/PARITY.json')
    binding = read(pack/'binding.json')
    packed = read(pack/'PACK.json')
    job = read(root.parent/'jobs/standing.json')
    cleanup = read(root.parent/'cleanup.json')
    freeze = binding['source_freeze_sha256']
    verify_tree(pack/'source', freeze)
    if (proof['schema'] != 'hexapod_refactor_parity_v1' or proof['role'] != role
            or proof['baseline_commit'] != 'b2a6168c35156dfe0a8202201aa574b8d9e43fb1'
            or packed.get('test_only_parity') is not True
            or packed['binding_sha256'] != sha(pack/'binding.json')
            or read(root.parent/'launch_binding.json') != binding
            or state['runtime_binding']['runtime_tree_sha256'] != freeze
            or job['source_freeze_sha256'] != freeze
            or not {CHECKPOINT_SHA, DECLARATION_SHA} <= set(binding['input_files'].values())):
        raise ValueError('Native source or checkpoint provenance differs')
    # The host writes cleanup.json after rechecking source and input hashes. Kit can exit at app.close().
    if (state['status'] != 'completed' or state['errors'] or job['exit_code'] != 0
            or job['status'] != 'completed' or not job['cleanup_checked']
            or not cleanup['cleanup_checked'] or cleanup['container_id'] != job['container_id']):
        raise ValueError('Native attempt or cleanup did not complete')
    summary = json.loads((root/'evaluation/summary.json').read_text())
    if summary['allocation']['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Native checkpoint provenance differs')
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
                             'source_freeze_sha256': freeze,
                             'source_declaration_sha256': sha(pack/'source/PARITY.json'),
                             'binding_sha256': sha(pack/'binding.json'),
                             'job_sha256': sha(root.parent/'jobs/standing.json'),
                             'cleanup_sha256': sha(root.parent/'cleanup.json')}
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
