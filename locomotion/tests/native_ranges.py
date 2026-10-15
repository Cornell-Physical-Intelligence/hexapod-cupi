"""Freeze observed native ranges before reading candidate measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from locomotion.tools.checkpoint import numbers, measurements, freeze, compare


def read_run(root, role):
    from locomotion.tools.safety import verify_tree
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
    parser.add_argument('command', choices=('freeze', 'compare', 'certify'))
    parser.add_argument('--baseline', type=Path, action='append', default=[])
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--ranges', type=Path)
    parser.add_argument('--ranges-sha256')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--source', type=Path)
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
        if args.command == 'certify':
            import locomotion.tools.checkpoint as compat
            from locomotion.tools.sources import verify_runtime
            from locomotion.tests.native_parity import verify_weights
            if not value['pass']:
                raise ValueError('Cannot certify failed native parity')
            verify_weights(args.checkpoint)
            runtime = verify_runtime(args.source)
            recorded_source = args.candidate.parents[1]/'source'
            candidate_runtime = verify_runtime(recorded_source)
            harness = {'locomotion/tests/__init__.py', 'locomotion/tests/native_parity.py'}
            production = {name: digest for name, digest in candidate_runtime['files'].items() if name not in harness}
            if production != runtime['files']:
                raise ValueError('Certified production sources differ from the tested candidate')
            proof = json.loads((recorded_source/'PARITY.json').read_text())
            if proof.get('ranges_sha256') != args.ranges_sha256:
                raise ValueError('Candidate did not pin these baseline ranges before replay')
            record = json.loads(args.checkpoint.with_suffix('.json').read_text())
            args.output.mkdir(exist_ok=False)
            (args.output/'ranges.json').write_bytes(args.ranges.read_bytes())
            (args.output/'summary.json').write_text(json.dumps(candidate, indent=2, allow_nan=False)+'\n')
            receipt = {'schema': compat.SCHEMA, 'mode': 'evaluate', 'eval_scope': 'probes',
                       'checkpoint_sha256': compat.CHECKPOINT_SHA, 'declaration_sha256': compat.DECLARATION_SHA,
                       'original_identity': record['identity'], 'candidate_runtime_sources': runtime,
                       'files': {name: sha(args.output/name) for name in ('ranges.json', 'summary.json')}}
            path = args.output/'receipt.json'
            path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
            compat.read_receipt(path, sha(path))
            print(json.dumps({'receipt': str(path), 'sha256': sha(path)}))
            return 0
    with args.output.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'pass': value.get('pass')}))
    return 1 if value.get('pass') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
