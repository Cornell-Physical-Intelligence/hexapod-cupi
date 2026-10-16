"""Prepare a test-only replay pack: the evaluation pack plus the parity harness and its declaration."""
import argparse
import json
from pathlib import Path
import shutil

from locomotion.robots.hexapod import sha
from locomotion.spark import pack
from locomotion.spark.prepare import prepare
from locomotion.spark.sources import physics_files, runtime_manifest, source_files

BASELINE_COMMIT = 'b2a6168c35156dfe0a8202201aa574b8d9e43fb1'
HARNESS = ('native_parity.py',)


def candidate(output, remote_root, *, source_commit, ranges, baseline_sources, checkpoint_sha, **options):
    """Write the probes evaluation pack, add the harness and declare the candidate role with its pinned ranges."""
    output, ranges = Path(output), Path(ranges)
    binding = prepare(output, remote_root, mode='evaluate', eval_scope='probes', num_envs=1,
                      checkpoint_sha=checkpoint_sha, **options)
    tests = output/'source/locomotion/tests'
    tests.mkdir()
    (tests/'__init__.py').write_text('')
    for name in HARNESS:
        shutil.copy2(Path(__file__).resolve().parent/name, tests/name)
    provenance = {'schema': 'hexapod_refactor_source_provenance_v1', 'baseline_commit': BASELINE_COMMIT,
                  'candidate_commit': source_commit, 'baseline_sources': baseline_sources,
                  'candidate_sources': runtime_manifest(output/'source')['files'],
                  'candidate_physics_sources': physics_files(source_files(output/'source/locomotion'),
                                                             output/'source/locomotion'),
                  'checkpoint_sha256': checkpoint_sha}
    pack.save(output/'source_provenance.json', provenance)
    pack.save(output/'source/PARITY.json', {
        'schema': 'hexapod_refactor_parity_v1', 'baseline_commit': BASELINE_COMMIT,
        'source_commit': source_commit, 'role': 'candidate', 'harness_commit': source_commit,
        'source_provenance_sha256': sha(output/'source_provenance.json'), 'ranges_sha256': sha(ranges)})
    packed = json.loads((output/'PACK.json').read_text())
    packed['test_only_parity'] = True
    pack.save(output/'PACK.json', packed)
    pack.refreeze(output)
    return json.loads((output/'binding.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='The admitted input declaration')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha', required=True)
    parser.add_argument('--checkpoint-declaration-sha', required=True)
    parser.add_argument('--ranges', type=Path, required=True, help='Frozen baseline ranges from native_ranges freeze')
    parser.add_argument('--baseline-sources', type=Path, required=True,
                        help='A compatibility receipt whose original identity lists the baseline sources')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--allocation-profile', default='flat_pilot_v1')
    parser.add_argument('--max-wall-seconds', type=int, default=21600)
    args = parser.parse_args()
    receipt = json.loads(args.baseline_sources.read_text())
    binding = candidate(args.output, args.remote_root, source_commit=args.source_commit, ranges=args.ranges,
                        baseline_sources=receipt['original_identity']['source_files'],
                        checkpoint_sha=args.checkpoint_sha, inputs=args.inputs, checkpoint=args.checkpoint,
                        checkpoint_declaration_sha=args.checkpoint_declaration_sha, seed=args.seed,
                        allocation_profile=args.allocation_profile, max_wall_seconds=args.max_wall_seconds)
    print(json.dumps(binding, indent=2))


if __name__ == '__main__':
    main()
