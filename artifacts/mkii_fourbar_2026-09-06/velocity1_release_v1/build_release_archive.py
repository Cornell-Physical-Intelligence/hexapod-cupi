#!/usr/bin/env python3
"""Build a local deployment archive from exact committed Git files; never sync or launch."""
from pathlib import Path
import argparse, datetime, hashlib, json, re, subprocess, sys, tarfile
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'packages/hexapod_core')]
import check_pipeline_lineages as lineage
from mkii_training_contract import identity
from hexapod_core.fourbar_v1 import numerical_recipe


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--archive', type=Path, required=True)
    args = p.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.source_commit+'^{commit}'], cwd=ROOT, text=True).strip()
    if commit != args.source_commit or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Require exact resolved source commit')
    archive = args.archive.resolve()
    if archive.exists() or (HERE/'source_release.json').exists():
        raise ValueError('Refusing existing archive or release evidence')
    current = lineage.verify_current(ROOT)
    historical = lineage.verify_historical(ROOT)
    contract = identity(ROOT)
    manifest = lineage.CURRENT_MANIFEST
    content = subprocess.check_output(['git', 'show', f'{commit}:{manifest}'], cwd=ROOT)
    if content != (ROOT/manifest).read_bytes():
        raise ValueError('Committed manifest differs from verified worktree manifest')
    expected = lineage.parse_manifest(content)
    if len(expected) != 309 or not set(contract['files']).issubset(expected):
        raise ValueError('Incomplete release coverage')
    for name, value in contract['files'].items():
        if expected[name] != value:
            raise ValueError('Functional identity not covered: '+name)
    expected[manifest] = hashlib.sha256(content).hexdigest()
    previous = {}
    for path in sorted((ROOT/'isaaclab/deploy').glob('*pipeline.sha256')):
        rel = path.relative_to(ROOT).as_posix()
        if rel == manifest:
            continue
        old = subprocess.check_output(['git', 'show', 'c2af43ca0f384a4c2c7ab8f1d627f309dc78a683:'+rel], cwd=ROOT)
        if path.read_bytes() != old:
            raise ValueError('Prior published manifest changed: '+rel)
        previous[rel] = hashlib.sha256(old).hexdigest()
    archive.parent.mkdir(parents=True, exist_ok=True)
    argv = ['git', 'archive', '--format=tar.gz', '--output='+str(archive), commit, '--', *sorted(expected)]
    subprocess.run(argv, cwd=ROOT, check=True)
    actual, sizes = {}, {}
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            if member.isdir():
                continue
            if not member.isfile() or member.name not in expected or member.name in actual:
                raise ValueError('Unexpected/nonregular/duplicate archive entry: '+member.name)
            stream = bundle.extractfile(member)
            with stream:
                actual[member.name] = lineage.digest_stream(stream)
            sizes[member.name] = member.size
    if actual != expected:
        raise ValueError('Archive bytes or path set differ from committed release manifest')
    checks = []
    for filename in ('lineage_tests.log.txt', 'full_suite.log.txt'):
        text = (HERE/filename).read_text()
        match = re.search(r'Ran (\d+) tests in ([\d.]+)s\n\nOK\n', text)
        if not match or 'FAILED (' in text:
            raise ValueError('CPU suite did not pass: '+filename)
        checks.append({'log': filename, 'sha256': digest(HERE/filename),
                       'tests': int(match[1]), 'seconds': float(match[2]), 'pass': True})
    result = {'schema_version': 1, 'source_commit': commit, 'functional_identity': contract,
              'manifest': manifest, 'manifest_sha256': expected[manifest], 'manifest_paths': 309,
              'archive_files': len(actual), 'archive_sha256': digest(archive),
              'archive_bytes': archive.stat().st_size, 'archive_local_path': str(archive),
              'archive_git_argv': argv, 'archive_file_sha256': actual, 'archive_file_sizes': sizes,
              'prior_manifest_sha256_unchanged_from_c2af43c': previous,
              'prepared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'local_ci_checks': checks, 'historical_lineage': historical, 'current_lineage': current,
              'numerical_recipes': {'nominal': numerical_recipe(1), 'refined': numerical_recipe(2)},
              'previous_cpu_candidate_commit': 'efe43384acf2fe4caa29ba01d7e3c1e93d642496',
              'previous_cpu_candidate_identity_superseded': '328efb07dded31e871d0b6dad3c50d949a7af798bacd8f8e6f3451ccbb7ca340',
              'github_ci_run_claimed': False, 'spark_synced': False, 'gpu_job_started': False,
              'simulation_training_admission': False, 'hardware_admission': False}
    (HERE/'source_release.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('source_commit','manifest_sha256','manifest_paths','archive_files',
                    'archive_sha256','archive_bytes','archive_local_path')}, indent=2))
    print('functional_sha256='+contract['sha256'])


if __name__ == '__main__':
    main()
