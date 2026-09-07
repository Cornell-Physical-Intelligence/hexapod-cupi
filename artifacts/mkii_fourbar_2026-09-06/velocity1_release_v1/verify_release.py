#!/usr/bin/env python3
"""Verify the frozen local release and its archive without any GPU/Isaac import."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, sys, tarfile, tempfile
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'tools'))
import check_pipeline_lineages as lineage
from mkii_training_contract import identity


def digest(path):
    with Path(path).open('rb') as stream:
        return lineage.digest_stream(stream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path)
    args = p.parse_args()
    meta = json.loads((HERE/'source_release.json').read_text())
    archive = (args.archive or Path(meta['archive_local_path'])).resolve()
    if digest(archive) != meta['archive_sha256'] or archive.stat().st_size != meta['archive_bytes']:
        raise ValueError('Archive identity differs')
    if identity(ROOT) != meta['functional_identity']:
        raise ValueError('Local functional source changed')
    if lineage.CURRENT_MANIFEST != meta['manifest']:
        raise ValueError('Current release pointer changed')
    current = lineage.verify_current(ROOT)
    historical = lineage.verify_historical(ROOT)
    if current['manifest_sha256'] != meta['manifest_sha256'] or current['files_verified'] != 309:
        raise ValueError('Current manifest identity/coverage differs')
    if historical['files_verified'] != 112:
        raise ValueError('Historical verification differs')
    for name, expected in meta['prior_manifest_sha256_unchanged_from_c2af43c'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Prior manifest changed: '+name)
    for row in meta['local_ci_checks']:
        if row['pass'] is not True or digest(HERE/row['log']) != row['sha256']:
            raise ValueError('CPU test evidence changed')
    content = subprocess.check_output(['git', 'show', meta['source_commit']+':'+meta['manifest']], cwd=ROOT)
    if hashlib.sha256(content).hexdigest() != meta['manifest_sha256']:
        raise ValueError('Committed source manifest differs')
    expected = lineage.parse_manifest(content)
    expected[meta['manifest']] = meta['manifest_sha256']
    if expected != meta['archive_file_sha256']:
        raise ValueError('Frozen archive path coverage differs from committed manifest')
    actual = {}
    with tempfile.TemporaryDirectory(prefix='hexapod-velocity1-release-check-') as directory:
        extracted = Path(directory)
        with tarfile.open(archive, 'r:gz') as bundle:
            for member in bundle:
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in expected or member.name in actual:
                    raise ValueError('Unexpected archive entry: '+member.name)
                target = extracted/member.name
                if not target.resolve().is_relative_to(extracted):
                    raise ValueError('Archive path escape')
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as src, target.open('xb') as dest:
                    shutil.copyfileobj(src, dest)
                target.chmod(member.mode & 0o777)
                actual[member.name] = digest(target)
        if actual != expected or len(actual) != 310:
            raise ValueError('Archive hashes/path set differ from committed release')
        cmd = [sys.executable, str(extracted/'tools/check_pipeline_lineages.py'), 'current', '--root', str(extracted)]
        check = subprocess.run(cmd, cwd=extracted, capture_output=True, text=True)
        if check.returncode:
            raise ValueError('Extracted standalone release verification failed: '+check.stdout+check.stderr)
        extracted_result = json.loads(check.stdout)
        if (extracted_result.get('pass') is not True or extracted_result.get('files_verified') != 309
                or extracted_result.get('manifest_sha256') != meta['manifest_sha256']):
            raise ValueError('Extracted standalone checker selected a different release')
    result = {'pass': True, 'source_commit': meta['source_commit'],
              'functional_sha256': meta['functional_identity']['sha256'],
              'manifest_sha256': meta['manifest_sha256'], 'current_files_verified': 309,
              'historical_files_verified': 112, 'archive_files_verified': len(actual),
              'archive_sha256': meta['archive_sha256'], 'archive_bytes': meta['archive_bytes'],
              'preserved_prior_manifests': len(meta['prior_manifest_sha256_unchanged_from_c2af43c']),
              'standalone_extracted_current_check': {'pass': True, 'files_verified': 309,
                                                    'manifest_sha256': meta['manifest_sha256']},
              'gpu_job_started': False, 'spark_synced': False}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
