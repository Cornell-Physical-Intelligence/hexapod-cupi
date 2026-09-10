"""Fetch each raw byte once; verify resumed files and inventory again afterwards."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REMOTE = '/home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_cold_001'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def remote_inventory():
    raw = subprocess.check_output(['ssh', 'spark', 'python3', '-'], input=(HERE / 'remote_inventory.py').read_bytes())
    return raw, json.loads(raw)

def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(x in ('..', '.') for x in p.parts) or str(p) != name:
        raise ValueError('Unsafe inventory path')
    if any(not (c.isalnum() or c in '/._-') for c in name): raise ValueError('Unexpected remote filename')
    return p

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-bytes', type=int, default=512 << 20)
    args = p.parse_args(); out = args.output.resolve()
    if out.is_symlink(): raise ValueError('Output symlink')
    raw, before = remote_inventory()
    if before['run'] != REMOTE or before['bytes'] > args.max_bytes: raise ValueError('Wrong or excessive raw allocation')
    out.mkdir(parents=True, exist_ok=True); run = out / 'run'; run.mkdir(exist_ok=True)
    if run.is_symlink() or any(x.is_symlink() for x in out.rglob('*')): raise ValueError('Local symlink')
    remaining = sum(v['bytes'] for k, v in before['files'].items() if not (run / str(safe_relative(k))).exists())
    if shutil.disk_usage(out).free < remaining + (256 << 20): raise ValueError('Insufficient disk reserve; no transfer started')
    receipt = out / 'remote_inventory_before.json'
    if receipt.exists() and json.loads(receipt.read_text()) != before: raise ValueError('Existing terminal receipt differs')
    if not receipt.exists(): receipt.write_bytes(raw)
    for name, item in before['files'].items():
        target = run / str(safe_relative(name)); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != item['bytes'] or sha(target) != item['sha256']: raise ValueError('Existing raw file mismatch: ' + name)
            continue
        partial = target.with_name(target.name + '.partial')
        if partial.exists(): raise ValueError('Interrupted partial transfer requires explicit owner handling')
        subprocess.run(['scp', 'spark:' + REMOTE + '/' + name, str(partial)], check=True)
        if partial.stat().st_size != item['bytes'] or sha(partial) != item['sha256']: raise ValueError('Transferred raw mismatch: ' + name)
        partial.rename(target)
    raw_after, after = remote_inventory(); (out / 'remote_inventory_after.json').write_bytes(raw_after)
    if before != after: raise ValueError('Terminal outputs changed during fetch')
    actual = {x.relative_to(run).as_posix() for x in run.rglob('*') if x.is_file()}
    if actual != set(before['files']): raise ValueError('Unexpected local raw inventory')
    result = {'all_raw_hashes_matched': True, 'files': len(actual), 'bytes': before['bytes'],
              'remote_before_sha256': sha(receipt), 'remote_after_sha256': sha(out / 'remote_inventory_after.json'),
              'source_asset_cleanup_restoration_audit': 'Separate root-owned proof required'}
    (out / 'local_raw_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__': main()
