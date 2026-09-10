"""Copy only declared small files via one read-only SSH call; no native execution."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent

def main():
    selection = json.loads((ROOT / 'RAW_SELECTION.json').read_text())
    selected = selection['selected']
    assert all(0 <= row['size_bytes'] <= 2_000_000 for row in selected.values())
    assert not (ROOT / 'curated').exists(), 'Refuse to overwrite prior fetched evidence'
    payload = base64.b64encode(json.dumps(selected).encode()).decode()
    remote = '''import base64,hashlib,io,json,pathlib,sys,tarfile
rows=json.loads(base64.b64decode(PAYLOAD))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for name,row in sorted(rows.items()):
  p=pathlib.Path(row['remote_path'])
  assert p.is_file() and not p.is_symlink()
  assert 0<=p.stat().st_size==row['size_bytes']<=2000000
  with p.open('rb') as f: data=f.read(2000001)
  assert len(data)==row['size_bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0
  archive.addfile(info,io.BytesIO(data))
'''.replace('PAYLOAD', repr(payload))
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               selection['remote_host_ssh_alias'], 'python3 -B -S -c ' + shlex.quote(remote)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (ROOT / 'FETCH_STDERR.txt').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError('Read-only selected fetch failed, exit ' + str(result.returncode))
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
        members=archive.getmembers()
        assert len(members)==len(selected) and {m.name for m in members}==set(selected)
        for member in members:
            row=selected[member.name]
            assert member.isfile() and member.size==row['size_bytes']
            data=archive.extractfile(member).read(2_000_001)
            assert len(data)==row['size_bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            dest=ROOT/row['stored_path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    receipt={'schema':'readonly_selected_fetch_v1','read_only':True,'remote_mutation':False,
             'remote_host_ssh_alias':selection['remote_host_ssh_alias'],'selected_count':len(selected),
             'selected_total_bytes':sum(r['size_bytes'] for r in selected.values()),
             'all_selected_size_and_sha256_verified':True,'remote_only_files_not_read_or_transferred':True,
             'remote_script_sha256':hashlib.sha256(remote.encode()).hexdigest(),
             'selection_sha256':hashlib.sha256((ROOT/'RAW_SELECTION.json').read_bytes()).hexdigest(),
             'audit_sha256':selection['audit_sha256'],
             'local_copy_sha256':{name:hashlib.sha256((ROOT/r['stored_path']).read_bytes()).hexdigest() for name,r in selected.items()}}
    (ROOT/'FETCH_VERIFICATION.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':True,'files':len(selected),'bytes':receipt['selected_total_bytes']}))

if __name__=='__main__':main()
