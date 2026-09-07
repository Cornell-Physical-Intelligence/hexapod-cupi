import json,hashlib,tarfile,sys,datetime,importlib.util,importlib.machinery
from pathlib import Path
base=Path('/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1')
archive=base.parent/'hexapod_1600hz_metrics_release.tar.gz'
meta=json.loads((base.parent/'hexapod_1600hz_metrics_release.json').read_text())
assert meta['source_commit']=='c2af43ca0f384a4c2c7ab8f1d627f309dc78a683'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['archive_sha256']=='8cc802adc118b48ffc10777ca7a703091d51b34da0044f7f28e033043baa5461'
base.mkdir(exist_ok=False)
source=base/'source';source.mkdir()
with tarfile.open(archive) as tf:
 members=tf.getmembers()
 files=[m for m in members if m.isfile()]
 assert len(files)==309
 assert all(not m.issym() and not m.islnk() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 tf.extractall(source,filter='data')
manifest=source/meta['manifest']
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==meta['manifest_sha256']
rows=manifest.read_text().splitlines()
assert len(rows)==308
for row in rows:
 expected,relative=row.split(None,1);relative=relative.lstrip('*')
 assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==expected,relative
sys.path.insert(0,str(source/'tools'))
from mkii_training_contract import identity
contract=identity(source)
assert contract==meta['functional_identity']
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':meta['source_commit'],'functional_sha256':contract['sha256'],'manifest':meta['manifest'],'manifest_sha256':meta['manifest_sha256'],'manifest_paths':308,'archive_files':309,'archive_sha256':meta['archive_sha256'],'source':str(source),'source_files_verified':308,'gpu_job_started':False,'reservation_created':False}
(base/'staging.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
