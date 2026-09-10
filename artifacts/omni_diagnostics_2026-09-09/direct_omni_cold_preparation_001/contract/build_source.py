"""Build a fresh exact-old-runtime cold baseline; never mutate the parent."""
import argparse,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
PLAN='robot/hexapod_mkii_length_study/training_plan.json'
ENTRY='tools/train_length_study.py'
PARENT_MAP='f469abdad81cab7bf63719c72cf4468b8827b802499f85a66d6b1abe8e9bd5e3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def exact_files(source,expected):
 observed={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 if observed!=set(expected):raise ValueError('Parent complete file inventory differs')
 for name,digest in expected.items():
  p=source/name
  if p.is_symlink() or sha(p)!=digest:raise ValueError('Parent payload mismatch: '+name)
def new_plan(old):
 p=json.loads(json.dumps(old))
 if p['omni']['overrides']['target_slew_rad_per_20ms']!=.03:raise ValueError('Expected historical .03 diagnostic plan')
 p['omni']['overrides']['target_slew_rad_per_20ms']=.04
 return p
def new_entry(old):
 before='from repair_length_study_inertias import repair_and_verify'
 if old.count(before)!=1:raise ValueError('Expected exact historical asset-writer import')
 return old.replace(before,'from candidate_asset_audit import audit_candidate_usd as repair_and_verify')
def build(parent,output):
 parent=parent.resolve();output=output.resolve()
 if output.exists() or output==parent or parent in output.parents:raise ValueError('Fresh output outside immutable parent required')
 inventory=read(HERE/'remote_base_inventory.json')
 if sha(parent/'campaign_source_hashes.json')!=PARENT_MAP:raise ValueError('Historical 146-file manifest mismatch')
 exact_files(parent,inventory['current_complete_files'])
 output.mkdir(parents=True)
 for name in inventory['current_complete_files']:
  dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/name,dest)
 (output/PLAN).write_text(json.dumps(new_plan(read(parent/PLAN)),indent=2)+'\n')
 (output/ENTRY).write_text(new_entry((parent/ENTRY).read_text()))
 shutil.copy2(HERE/'candidate_asset_audit.py',output/'tools/candidate_asset_audit.py')
 origin={'schema':'direct315_cold_formal004_v1','parent':str(parent),'parent_historical_manifest_sha256':PARENT_MAP,
 'parent_complete_inventory_sha256':sha(HERE/'remote_base_inventory.json'),
 'changes':{PLAN:'Only target slew .03 -> .04; same rewards, solver16/4, noise, filter, stance and commands.',ENTRY:'Only import swaps idempotent USD authoring for read-only mass/inertia/contact audit.','tools/candidate_asset_audit.py':'Previously Isaac-tested read-only audit.'},
 'scope':'Standing32x1000 then cold12-case48-replica12s diagnostics only; no training authority or formal qualification.',
 'historical03_metrics_comparable_without_intervention':False,'physics_external_forces_every_iteration':'historical installed default false; resolve/readback before run',
 'checkpoint_sha256':'1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'}
 (output/'source_origin.json').write_text(json.dumps(origin,indent=2)+'\n')
 manifest={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 (output/'campaign_source_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
 exact_files(parent,inventory['current_complete_files'])
 return {'source':str(output),'files':len(manifest),'source_manifest_sha256':sha(output/'campaign_source_hashes.json'),'plan_sha256':sha(output/PLAN),'origin_sha256':sha(output/'source_origin.json'),'entry_sha256':sha(output/ENTRY)}
if __name__=='__main__':
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.parent,a.output),indent=2))
