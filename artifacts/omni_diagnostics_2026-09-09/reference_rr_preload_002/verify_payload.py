"""Read-only, standard-library validation of the immutable terminal evidence."""
import argparse,hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def load(path):return json.loads(path.read_text())
def check(base,mapping):
 for name,digest in mapping.items():
  p=Path(name)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Non-relative manifest path')
  f=base/p
  if f.is_symlink() or not f.is_file() or sha(f)!=digest:raise ValueError(f'Hash mismatch: {f}')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);ap.add_argument('--assets',type=Path);ap.add_argument('--original-trace',type=Path);args=ap.parse_args()
 root=Path(__file__).resolve().parent;m=load(root/'BUNDLE_SHA256.json')
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='BUNDLE_SHA256.json'}
 if actual!=set(m):raise ValueError('Bundle inventory mismatch')
 if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symlink in payload')
 check(root,m);refs=load(root/'REFERENCED_INPUTS.json');audit=load(root/'evidence/remote_audit.json')
 check(root/'evidence',audit['raw_payloads'])
 if len(audit['raw_payloads'])!=33 or audit['source_files']!=932 or audit['admitted_asset_files']!=550:raise ValueError('Unexpected audited counts')
 if not audit['source_unchanged'] or not audit['admitted_assets_unchanged']:raise ValueError('Inputs changed')
 if audit['observed_job_count']!=2 or set(audit['recorded_job_names'])!={'standing','left_strafe'}:raise ValueError('Unexpected jobs')
 absent=audit['owned_containers_absent']
 if len(absent)!=4 or any(v['returncode']!=1 or 'no such object:' not in v['stderr'].lower() for v in absent.values()):raise ValueError('Missing name/ID absence')
 if 'ExecMainStatus=1' not in audit['unit'] or 'InvocationID=7f32b909f3b5411a8f4a57d0200519ef' not in audit['unit']:raise ValueError('Terminal unit mismatch')
 if audit['pause_restoration']['restored_unix']!=1789035350.4741075:raise ValueError('Restoration mismatch')
 for name in ['review','guard']:
  f=root/name/'FREEZE_SHA256.json'
  if sha(f)!=refs[name+'_freeze_sha256']:raise ValueError('Nested freeze mismatch')
  check(f.parent,load(f))
 source_map=root/'inputs/source_002_sha256.json'
 if sha(source_map)!=refs['source_manifest_sha256'] or audit['source_manifest_sha256']!=refs['source_manifest_sha256']:raise ValueError('Source binding mismatch')
 sources=load(source_map);assets=load(root/refs['admitted_asset_map'])
 if len(sources)!=932 or len(assets)!=550:raise ValueError('Input map mismatch')
 if args.source:check(args.source,sources)
 if args.assets:check(args.assets,assets)
 if args.original_trace and sha(args.original_trace)!=refs['original_comparison']['trace_sha256']:raise ValueError('Original trace mismatch')
 state=load(root/'evidence/run/left_strafe/state.json');campaign=load(root/'evidence/run/campaign.json')
 if state['control_steps']!=599 or state['status']!='rejected' or campaign['status']!='failed':raise ValueError('Rejected result changed')
 print(json.dumps({'verified':True,'payloads':len(m),'raw_payloads':33,'source_files':932,'assets':550,'nested_review_and_guard_exact':True,'result':'rejected; not walking/torque/Stage2 admission','external_source_checked':bool(args.source),'external_assets_checked':bool(args.assets)},indent=2))
if __name__=='__main__':main()
