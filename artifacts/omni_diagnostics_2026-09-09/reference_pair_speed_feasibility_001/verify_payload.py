"""Read-only standard-library check for the frozen target study."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=json.loads((H/'FREEZE_SHA256.json').read_text())
 actual={str(p.relative_to(H)) for p in H.rglob('*') if p.is_file() and p.name!='FREEZE_SHA256.json'}
 if actual!=set(manifest) or any(p.is_symlink() for p in H.rglob('*')):raise ValueError('Inventory/symlink mismatch')
 for f,h in manifest.items():
  if Path(f).is_absolute() or '..' in Path(f).parts or sha(H/f)!=h:raise ValueError('Payload mismatch: '+f)
 inputs=json.loads((H/'PARENT_INPUTS_SHA256.json').read_text())
 for f,h in inputs['copied_files_sha256'].items():
  if sha(H/f)!=h:raise ValueError('Copied frozen parent changed: '+f)
 summary=json.loads((H/'SUMMARY.json').read_text())
 if len(summary['cases'])!=22 or summary['measured_physics_admission'] is not False:raise ValueError('Wrong matrix/scope')
 if sha(H/'PLAN.json')!=summary['matrix_plan_sha256'] or sha(H/'study.py')!=summary['driver_sha256']:raise ValueError('Plan/driver binding mismatch')
 for case in summary['cases']:
  p=case['parameters'];name=p['candidate_id']
  if hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()!=case['parameters_sha256']:raise ValueError('Parameter binding mismatch')
  if sha(H/'results'/(name+'.npz'))!=case['target_npz_sha256']:raise ValueError('Target array changed')
  if json.loads((H/'results'/(name+'.json')).read_text())!=case:raise ValueError('Case receipt mismatch')
 print(json.dumps({'verified':True,'payloads':len(manifest),'planned_cases':22,'completed_target_cases':sum(c['result']['completed_target_sequence'] for c in summary['cases']),'actual_physics_controls':0,'physical_admission':False},indent=2))
if __name__=='__main__':main()
