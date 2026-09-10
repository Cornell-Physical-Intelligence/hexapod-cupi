"""Portable strict payload and historical evidence verification; no GPU or mutation."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,manifest,expected=None):
 p=root/manifest;m=json.loads(p.read_text())
 if expected is not None:assert sha(p)==expected
 actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and f!=p}
 assert actual==set(m) and not any(f.is_symlink() for f in root.rglob('*'))
 for f,h in m.items():assert sha(root/f)==h,f
 return len(m)
def main():
 count=verify(R,'BUNDLE_SHA256.json');r=json.loads((R/'RECONSTRUCTION.json').read_text())
 for name,spec in r['copied_frozen_bundles'].items():assert verify(R/name,spec['manifest'],spec['sha256'])==spec['payloads']
 a=json.loads((R/'remote_audit.json').read_text());assert len(a['raw_payloads'])==32
 for f,h in a['raw_payloads'].items():assert sha(R/'raw'/f)==h,f
 assert a['source_manifest_sha256']==r['source_manifest_sha256'] and a['source_files']==930 and a['admitted_asset_files']==550
 assert a['source_unchanged'] and a['admitted_assets_unchanged'] and len(a['owned_containers_absent'])==4
 for token,value in a['owned_containers_absent'].items():assert value['returncode']!=0 and ('no such object' in value['stderr'].lower() or 'no such container' in value['stderr'].lower())
 assert json.loads((R/'raw/forecast_pause/restored.json').read_text())==a['pause_restoration']
 tree=ast.parse((R/'guard/launch_guarded_remote.py').read_text());scripts=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text' and n.args and isinstance(n.args[0],ast.Constant)]
 assert len(scripts)==1 and scripts[0].encode()==(R/'raw/forecast_pause/resume_forecasting.py').read_bytes()
 state=json.loads((R/'raw/run/left_strafe/state.json').read_text());review=json.loads((R/'actual_review/report.json').read_text())
 assert state['status']=='rejected' and state['control_steps']==505 and state['gate']==review['original_gate'] and not state['gate']['passed']
 assert not (R/'raw/run/left_turn').exists() and not (R/'raw/run/forward_right_arc').exists()
 json.dumps(state,allow_nan=False)
 print(json.dumps({'payloads_verified':count,'raw_payloads_verified':32,'source930_assets550_remote_proof_verified':True,'terminal_rejected_JSON_preserved':True,'left_strafe_admitted':False,'read_only':True},indent=2))
if __name__=='__main__':main()
