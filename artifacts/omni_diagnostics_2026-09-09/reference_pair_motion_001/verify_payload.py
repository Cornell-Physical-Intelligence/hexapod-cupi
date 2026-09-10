"""Portable immutable terminal verification; actual rejected gate is retained."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,m):
 for f,d in m.get('files',m).items():
  p=(root/f).resolve();assert root.resolve() in p.parents and sha(p)==d,f
b=json.loads((H/'BUNDLE_SHA256.json').read_text());verify(H,b)
a=json.loads((H/'remote_audit.json').read_text());verify(H,a['raw_payloads'])
for folder,name in [('preparation','BUNDLE_SHA256.json'),('actual_review','FREEZE_SHA256.json'),('outer_guard','FREEZE_SHA256.json')]:verify(H/folder,json.loads((H/folder/name).read_text()))
assert a['source_files']==946 and a['source_unchanged'] and a['admitted_asset_files']==550 and a['admitted_assets_unchanged']
assert len(a['owned_containers_absent'])==4 and 'InvocationID=aeb33fdfce794cfca50e4227ebfada1d' in a['unit']
assert a['pause_restoration']==json.loads((H/'forecast_pause/restored.json').read_text())
state=json.loads((H/'run/paired_forward/state.json').read_text());g=state['gate']
assert state['status']=='rejected' and not g['passed'] and state['control_steps']==2400
assert g['independent_forward_motion']['displacement_integral_difference_m']>.005
assert g['complete400Hz']['passed'] and g['completed_pairs']==11 and g['generator_confirmed_touchdowns']==22
assert g['full_state_numerical_replay']['poststep_checks_independently_replayed']==2200
assert g['final_quiet_stop_window']['pass']
source=json.loads((H/'preparation/source_hashes.json').read_text());assert sha(H/'portable_replay/omni_diagnostics.py')==source['tools/omni_diagnostics.py']
print(json.dumps({'terminal_payloads_verified':len(b),'raw_payloads_verified':len(a['raw_payloads']),'source946_assets550_unchanged':True,
 'owned2containers_absent_and_pause050_restored':True,'full_actual_pair_steps':11,'original5mm_rejection_preserved':True,'walking_or_PPO_admitted':False},indent=2))
