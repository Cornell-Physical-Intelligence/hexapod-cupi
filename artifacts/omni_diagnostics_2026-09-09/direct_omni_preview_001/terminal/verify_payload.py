"""Read-only portable verification of frozen terminal evidence, no simulator."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=root/'FREEZE_SHA256.json';expected=json.loads(manifest.read_text())
actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
assert not any(p.is_symlink() for p in root.rglob('*'))
assert actual==expected,'Frozen inventory differs'
a=json.loads((root/'remote_audit.json').read_text());raw={p.relative_to(root).as_posix():sha(p) for name in ('run','forecast_pause') for p in (root/name).rglob('*') if p.is_file()};assert raw==a['raw_payloads']
v=json.loads((root/'run/recording/video.json').read_text());c=json.loads((root/'run/campaign.json').read_text());probe=json.loads((root/'ffprobe.json').read_text())
assert c['status']=='failed' and not (root/'run/recording/final_integrity.json').exists()
assert v['complete'] and v['frames']==950 and v['recorded_control_steps']==1900
assert sha(root/'run/recording/rollout.mp4')==v['video_sha256'] and sha(root/'run/recording/trace.npz')==v['trace_sha256']
assert probe['streams'][0]['nb_read_frames']=='950' and float(probe['format']['duration'])==38.
assert a['all_inputs_and_selected_pilot_verified'] and len(a['owned_names_IDs_absent'])==2
print(json.dumps({'passed':True,'files':len(actual),'raw_files':len(raw),'freeze_sha256':sha(manifest),'scope':'Complete950-frame media; failed host finalization preserved; no policy qualification'}))
