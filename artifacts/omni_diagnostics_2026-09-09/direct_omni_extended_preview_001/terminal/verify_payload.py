"""Read-only portable exact terminal/raw/media receipt verification."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

manifest=root/'FREEZE_SHA256.json';expected=json.loads(manifest.read_text())
assert not any(p.is_symlink() for p in root.rglob('*'))
actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
assert actual==expected,'Frozen inventory differs'
a=json.loads((root/'remote_audit.json').read_text());raw={p.relative_to(root).as_posix():sha(p) for name in ('run','forecast_pause') for p in (root/name).rglob('*') if p.is_file()}
assert raw==a['raw_payloads'] and len(raw)==29
v=json.loads((root/'run/recording/video.json').read_text());c=json.loads((root/'run/campaign.json').read_text());probe=json.loads((root/'ffprobe.json').read_text())
assert c['status']=='completed' and c['terminal_inputs_unchanged'] and c['post_exit_original_inputs_reverified']
assert a['recording_completion_replay_passed'] and a['all_inputs_and_selected_pilot_verified'] and len(a['owned_names_IDs_absent'])==2
assert v['complete'] and v['frames']==950 and v['recorded_control_steps']==1900 and not v['terminal_event'] and not v['error']
assert sha(root/'run/recording/rollout.mp4')==v['video_sha256'] and sha(root/'run/recording/trace.npz')==v['trace_sha256']
s=probe['streams'][0];assert (s['nb_read_frames'],s['r_frame_rate'],s['width'],s['height'],float(probe['format']['duration']))==('950','25/1',1280,720,38.)
seal=json.loads((root/'run/recording/pre_shutdown_integrity.json').read_text());recording=root/'run/recording'
sealed={p.relative_to(recording).as_posix():sha(p) for p in recording.rglob('*') if p.is_file() and p.name!='pre_shutdown_integrity.json'}
assert seal['passed'] and seal['scope']=='before_native_app_close' and seal['process_exit_verified'] is False and sealed==seal['output_hashes']
assert v['stage2_complete'] is False and v['qualification_performed'] is False
print(json.dumps({'passed':True,'files':len(actual),'raw_files':len(raw),'freeze_sha256':sha(manifest),'scope':'Completed historical C-study CAPS500 recording; no Stage2 or new-CAD qualification'}))
