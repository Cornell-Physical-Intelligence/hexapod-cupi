"""Verify frozen CAD media and recorded frame evidence; no renderer/native execution."""
from pathlib import Path
import hashlib,json,math
r=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
assert not any(p.is_symlink()for p in r.rglob('*'))
actual={p.relative_to(r).as_posix():sha(p)for p in r.rglob('*')if p.is_file()and p!=r/'BUNDLE_SHA256.json'and '__pycache__'not in p.parts}
assert actual==read(r/'BUNDLE_SHA256.json'),'Changed public inventory'
v=read(r/'VIDEO_VERIFICATION.json');trace=read(r/'capture_trace_final.json');probe=read(r/'qa_final/ffprobe.json');inputs=read(r/'INPUTS.json')
video=r/v['selected_video'];assert v['selected_video']=='hexapod_detailed_cad_showcase_final.mp4';assert sha(video)==v['sha256']and video.stat().st_size==v['size_bytes']
assert v['no_policy']and v['no_native_physics']and v['no_Spark_execution']and not v['draft_selected']
assert trace['fps']==25 and trace['frames']==600 and len(trace['trace'])==600 and trace['errors']==[]
for i,row in enumerate(trace['trace']):
 t=i/25;assert row['time']==t and row['model_sha256']==v['model_sha256'];assert(row['parts'],row['bodies'],row['joints'])==(1753,19,18)
 assert len(row['angles'])==18
 demo=math.sin(math.pi*(t-12)/9)**2 if 12<=t<=21 else 0
 for name,q in row['angles'].items():
  expected=.22*demo if name=='lf_femur_pitch'else .28*demo if name=='lf_tibia_pitch'else 0
  assert math.isfinite(q)and math.isclose(q,expected,abs_tol=1e-14)
 b=row['projectedBounds'];assert -.9<b['minX']<b['maxX']<.9 and -.71<b['minY']<b['maxY']<.63
s=probe['streams'][0];assert(s['codec_name'],s['width'],s['height'],s['r_frame_rate'],int(s['nb_frames']))==('h264',1920,1080,'25/1',600)
assert float(probe['format']['duration'])==24 and v['full_ffmpeg_decode_exit_code']==0 and(r/'qa_final/full_decode.stderr').read_bytes()==b''
assert len(inputs['source_meshes_sha256'])==59 and inputs['active_model']['model']['sha256']==v['model_sha256']and inputs['active_model']['urdf']['sha256']==v['urdf_sha256']
for path,digest in v['capture_files_sha256'].items():assert sha(r/path)==digest
repo=next((p for p in r.parents if(p/'robot/active_model.json').is_file()),None)
if repo:
 for name,digest in {**inputs['files'],**inputs['source_meshes_sha256']}.items():assert sha(repo/name)==digest,name
print(json.dumps({'passed':True,'payloads':len(actual),'selected_video':v['selected_video'],'video_sha256':v['sha256'],'frames':600,'resolution':[1920,1080],'fps':25,'duration_s':24,'kinematic_only':True,'recorded_full_decode_verified':True,'source_files_rechecked':repo is not None},sort_keys=True))
