"""Rebuild the bounded excerpts from frozen004 evidence; no simulation."""
from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    old=REPO/'tmp/sequential_sensor_replay_001';actual=REPO/'tmp/reference_physics_results_004/run/wave'
    if sha(old/'FREEZE_SHA256.json')!='960cc71bb4607cb33764b4ae59a64f5a9b027d7eb483fd688d99c157091bbfa9':raise ValueError('Exact reviewed replay freeze required')
    m=json.loads((old/'FREEZE_SHA256.json').read_text());m=m.get('files',m)
    if not all(sha(old/k)==v for k,v in m.items()):raise ValueError('Original frozen replay changed')
    lineage=json.loads((old/'SOURCE_SHA256.json').read_text())
    if sha(actual/'trace.npz')!=lineage['actual_motion_trace_sha256'] or sha(actual/'reference_states.json')!=lineage['actual_reference_states_sha256']:raise ValueError('Exact previously reviewed raw004 evidence required')
    d=dict(np.load(ROOT/'inputs/motion_source004.npz'));raw=np.load(actual/'trace.npz')
    for k in ['time_s','distal_contact','reference_point_world_m']:
        if not np.array_equal(d[k],raw[k],equal_nan=True):raise ValueError('Actual004 motion excerpt mismatch')
    np.savez_compressed(ROOT/'inputs/measured_contacts_source004.npz',time_s=raw['time_s'],contact_point_world_m=raw['contact_point_world_m'],contact_point_valid=raw['contact_point_valid'],distal_contact=raw['distal_contact'])
    refs=json.loads((actual/'reference_states.json').read_text());rows=[]
    for entry in refs:
        result=entry.get('result',entry.get('reset'))
        if not result['valid'][0]:continue
        time=result['target_time_s'];matches=np.flatnonzero(np.isclose(d['time_s'][:,0],time,atol=1e-9,rtol=0))
        if len(matches)!=1:raise ValueError('Reference time lacks unique recorded control')
        i=int(matches[0]);s=result['state'];active=int(d['active_swing_leg_index'][i]);swing=s['swing']
        if active>=0:
            if s['current_leg']!=['lf','lm','lr','rf','rm','rr'][active] or not np.array_equal(d['planned_footprint_centres_world_m'][i,active],swing['endpoint_world_m']):raise ValueError('Original planned footprint mismatch')
        rows.append(dict(index=i,time_s=time,mode=s['mode'],current_leg=s['current_leg'],swing=swing and {k:swing[k] for k in ['start_s','duration_s','endpoint_world_m']},landing=s['landing'] and {k:s['landing'][k] for k in ['start_s','duration_s','endpoint_world_m']}))
    (ROOT/'inputs/reference_timing_source004.json').write_text(json.dumps(rows,indent=2)+'\n')
    origin={'scope':'Bounded synthetic new-footprint lease checker; actual004 flat acquisition replay only','old_replay_freeze_sha256':sha(old/'FREEZE_SHA256.json'),'old_replay_all_files_verified':len(m),'source_motion_trace_sha256':sha(actual/'trace.npz'),'source_reference_states_sha256':sha(actual/'reference_states.json'),'valid_reference_rows_matched_to_actual_times':len(rows),'cache_lineage':'Copied frozen exact stereo visual-mesh visibility; receipt binds original actual004 motion, hypothetical mounts and exact assets; no new ray claim','files':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'inputs').glob('*'))},'runtime_files':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'runtime').glob('*.py'))}}
    (ROOT/'INPUTS_SHA256.json').write_text(json.dumps(origin,indent=2)+'\n');print('Matched valid actual004 reference rows:',len(rows))
if __name__=='__main__':main()
