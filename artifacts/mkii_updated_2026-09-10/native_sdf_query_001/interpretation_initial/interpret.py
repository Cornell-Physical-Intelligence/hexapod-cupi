"""Explicit CPU interpretation of exact saved native tuples; never rewrite acquisition."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
sys.dont_write_bytecode=True
repo=Path(__file__).resolve().parents[2]
probe=repo/'tmp/canonical_sdf_distance_probe_001'
run=repo/'tmp/canonical_native_query_terminal_001/run/query/query'
output=Path(__file__).resolve().parent/'RESULT.json'
if output.exists():raise ValueError('Use a new output version')
sys.path.insert(0,str(probe))
spec=importlib.util.spec_from_file_location('_original_native_query_replay',probe/'replay_probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_score=m.score;captured=[]
def capture(raw,repeated,fixture,world,link):
 captured.append((np.asarray(raw).copy(),np.asarray(repeated).copy(),fixture,world,link))
 return original_score(raw,repeated,fixture,world,link)
m.score=capture
state='3a73ee557a6155a12ee7a10162f406d2dfa0b60ac199d7db71655b2d1d0b7af4'
original=m.replay(run,state)
assert original['raw_verified'] and original['recorded_score_exact_replay']
assert len(captured)==1
raw,repeated,fixture,world,link=captured[0]
# Actual native hypothesis ranking identifies (gradient_x, gradient_y, gradient_z, distance).
# Decode once to the unchanged scorer's (distance, gradient_x, gradient_y, gradient_z).
permutation=[3,0,1,2]
corrected=original_score(raw[...,permutation],repeated[...,permutation],fixture,world,link)
assert corrected['bounds']==original['bounds']
result={'schema':'canonical_native_sdf_tuple_interpretation_v1','raw_state_sha256':state,
 'original_report_sha256':hashlib.sha256((run/'report.json').read_bytes()).hexdigest(),
 'original_exact_replay':original,'decoded_diagnostic':corrected,
 'explicit_native_layout':{'distance_channel':3,'gradient_channels':[0,1,2],'sign':1,'distance_scale':1.,'query_frame':'source shape local'},
 'decoded_scorer_layout':['distance','gradient_x','gradient_y','gradient_z'],
 'scope':'CPU reinterpretation of unchanged actual native data. Original failed declared-layout result preserved. Bounds and geometry unchanged; source-mesh agreement is not hardware/contact/support admission.',
 'changed_native_bytes':False,'changed_bounds':False,'native_rerun':False,'contact_admitted':False,'standing_admitted':False,'training_allowed':False}
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'original_exact_replay':original['recorded_score_exact_replay'],'decoded_semantics':corrected['declared_semantics_supported'],'decoded_geometry':corrected['geometry_accuracy_within_proposed_bounds'],'native_layout':result['explicit_native_layout'],'rms_m':corrected['declared']['distance_rms_m'],'maximum_distance_error_m':max(x['distance_abs_max_m']for x in corrected['regions'].values()),'checks':corrected['semantics_checks']},indent=2))
