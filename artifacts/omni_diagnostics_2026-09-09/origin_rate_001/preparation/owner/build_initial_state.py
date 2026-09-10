"""Select the audited worst009 standing replica before translated trials."""
from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent
raw=P.parent/'reference_physics_results_009/run/standing'
review=json.loads((P.parent/'reference_substep_actual009_review/extra_measurements.json').read_text())
rows=review['standing']['post_settle_joints']['per_environment_joint']
selected=sorted(rows,key=lambda r:(-abs(r['integral_minus_angle_delta_rad']['trapezoid']),r['environment'],r['joint']))[0]
e=selected['environment'];state=json.loads((raw/'state.json').read_text())
with np.load(raw/'physics_substeps.npz') as d:
 names=d['joint_names'].tolist();initial={k:d[k][0,e].tolist() for k in ['root_link_position_world_m','root_link_quaternion_world_xyzw','root_com_velocity_world_mps','root_link_velocity_world_mps','root_angular_velocity_world_rad_s','joint_position_rad','joint_velocity_rad_s']}
 # Actual reset root is exactly the grid origin plus common plate height.
 origin=initial['root_link_position_world_m'][:2];initial['root_link_position_relative_m']=[0.,0.,initial['root_link_position_world_m'][2]]
 initial['joint_target_rad']=state['startup_reference']['randomized_start_target_rad'][e]
 assert np.array_equal(initial['joint_target_rad'],initial['joint_position_rad'])
 assert not np.any(initial['root_com_velocity_world_mps']) and not np.any(initial['root_angular_velocity_world_rad_s'])
result=dict(schema='matched_origin_initial_state_v1',selected_environment=e,selected_joint=selected['joint'],selection='Maximum absolute009 settled q-minus-integrated-qdot mismatch; stable replica/joint tie-break',selected_discrepancy_rad=selected['integral_minus_angle_delta_rad']['trapezoid'],joint_names_runtime=names,source_grid_origin_xy_m=origin,initial=initial,source_manifest_sha256=state['identity']['source_manifest_sha256'],inputs={str(f.relative_to(raw)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [raw/'physics_substeps.npz',raw/'state.json']},quaternion_convention='XYZW',root_velocity_write_semantics='COM linear velocity and angular velocity in world frame',cold_reset_only=True,native_solver_cache_copied=False)
output=P/'origin_initial_state.json'
if output.exists():raise FileExistsError(output)
output.write_text(json.dumps(result,indent=2)+'\n');print(e,selected['joint'],hashlib.sha256(output.read_bytes()).hexdigest())
