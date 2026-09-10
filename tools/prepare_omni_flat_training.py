#!/usr/bin/env python3
"""Version new omni inputs while preserving the selected C benchmark verbatim."""
import json
from pathlib import Path
from omni_flat_math import evaluation_scenarios, transition_sequence

root = Path(__file__).resolve().parents[1]
out = root / 'artifacts/omni_flat_2026-09-09/inputs'
out.mkdir(parents=True, exist_ok=True)
baseline = root / 'artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300'
plan = json.loads((baseline/'training_plan.json').read_text())
plan.pop('reference_controller')
plan.pop('evaluation_forward_speeds_mps')
plan.update(method='C full-robot flat omni PPO; direct joints, proprioceptive history actor, velocity-privileged critic',
            training_seed=157, training_num_envs=1024, training_iterations=100,
            evaluation_num_envs=len(evaluation_scenarios())*8,
            ranking='All direction/turn/combined and uninterrupted transition scenarios must pass; preserve Benchmark 1.',
            omni={'architecture':'omni_history_direct_v1', 'history_frames':5, 'actor_frame_width':63,
                  'actor_fields':['angular_velocity_navigation_scaled', 'projected_gravity_navigation',
                                  'ramped_v_forward_v_left_yaw_command_scaled', 'joint_position_offset',
                                  'joint_velocity_scaled', 'previous_applied_action'],
                  'critic_extra_fields':['linear_velocity_navigation_truth_scaled'],
                  'command_frame':'forward=-body_y, left=+body_x, positive_yaw=+body_z',
                  'max_planar_speed_mps':.20, 'max_yaw_rad_s':.40,
                  'curriculum_version':2, 'combined_yaw_range_rad_s':[-.40,.40],
                  'navigation_contract':'Plan position and body heading separately. Local predictive controller supplies continuous body twist; constant twist yields straight motion, arcs or pure turns without changing policy.',
                  'resampling_seconds':[3,6], 'linear_acceleration_mps2':.25, 'yaw_acceleration_rad_s2':.8,
                  'evaluation_seeds':[7057,17057], 'scenarios':evaluation_scenarios(),
                  'transition_sequence':transition_sequence(),
                  'future_terrain_contract':'Keep body-twist commands and joint-name action mapping. Add terrain-relative clearance/support observations, age/validity and recurrent terrain encoder after bounded terrain baseline. Actor never receives simulator terrain/state truth implicitly.',
                  'scope':'Flat ground only. No terrain or camera/LiDAR training is launched in this step.'})
(out/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'out':str(out), 'scenarios':len(evaluation_scenarios()), 'evaluation_envs':plan['evaluation_num_envs']}))
