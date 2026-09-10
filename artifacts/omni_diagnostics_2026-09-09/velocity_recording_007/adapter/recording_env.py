"""Rendering-only construction of the immutable 495/498 candidate environment.

The physical configuration and actor construction mirror the frozen training
entrypoint. Explicit recording overrides are one environment and RGB output.
Imports requiring Isaac are delayed until after AppLauncher.
"""
import json
from importlib import metadata
from recording_contract import digest, IDENTITY, save_json


def build_env_and_runner(args, candidate_config):
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    from isaaclab.utils.io import dump_yaml
    from hexapod_rl import env as env_module
    from hexapod_rl.env_cfg import _contact_sensor
    from hexapod_rl.phase1_v5_cfg import HexapodPhase1V5EnvCfg, HexapodPhase1V5PPORunnerCfg
    from isaaclab.actuators import DCMotorCfg
    from candidate_asset_audit import audit_candidate_usd
    from candidate_env import configure_velocity_candidate, VelocityCandidateEnv
    from candidate_runner import INITIAL_STD, LEARNING_RATE

    manifest = json.loads((args.package / 'manifest.json').read_text())
    plan = json.loads((args.package / 'training_plan.json').read_text())
    record = next(r for r in manifest['variants'] if r['variant'] == args.variant)
    urdf = args.package / record['urdf']
    if digest(urdf) != record['sha256']:
        raise RuntimeError('URDF hash mismatch')
    stance = plan['variants'][args.variant]['stances'][args.stance_index]
    usd = args.package / 'training_usd' / args.variant / (args.variant + '.usda')
    tensor_audit = audit_candidate_usd(urdf, usd)
    cfg = HexapodPhase1V5EnvCfg()
    cfg.seed = plan['training_seed']
    cfg.events = None
    cfg.sim.dt = plan['physics_dt_s']
    cfg.decimation = plan['decimation']
    cfg.sim.render_interval = cfg.decimation
    # Kept as the exact evaluation values until explicit recording overrides below.
    cfg.scene.num_envs = plan['evaluation_num_envs']
    cfg.sim.device = args.device
    omni = plan.get('omni')
    cfg.episode_length_s = 90.0 if omni else 25.0
    cfg.robot.spawn.usd_path = str(usd)
    cfg.robot.spawn.articulation_props.solver_position_iteration_count = 16
    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    cfg.robot.init_state.pos = (0, 0, stance['suggested_reset_root_height_m'])
    cfg.robot.init_state.joint_pos = stance['joint_positions_rad']
    cfg.robot.actuators = {'legs': DCMotorCfg(**manifest['actuator_config_snapshot'])}
    cfg.nominal_height_m = stance['root_height_at_contact_m']
    cfg.distal_foot_min_y_m = record['tibia_length_m'] * manifest['study']['distal_foot_fraction']
    cfg.swing_clearance_pad_offset_y_m = record['tibia_length_m']
    cfg.command_lin_vel_x_range_mps = (0.1, 0.3)
    cfg.command_frame = 'navigation'
    cfg.action_scale = 0.20
    cfg.rated_torque_excess_reward_scale = -0.12
    cfg.torque_saturation_reward_scale = -0.5
    cfg.max_joint_rated_torque_excess_reward_scale = -0.05
    cfg.fall_penalty = -2.0
    names = tuple(tuple(manifest['link_joint_mapping'][leg]['links'][k] for k in ('coxa', 'femur', 'tibia'))
                  for leg in ('lf', 'lm', 'lr', 'rf', 'rm', 'rr'))
    env_module.LEG_LINK_NAMES = names
    root = '/World/envs/env_.*/Robot/Geometry/body_mock'
    cfg.base_contact_sensor = _contact_sensor(root)
    cfg.coxa_contact_sensor = _contact_sensor(root + '/coxa.*')
    cfg.feet_contact_sensors = tuple(_contact_sensor(f'{root}/{c}/{f}/{t}', track_air_time=True,
                                                   track_contact_points=True, track_friction_forces=True) for c, f, t in names)
    cfg.femur_contact_sensors = tuple(_contact_sensor(f'{root}/{c}/{f}') for c, f, _ in names)
    for sensor in (cfg.base_contact_sensor, cfg.coxa_contact_sensor, *cfg.feet_contact_sensors, *cfg.femur_contact_sensors):
        sensor.update_period = cfg.sim.dt
    configure_velocity_candidate(cfg, omni['velocity_candidate'], omni.get('overrides'))
    dump_yaml(str(args.output / 'matched_evaluation_environment.yaml'), cfg)
    recording_overrides = {
        'scene.num_envs': {'from': cfg.scene.num_envs, 'to': 1},
        'video_recorder.window_width': {'from': cfg.video_recorder.window_width, 'to': 1280},
        'video_recorder.window_height': {'from': cfg.video_recorder.window_height, 'to': 720},
        'render_mode': 'rgb_array', 'enable_cameras': True,
        'seed': args.seed, 'omni_evaluation': True,
        'commands': 'fixed external recording schedule; no pose feedback',
        'episode_length_s': cfg.episode_length_s,
        'root_quaternion_convention': 'installed Isaac Lab XYZW; inherited reset orientation unchanged',
    }
    cfg.scene.num_envs = 1
    cfg.video_recorder.window_width = 1280
    cfg.video_recorder.window_height = 720
    dump_yaml(str(args.output / 'recording_environment.yaml'), cfg)
    save_json(args.output / 'recording_overrides.json', recording_overrides)
    torch.manual_seed(args.seed)
    env = VelocityCandidateEnv(cfg=cfg, render_mode='rgb_array', evaluation=True)
    if (env._robot.num_joints != 18 or env._robot.num_bodies != 19
        or set(env._robot.joint_names) != set(stance['joint_positions_rad'])
        or sum(len(sensor.body_names) for sensor in env._feet_contact_sensors) != 6):
        raise RuntimeError('Expected the complete C robot: 19 bodies, 18 named joints, six feet')
    if env.num_envs != 1 or abs(env.step_dt - .02) > 1e-8:
        raise RuntimeError('One 50 Hz full-robot recording environment required')
    runner_cfg = HexapodPhase1V5PPORunnerCfg()
    runner_cfg.seed = plan['training_seed']
    runner_cfg.device = env.device
    runner_cfg.max_iterations = plan['training_iterations']
    runner_cfg.save_interval = 25
    runner_cfg.logger = 'tensorboard'
    runner_cfg.obs_groups = {'actor': ['policy'], 'critic': ['critic']}
    runner_cfg.experiment_name = 'hexapod_c_target_velocity_v1_' + candidate_config.profile
    runner_cfg.actor.distribution_cfg.init_std = INITIAL_STD
    runner_cfg.algorithm.learning_rate = LEARNING_RATE
    runner_cfg.algorithm.entropy_coef = 0.
    if runner_cfg.clip_actions is not None:
        raise RuntimeError('Raw candidate actions require unclipped wrapper')
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version('rsl-rl-lib'))
    dump_yaml(str(args.output / 'agent.yaml'), runner_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=str(args.output / 'runner'), device=env.device)
    return env, runner, plan, tensor_audit
