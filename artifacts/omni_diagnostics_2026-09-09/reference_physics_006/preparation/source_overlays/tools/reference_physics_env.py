"""Exact full-C physical configuration for a physics-only reference screen.

Import/build only after AppLauncher. No actor, optimizer, task registration,
asset mutation, root-pose control, or old checkpoint transfer is provided.
"""
import json
from pathlib import Path


def build_reference_environment(args, residual_options):
    from isaaclab.actuators import DCMotorCfg
    from isaaclab.utils.io import dump_yaml
    from hexapod_rl import env as env_module
    from hexapod_rl.env_cfg import _contact_sensor
    from hexapod_rl.phase1_v5_cfg import HexapodPhase1V5EnvCfg
    from candidate_asset_audit import audit_candidate_usd
    from reference_residual_env import configure_reference_residual_physics, ReferenceResidualPhysicsEnv
    from physics_telemetry import named_layout

    package = Path(args.package)
    manifest = json.loads((package / 'manifest.json').read_text())
    plan = json.loads((package / 'training_plan.json').read_text())
    record = next(row for row in manifest['variants'] if row['variant'] == args.variant)
    stance = plan['variants'][args.variant]['stances'][args.stance_index]
    usd = package / 'training_usd' / args.variant / (args.variant + '.usda')
    asset_audit = audit_candidate_usd(package / record['urdf'], usd)
    from solver_comparison import configure_comparison, readback_comparison, configure_iteration_comparison
    cfg = HexapodPhase1V5EnvCfg()
    solver_comparison = configure_comparison(cfg)
    cfg.seed = plan['training_seed']
    cfg.events = None
    cfg.sim.dt = plan['physics_dt_s']
    cfg.decimation = plan['decimation']
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 90.
    cfg.robot.spawn.usd_path = str(usd)
    cfg.robot.spawn.articulation_props.solver_position_iteration_count = 16
    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    solver_comparison['iteration_comparison'] = configure_iteration_comparison(cfg)
    cfg.robot.init_state.pos = (0., 0., stance['suggested_reset_root_height_m'])
    cfg.robot.init_state.joint_pos = stance['joint_positions_rad']
    cfg.robot.actuators = {'legs': DCMotorCfg(**manifest['actuator_config_snapshot'])}
    cfg.nominal_height_m = stance['root_height_at_contact_m']
    cfg.distal_foot_min_y_m = record['tibia_length_m'] * manifest['study']['distal_foot_fraction']
    cfg.swing_clearance_pad_offset_y_m = record['tibia_length_m']
    cfg.command_lin_vel_x_range_mps = (.1, .3)
    cfg.command_frame = 'navigation'
    cfg.action_scale = .20
    cfg.rated_torque_excess_reward_scale = -.12
    cfg.torque_saturation_reward_scale = -.5
    cfg.max_joint_rated_torque_excess_reward_scale = -.05
    cfg.fall_penalty = -2.
    names = tuple(tuple(manifest['link_joint_mapping'][leg]['links'][kind]
                        for kind in ('coxa', 'femur', 'tibia'))
                  for leg in ('lf', 'lm', 'lr', 'rf', 'rm', 'rr'))
    env_module.LEG_LINK_NAMES = names
    root = '/World/envs/env_.*/Robot/Geometry/body_mock'
    cfg.base_contact_sensor = _contact_sensor(root)
    cfg.coxa_contact_sensor = _contact_sensor(root + '/coxa.*')
    cfg.feet_contact_sensors = tuple(_contact_sensor(f'{root}/{coxa}/{femur}/{tibia}',
        track_air_time=True, track_contact_points=True, track_friction_forces=True)
        for coxa, femur, tibia in names)
    cfg.femur_contact_sensors = tuple(_contact_sensor(f'{root}/{coxa}/{femur}') for coxa, femur, _ in names)
    for sensor in (cfg.base_contact_sensor, cfg.coxa_contact_sensor, *cfg.feet_contact_sensors, *cfg.femur_contact_sensors):
        sensor.update_period = cfg.sim.dt
        if sensor.track_contact_points or sensor.track_friction_forces:
            sensor.max_contact_data_count_per_prim = max(128, sensor.max_contact_data_count_per_prim)
    contract = configure_reference_residual_physics(cfg, residual_options, plan['omni'].get('overrides'))
    dump_yaml(str(args.output / 'environment.yaml'), cfg)
    env = ReferenceResidualPhysicsEnv(cfg=cfg, evaluation=True)
    try:
        solver_comparison = readback_comparison(env, solver_comparison)
    except Exception as error:
        solver_comparison['readback_error'] = repr(error)
        solver_comparison['actual_iteration_settings'] = getattr(error, 'iteration_readback', None)
        (args.output / 'solver_comparison_failed_readback.json').write_text(json.dumps(solver_comparison,indent=2,sort_keys=True)+'\n')
        raise
    (args.output / 'solver_comparison.json').write_text(json.dumps(solver_comparison,indent=2,sort_keys=True)+'\n')
    if abs(env.step_dt - .02) > 1e-9 or env.num_envs != args.num_envs:
        raise RuntimeError('Reference screen requires declared environment count at50Hz')
    layout = named_layout(manifest, env._robot.joint_names, env._robot.body_names,
                          [sensor.body_names for sensor in env._feet_contact_sensors])
    return env, manifest, plan, record, stance, layout, asset_audit, contract
