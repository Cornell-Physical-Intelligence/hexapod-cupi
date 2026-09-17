"""Direct flat-ground locomotion environment configuration."""

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from .asset_cfg import HEXAPOD_CFG
from .assets.articulation import HEXAPOD_MKII_V1_CFG
from .assets.spec import MKII_V1_ASSET


# Phase-0 mock link names. ``HexapodFlatEnvCfg`` and every task derived from
# it keep these; a new robot model supplies its own through the asset spec.
LEG_LINK_NAMES = (
    ("coxa", "femur", "tibia"),
    ("coxa_1", "femur_1", "tibia_1"),
    ("coxa_2", "femur_2", "tibia_2"),
    ("coxa_3", "femur_3", "tibia_3"),
    ("coxa_4", "femur_4", "tibia_4"),
    ("coxa_5", "femur_5", "tibia_5"),
)

GROUND_PLANE_COLLISION_PATH = "/World/ground/terrain/GroundPlane/CollisionPlane"


def _contact_sensor(
    path: str,
    *,
    track_air_time: bool = False,
    track_contact_points: bool = False,
    track_friction_forces: bool = False,
) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=path,
        history_length=3,
        update_period=0.005,
        track_air_time=track_air_time,
        track_pose=track_contact_points,
        track_contact_points=track_contact_points,
        track_friction_forces=track_friction_forces,
        max_contact_data_count_per_prim=8 if track_contact_points else None,
        filter_prim_paths_expr=[GROUND_PLANE_COLLISION_PATH] if track_contact_points else [],
    )


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="root"),
            "mass_distribution_params": (-0.20, 0.40),
            "operation": "add",
        },
    )


@configclass
class HexapodFlatEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.30
    # Multiplier applied to normalized policy actions only while every command
    # axis is inactive.  One preserves the historical action path exactly;
    # zero requests the robot's nominal/default joint targets while standing.
    stand_action_scale = 1.0
    action_space = 18
    observation_space = 66
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=2.0,
        replicate_physics=True,
    )
    events: EventCfg = EventCfg()
    robot = HEXAPOD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # PhysX contact views require bodies selected by one sensor to share a
    # parent. The URDF preserves a nested root/coxa/femur/tibia hierarchy, so
    # distal links use one exact-path sensor per leg.
    base_contact_sensor: ContactSensorCfg = _contact_sensor(
        "/World/envs/env_.*/Robot/Geometry/root"
    )
    coxa_contact_sensor: ContactSensorCfg = _contact_sensor(
        "/World/envs/env_.*/Robot/Geometry/root/coxa.*"
    )
    feet_contact_sensors: tuple[ContactSensorCfg, ...] = tuple(
        _contact_sensor(
            f"/World/envs/env_.*/Robot/Geometry/root/{coxa}/{femur}/{tibia}",
            track_air_time=True,
            track_contact_points=True,
            track_friction_forces=True,
        )
        for coxa, femur, tibia in LEG_LINK_NAMES
    )
    femur_contact_sensors: tuple[ContactSensorCfg, ...] = tuple(
        _contact_sensor(f"/World/envs/env_.*/Robot/Geometry/root/{coxa}/{femur}")
        for coxa, femur, _ in LEG_LINK_NAMES
    )

    rated_torque_nm = 1.6
    # Moving/legacy base-height target. Tasks with stand commands can opt into
    # a distinct target below; ``None`` preserves the historical single target
    # for every command.
    nominal_height_m = 0.205
    stand_nominal_height_m: float | None = None
    # Tibia link-frame +Y runs from the knee toward the terminal pad. The
    # imported mesh reaches y=0.210 m, leaving 30 mm of numerical margin here.
    distal_foot_min_y_m = 0.18
    # Per-asset link names the environment resolves foot bodies from. The
    # default is the mock's table above; asset configs point this at their
    # spec so ``HexapodEnv`` carries no asset literals of its own.
    leg_link_names: tuple[tuple[str, str, str], ...] = LEG_LINK_NAMES
    # When set, ``HexapodEnv`` compares the imported articulation's joint order
    # against this tuple at construction and refuses to run on a mismatch, so
    # a new asset cannot train against an unconfirmed action order. ``None``
    # preserves the historical behaviour for every existing task.
    expected_runtime_joint_names: tuple[str, ...] | None = None
    # Phase 1 trains forward motion only. Keep these ranges explicit so later
    # phases can introduce standing, lateral motion, and turning without
    # changing environment code.
    # Body is the legacy convention. Corrected tasks opt into anatomical
    # navigation axes, where forward=-body-Y and lateral=body+X.
    command_frame = "body"
    command_lin_vel_x_range_mps = (0.15, 0.35)
    command_lin_vel_y_range_mps = (0.0, 0.0)
    command_yaw_rate_range_rad_s = (0.0, 0.0)
    stand_command_fraction = 0.0
    moving_command_threshold_mps = 0.05
    support_contact_target = 3.0
    # Optional speed-conditioned support schedule.  With the switch disabled,
    # ``support_contact_target`` remains the sole target and preserves legacy
    # reward behavior exactly.  The opt-in defaults describe a conservative
    # wave/ripple/tripod progression as commanded planar speed increases.
    speed_conditioned_support_targets = False
    support_contact_low_speed_threshold_mps = 0.10
    support_contact_high_speed_threshold_mps = 0.25
    support_contact_target_low_speed = 5.0
    support_contact_target_medium_speed = 4.0
    support_contact_target_high_speed = 3.0
    forward_tracking_tolerance_mps = 0.10
    lin_vel_tracking_std_mps = 0.20
    yaw_rate_tracking_std_rad_s = 0.50

    # Legacy tasks use one coupled planar Gaussian. Recovery curricula can opt
    # into independent axes so success on an already-learned direction cannot
    # hide a collapsed lateral or yaw response.
    axiswise_velocity_rewards = False
    lin_vel_x_tracking_std_mps = 0.20
    lin_vel_y_tracking_std_mps = 0.20
    lin_vel_x_reward_scale = 3.0
    lin_vel_y_reward_scale = 0.0
    # A task may independently suppress the longitudinal Gaussian when the
    # x command is inactive.  Keep this separate from the later-axis y/yaw
    # gate so every existing reward path remains unchanged unless it opts in.
    gate_longitudinal_reward_by_command = False
    # Later recovery stages can restrict new-axis tracking rewards to episodes
    # where that axis is actually commanded.  Defaults preserve every legacy
    # reward path exactly.
    gate_axis_rewards_by_command = False
    axis_command_active_threshold = 0.01
    # Optional signed/capped longitudinal acquisition terms.  These remain
    # dormant for all existing tasks; Stage2E enables them so small reverse-x
    # commands cannot earn the broad x Gaussian by simply standing still.
    longitudinal_signed_progress_reward_scale = 0.0
    longitudinal_normalized_error_penalty_scale = 0.0
    longitudinal_normalized_error_cap = 2.0
    lateral_signed_progress_reward_scale = 0.0
    yaw_signed_progress_reward_scale = 0.0
    # Optional low-pass shaping for navigation-frame lateral acquisition.
    # ``None`` preserves the instantaneous signed-progress path exactly.  When
    # enabled, the reset-safe EMA feeds both lateral signed progress and the
    # capped normalized-error penalty below.  The penalty scale is a positive
    # cost magnitude and remains fully dormant at zero.
    navigation_lateral_velocity_ema_tau_s: float | None = None
    lateral_normalized_error_penalty_scale = 0.0
    lateral_normalized_error_cap = 2.0
    inactive_lateral_velocity_reward_scale = 0.0
    inactive_yaw_rate_reward_scale = 0.0
    # Bounded penalty on command-frame yaw-rate change between policy steps.
    # It is reset-safe and applies only while yaw is uncommanded.  The scale
    # stays dormant by default so existing reward tensors are unchanged.
    inactive_yaw_rate_slew_reward_scale = 0.0
    inactive_yaw_rate_slew_reference_rad_s_per_step = 0.04
    # Bounded instantaneous penalty on the net ground-reaction yaw moment
    # about the robot root while yaw is uncommanded.  This targets the wrench
    # that causes straight-gait yaw oscillation, including asymmetric
    # left/right tangential loading, without carrying state across resets.
    inactive_ground_contact_yaw_moment_reward_scale = 0.0
    inactive_ground_contact_yaw_moment_reference_nm = 0.50
    # Longitudinal-only component of the same contact wrench. This leaves
    # lateral corrective forces and alternating tripod support untouched while
    # penalizing differential left/right forward traction. Opt-in only.
    inactive_bilateral_longitudinal_contact_moment_reward_scale = 0.0
    inactive_bilateral_longitudinal_contact_moment_reference_nm = 0.50

    # --- Biomechanical insect-gait shaping (Stage2G). All terms are dormant
    # at zero scale and preserve every legacy reward tensor exactly. ---
    # Alternating-tripod phase reward: a per-environment gait clock advances at
    # a commanded-speed-proportional stride frequency; tripod A (front-left,
    # mid-right, hind-left, resolved geometrically at runtime) is expected in
    # stance during the first half-cycle and tripod B during the second half.
    # Feet matching their expected stance/swing state earn the reward.
    gait_phase_contact_reward_scale = 0.0
    # Duty factor: stance fraction of the cycle per leg. Insects run tripod at
    # ~0.5; values above 0.5 overlap the tripods (tetrapod-like) for slower,
    # more conservative stepping.
    gait_duty_factor = 0.5
    # Stride frequency model: f = clamp(cycles_per_meter * |v_cmd|, min, max).
    # 8 cycles/m is a 0.125 m stride: 1.28 Hz at 0.16 m/s, 2.4 Hz at 0.30 m/s.
    gait_cycles_per_meter = 8.0
    gait_min_frequency_hz = 1.2
    gait_max_frequency_hz = 3.0
    # Sigmoid steepness of the smooth stance-window indicator in phase units.
    gait_phase_transition_sharpness = 30.0
    # Swing-clearance reward: feet in their expected swing window earn a
    # Gaussian bump for lifting the distal pad toward the target apex height,
    # which suppresses dragging, shuffling swings.
    swing_clearance_reward_scale = 0.0
    swing_clearance_target_m = 0.030
    swing_clearance_tolerance_m = 0.015
    # Distal pad offset along tibia-link +Y used to estimate pad height.
    swing_clearance_pad_offset_y_m = 0.21
    # Append [sin(2*pi*phase), cos(2*pi*phase)] to the policy observation.
    # Tasks enabling this must also widen ``observation_space`` by two.
    include_gait_phase_observation = False

    lin_vel_reward_scale = 3.0
    yaw_rate_reward_scale = 0.5
    alive_reward_scale = 0.10
    z_vel_reward_scale = -1.5
    ang_vel_reward_scale = -0.10
    joint_torque_reward_scale = -1.0e-4
    rated_torque_excess_reward_scale = -0.03
    # Squared excess of the single worst raw-PD-demand joint above the RS05
    # continuous rating. This complements the aggregate excess cost without
    # letting one overloaded joint be diluted by the other 17. Opt-in only.
    max_joint_rated_torque_excess_reward_scale = 0.0
    # Linear worst-joint hinge at the continuous rating. Unlike the squared
    # term, this retains a useful gradient for sustained, small over-rating
    # excursions, matching the admission gate's duty-cycle sensitivity.
    max_joint_rated_torque_excess_l1_reward_scale = 0.0
    # Optional gate-aligned cost for how often raw PD demand exceeds the
    # RS05 continuous rating. Kept disabled for existing task variants.
    torque_saturation_reward_scale = 0.0
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.02
    # Squared change in applied joint torque between policy steps.  Disabled
    # by default; reset transitions never contribute to this term.
    joint_torque_slew_reward_scale = 0.0
    # Optional limiter on the final, soft-limit-clamped joint target.  Units
    # are radians per 20 ms policy step and are scaled by the actual step_dt;
    # ``None`` leaves the historical processed-action semantics unchanged.
    processed_joint_target_slew_limit_rad_per_20ms: float | None = None
    feet_air_time_reward_scale = 0.0
    foot_slip_reward_scale = -0.10
    support_shortfall_reward_scale = -0.15
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -2.0
    base_height_reward_scale = -10.0
    joint_limit_reward_scale = -0.25
    # Bounded [0, 1] deck-stability score, multiplied by this reward scale.
    # Each positive finite normalization scale is the error at which its
    # equal-weight component score falls to 0.5.  Disabled by default.
    deck_stability_reward_scale = 0.0
    deck_stability_vertical_velocity_scale_mps = 0.10
    deck_stability_roll_pitch_rate_scale_rad_s = 0.50
    deck_stability_projected_gravity_xy_scale = 0.15
    deck_stability_height_error_scale_m = 0.025
    # Applied once on a terminal fall, outside policy-step time scaling.
    fall_penalty = 0.0


# ---------------------------------------------------------------------------
# Asset v1: the CAD assembly (ADR-0001). Same task, different robot. Every
# asset-specific value comes from ``MKII_V1_ASSET``; reward scales, command
# ranges and timing are inherited unchanged from ``HexapodFlatEnvCfg`` and are
# expected to be re-derived for the 8.26 kg mass distribution (stance sweep).
# ---------------------------------------------------------------------------

# Isaac Sim's URDF importer nests every link under its parent inside the
# robot's Geometry scope: Robot/Geometry/body/lf_coxa/lf_femur/lf_tibia.
MKII_V1_GEOMETRY_ROOT = MKII_V1_ASSET.geometry_root_prim("/World/envs/env_.*/Robot")


@configclass
class MkiiV1EventCfg(EventCfg):
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MKII_V1_ASSET.root_link),
            "mass_distribution_params": (-0.20, 0.40),
            "operation": "add",
        },
    )


@configclass
class HexapodMkiiV1FlatEnvCfg(HexapodFlatEnvCfg):
    """Flat-ground forward-walking task on asset v1 (``hexapod_mkii_serial``)."""

    events: MkiiV1EventCfg = MkiiV1EventCfg()
    robot = HEXAPOD_MKII_V1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    base_contact_sensor: ContactSensorCfg = _contact_sensor(MKII_V1_GEOMETRY_ROOT)
    coxa_contact_sensor: ContactSensorCfg = _contact_sensor(
        f"{MKII_V1_GEOMETRY_ROOT}/{MKII_V1_ASSET.coxa_link_regex}"
    )
    feet_contact_sensors: tuple[ContactSensorCfg, ...] = tuple(
        _contact_sensor(
            f"{MKII_V1_GEOMETRY_ROOT}/{coxa}/{femur}/{tibia}",
            track_air_time=True,
            track_contact_points=True,
            track_friction_forces=True,
        )
        for coxa, femur, tibia in MKII_V1_ASSET.leg_link_names
    )
    femur_contact_sensors: tuple[ContactSensorCfg, ...] = tuple(
        _contact_sensor(f"{MKII_V1_GEOMETRY_ROOT}/{coxa}/{femur}")
        for coxa, femur, _ in MKII_V1_ASSET.leg_link_names
    )
    leg_link_names: tuple[tuple[str, str, str], ...] = MKII_V1_ASSET.leg_link_names
    expected_runtime_joint_names: tuple[str, ...] | None = MKII_V1_ASSET.runtime_joint_names
    # The root link is the bottom plate of the chassis.
    nominal_height_m = MKII_V1_ASSET.nominal_height_m
    distal_foot_min_y_m = MKII_V1_ASSET.distal_foot_min_y_m
    swing_clearance_pad_offset_y_m = MKII_V1_ASSET.swing_clearance_pad_offset_y_m
