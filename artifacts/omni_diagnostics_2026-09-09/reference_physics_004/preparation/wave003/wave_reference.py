"""Single-replica, measured-contact wave reference. CPU only; no physics writes.

Reference anchors and measured contact anchors are intentionally distinct: the
standing PD preload is preserved at reset and measured again at touchdown.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from serial_geometry import SerialGeometry, tensor
from wave_math import Swing, BodyState, MassGeometry, rotation

LEGS = ('lf', 'lm', 'lr', 'rf', 'rm', 'rr')


@dataclass(frozen=True)
class WaveConfig:
    gait_order: tuple = ('lf', 'rr', 'lm', 'rf', 'lr', 'rm')
    swing_s: float = 2.0
    contact_hold_s: float = .30
    lift_m: float = .007
    max_translation_mps: float = .005
    max_yaw_rad_s: float = .015
    command_filter_omega: float = 2.
    reference_velocity_rad_s: float = 1.75
    reference_acceleration_rad_s2: float = 6.
    joint_margin_rad: float = .02
    minimum_support_margin_m: float = .025
    maximum_body_tracking_error_m: float = .035
    maximum_stance_drift_m: float = .02
    maximum_touchdown_error_m: float = .012
    maximum_initial_preload_point_offset_m: float = .025
    maximum_initial_joint_preload_rad: float = .15
    max_touchdown_delay_s: float = .60
    contact_confirm_steps: int = 3
    minimum_measured_lift_m: float = .002
    landing_blend_s: float = .50
    minimum_landing_blend_s: float = .10
    descent_drop_from_peak_m: float = .00025
    maximum_landing_contact_gap_s: float = .10
    stop_command_tolerance: float = 1e-6
    stop_command_rate_tolerance: float = 2e-5


class WaveContactReference:
    """reset(snapshot), step(snapshot, requested_forward_left_yaw, dt=.02).

    Output q_ref/v_ref/a_ref have shape [1,18] in declared articulation order;
    valid is bool[1]. Invalid output has None targets and latches failure.
    The generator never sets robot pose or replaces a physics/contact result.
    """
    def __init__(self, joint_names_runtime, config=WaveConfig(), geometry=None):
        self.g = geometry or SerialGeometry()
        self.names = tuple(joint_names_runtime)
        self.leg_names = tuple(n for names in self.g.names for n in names)
        if len(self.names) != 18 or len(set(self.names)) != 18 or set(self.names) != set(self.leg_names):
            raise ValueError('Exact18 named C runtime joints required')
        if set(config.gait_order) != set(LEGS) or len(config.gait_order) != 6:
            raise ValueError('Explicit six-leg wave order required')
        if (config.swing_s <= 0 or config.contact_hold_s < .02 or config.contact_confirm_steps < 1
                or not 0 < config.minimum_landing_blend_s <= config.landing_blend_s or config.minimum_measured_lift_m < .002
                or config.descent_drop_from_peak_m <= 0 or not .02 <= config.maximum_landing_contact_gap_s <= .10):
            raise ValueError('Positive swing/confirmed contact dwell required')
        self.to_leg = np.array([self.names.index(name) for name in self.leg_names])
        self.to_runtime = np.argsort(self.to_leg)
        self.cfg = config
        self.ready = False
        self.failure = None

    def _one(self, snapshot, key, shape):
        value = np.asarray(snapshot[key])
        if value.shape != (1, *shape) or (key != 'contact_point_world_m' and not np.isfinite(value).all()):
            raise ValueError('One finite replica required for ' + key)
        return value[0].copy()

    def _read(self, snapshot):
        result = {key: self._one(snapshot, key, shape) for key, shape in (
            ('time_s', ()), ('position_world_m', (3,)), ('quaternion_world_xyzw', (4,)),
            ('rotation_world_from_body', (3, 3)), ('velocity_body_mps', (3,)), ('gyro_body_rad_s', (3,)),
            ('joint_position_rad', (18,)), ('joint_target_rad', (18,)),
            ('reference_point_world_m', (6, 3)), ('reference_point_velocity_world_mps', (6, 3)),
            ('contact_point_world_m', (6, 3)), ('contact_point_valid', (6,)),
            ('distal_contact', (6,)), ('shaft_contact', (6,)), ('coxa_contact', (6,)),
            ('femur_contact', (6,)), ('base_contact', ()))}
        q = result['quaternion_world_xyzw']; x, y, z, w = q
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        if abs(np.linalg.norm(q)-1) > 1e-4 or not np.allclose(R, result['rotation_world_from_body'], atol=2e-4):
            raise ValueError('Explicit measured XYZ W quaternion and body rotation disagree')
        for key in ('contact_point_valid', 'distal_contact', 'shaft_contact', 'coxa_contact', 'femur_contact', 'base_contact'):
            if result[key].dtype != bool:
                raise ValueError('Boolean contact classification required: ' + key)
        result['contact'] = result['distal_contact'] & result['contact_point_valid']
        if not np.isfinite(result['contact_point_world_m'][result['contact']]).all():
            raise ValueError('Confirmed contact has no finite measured point')
        result['terminal'] = any(bool(np.asarray(snapshot.get(key, [False])).any()) for key in ('terminated', 'truncated'))
        return result

    def _leg(self, runtime):
        return np.asarray(runtime)[self.to_leg].reshape(6, 3)

    def _runtime(self, leg):
        return np.asarray(leg).reshape(18)[self.to_runtime]

    def _fk(self, q):
        return self.g.fk(tensor(q))

    def _body(self):
        R = rotation(self.yaw) @ self.R0
        omega = np.array([0., 0., self.command[2]])
        body_v = np.array([self.command[1], -self.command[0], 0.])
        return BodyState(self.time, self.position.copy(), R, R @ body_v, np.zeros(3), omega, np.zeros(3),
                         provenance='virtual_desired_motion_not_measured_or_prescribed_physics_pose')

    def reset(self, snapshot):
        measured = self._read(snapshot)
        if measured['terminal'] or measured['base_contact'] or any(measured[k].any() for k in ('shaft_contact', 'coxa_contact', 'femur_contact')):
            raise ValueError('Reset needs nonterminal foot-only settled support')
        if measured['contact'].sum() < 5:
            raise ValueError('At least five actual distal contacts required at reset')
        target_v = self._one(snapshot, 'executable_target_velocity_rad_s', (18,))
        if np.max(np.abs(target_v)) > 1e-6:
            raise ValueError('Start from stationary executed target; no velocity reset jump')
        self.time = float(measured['time_s']); self.position = measured['position_world_m'].copy()
        actual_limits = self._one(snapshot, 'soft_joint_pos_limits_rad', (18, 2))
        self.lower = np.maximum(self.g.lower.numpy(), self._leg(actual_limits[:, 0]))
        self.upper = np.minimum(self.g.upper.numpy(), self._leg(actual_limits[:, 1]))
        self.R0 = measured['rotation_world_from_body'].copy(); self.yaw = 0.
        self.command = np.zeros(3); self.command_rate = np.zeros(3)
        self.q = self._leg(measured['joint_target_rad']); self.v = np.zeros((6, 3))
        self.preload_q = self.q - self._leg(measured['joint_position_rad'])
        if np.max(np.abs(self.preload_q)) > self.cfg.maximum_initial_joint_preload_rad:
            raise ValueError('Initial measured PD preload exceeds declared screening bound')
        self.neutral, _, _ = self._fk(self.q); self.neutral = self.neutral.numpy()
        self.anchors = (self.R0 @ self.neutral.T).T + self.position
        self.measured_anchors = measured['reference_point_world_m'].copy()
        self.preload_world = self.anchors - self.measured_anchors
        if np.linalg.norm(self.preload_world, axis=-1).max() > self.cfg.maximum_initial_preload_point_offset_m:
            raise ValueError('Initial virtual/measured foot offset exceeds declared bound')
        fk_measured, _, _ = self._fk(self._leg(measured['joint_position_rad']))
        fk_world = (self.R0 @ fk_measured.numpy().T).T + self.position
        if np.linalg.norm(fk_world - measured['reference_point_world_m'], axis=-1).max() > .001:
            raise ValueError('Measured named FK/toe points disagree with frozen C geometry by over1mm')
        self.helper = MassGeometry(self.g)
        self.current_leg = None; self.swing = None; self.mode = 'hold'; self.order_index = 0
        self.flight_seen = False; self.flight_count = 0; self.contact_count = 0; self.hold_until = self.time
        self.liftoffs = self.touchdowns = 0; self.failure = None; self.ready = True
        self.requested = np.zeros(3); self.command_target = np.zeros(3); self.factor = 1.
        self.stop_requested_time = None; self.reference_quiet_time = None
        self.landing = None; self.landing_trigger_s = None; self.landing_contact_origin = None
        self.flight_baseline_z = None; self.flight_peak_z = None; self.descent_seen = False
        self.landing_endpoint_correction_m = None; self.landing_contact_gap_steps = 0
        self.landing_original_contact_error_m = None; self.landing_target_excursion_m = None
        self.landing_preload_world_m = None
        self.last_diagnostics = {}
        self._bounds(self.q, self.v, self.v)
        return self._output(self.q, self.v, self.v, measured)

    def _com(self, q, position, R):
        _, _, Ts = self._fk(q)
        com = self.helper.body_mass * self.helper.body_com.copy()
        for j, T in enumerate(Ts):
            T = T.numpy()
            points = (T[:, :3, :3] @ self.helper.link_com[:, j, :, None])[..., 0] + T[:, :3, 3]
            com += (points * self.helper.link_mass[:, j, None]).sum(0)
        return position + R @ (com / self.helper.mass)

    def _support(self, measured, exclude=None):
        ids = [i for i in range(6) if measured['contact'][i] and i != exclude]
        if len(ids) < 5:
            raise ValueError('Fewer than five measured support contacts')
        com = self._com(self._leg(measured['joint_position_rad']), measured['position_world_m'], measured['rotation_world_from_body'])
        margin = self.helper.support_margin(measured['contact_point_world_m'], ids, com)
        if margin < self.cfg.minimum_support_margin_m:
            raise ValueError('Measured five-foot projected COM support margin insufficient')
        return margin

    def _advance(self, target, dt, *, position=None, yaw=None, command=None, rate=None):
        position = self.position.copy() if position is None else position.copy()
        yaw = self.yaw if yaw is None else yaw
        command = self.command.copy() if command is None else command.copy()
        rate = self.command_rate.copy() if rate is None else rate.copy()
        omega = self.cfg.command_filter_omega
        error = command - target; a = rate + omega * error; decay = math.exp(-omega * dt)
        next_command = target + (error + a * dt) * decay
        next_rate = (rate - omega * a * dt) * decay
        mid = .5*(command + next_command); dyaw = mid[2]*dt
        R = rotation(yaw + .5*dyaw) @ self.R0
        # Keep body height fixed in the declared flat-world proof.
        velocity = R @ np.array([mid[1], -mid[0], 0.]); velocity[2] = 0.
        position += dt*velocity
        return position, yaw+dyaw, next_command, next_rate

    def _predict(self, target, horizon):
        p = self.position.copy(); y = self.yaw; c = self.command.copy(); r = self.command_rate.copy()
        n = max(1, math.ceil(horizon/.04))
        for _ in range(n):
            p, y, c, r = self._advance(target, horizon/n, position=p, yaw=y, command=c, rate=r)
        return p, rotation(y) @ self.R0

    def _bounds(self, q, v, a):
        if (q < self.lower+self.cfg.joint_margin_rad).any() or (q > self.upper-self.cfg.joint_margin_rad).any():
            raise ValueError('Reference lacks full0.02rad residual joint margin')
        if np.max(np.abs(v)) > self.cfg.reference_velocity_rad_s+1e-5 or np.max(np.abs(a)) > self.cfg.reference_acceleration_rad_s2+1e-5:
            raise ValueError('Reference discrete velocity/acceleration budget exceeded; no clipping')

    def _output(self, q, v, a, measured):
        body = self._body()
        actual_twist = [float(-measured['velocity_body_mps'][1]), float(measured['velocity_body_mps'][0]), float(measured['gyro_body_rad_s'][2])]
        state = dict(mode=self.mode, current_leg=None if self.current_leg is None else LEGS[self.current_leg],
            next_wave_order_index=self.order_index, flight_seen=self.flight_seen, flight_count=self.flight_count, contact_count=self.contact_count,
            desired_time_s=self.time, desired_position_world_m=self.position.copy(), desired_rotation_world_from_body=body.rotation_wb.copy(),
            desired_yaw_delta_rad=self.yaw, command_filter_velocity=self.command.copy(), command_filter_rate=self.command_rate.copy(),
            reference_anchors_world_m=self.anchors.copy(), measured_anchors_world_m=self.measured_anchors.copy(),
            reference_minus_measured_preload_world_m=self.preload_world.copy(), initial_joint_preload_leg_major_rad=self.preload_q.copy(),
            hold_until_s=self.hold_until, liftoffs=self.liftoffs, confirmed_touchdowns=self.touchdowns,
            gait_order=self.cfg.gait_order,
            flight_baseline_z_m=self.flight_baseline_z, flight_peak_z_m=self.flight_peak_z,
            measured_flight_lift_m=None if self.flight_baseline_z is None else self.flight_peak_z-self.flight_baseline_z,
            actual_descent_seen=self.descent_seen, landing_trigger_s=self.landing_trigger_s,
            landing_contact_origin_world_m=self.landing_contact_origin,
            landing_endpoint_correction_m=self.landing_endpoint_correction_m,
            landing_original_contact_error_m=self.landing_original_contact_error_m,
            landing_target_excursion_m=self.landing_target_excursion_m,
            landing_preload_world_m=self.landing_preload_world_m,
            landing_contact_gap_steps=self.landing_contact_gap_steps,
            landing=None if self.landing is None else dict(start_s=self.landing.t0,duration_s=self.landing.duration,
                endpoint_world_m=self.landing.end.copy(),coefficients=self.landing.coeff.copy(),lift_m=0.),
            stop_requested_time_s=self.stop_requested_time, reference_quiet_time_s=self.reference_quiet_time,
            initial_desired_rotation_world_from_body=self.R0.copy(), neutral_reference_toes_body_m=self.neutral.copy(),
            joint_lower_leg_major_rad=self.lower.copy(), joint_upper_leg_major_rad=self.upper.copy(),
            swing=None if self.swing is None else dict(start_s=self.swing.t0, duration_s=self.swing.duration,
                endpoint_world_m=self.swing.end.copy(), coefficients=self.swing.coeff.copy(), lift_m=self.swing.lift))
        return dict(valid=np.array([self.failure is None]), failure_reason=self.failure,
            q_ref=None if self.failure else self._runtime(q)[None],
            v_ref=None if self.failure else self._runtime(v)[None],
            a_ref=None if self.failure else self._runtime(a)[None],
            target_time_s=self.time, joint_names_runtime=self.names,
            requested_command=self.requested.copy(), admitted_target_command=self.command_target.copy(),
            admitted_command=self.command.copy(), command_derating_factor=self.factor,
            actual_measured_command=np.array(actual_twist), state=state,
            diagnostics={**self.last_diagnostics, 'configuration': asdict(self.cfg),
                         'physics_qualified': False, 'pose_written_to_robot': False,
                         'derivative_semantics': 'actual_discrete_reference_knots_zero_motor_velocity_feedforward'})

    def step(self, snapshot, requested_forward_left_yaw, dt=.02):
        if not self.ready:
            raise RuntimeError('Reset from settled measured state first')
        measured = self._read(snapshot)
        if self.failure:
            return self._output(self.q, self.v, np.zeros_like(self.q), measured)
        try:
            if not math.isfinite(dt) or abs(dt-.02) > 1e-9 or abs(float(measured['time_s'])-self.time) > 1e-6:
                raise ValueError('Time-aligned50Hz measured snapshot required')
            requested = np.asarray(requested_forward_left_yaw, dtype=float)
            if requested.shape != (3,) or not np.isfinite(requested).all():
                raise ValueError('Finite forward/left/yaw requested twist required')
            if np.linalg.norm(requested) == 0 and np.linalg.norm(self.requested) > 0:
                self.stop_requested_time = self.time; self.reference_quiet_time = None
            if np.linalg.norm(requested) > 0:
                self.stop_requested_time = None; self.reference_quiet_time = None
            self.requested = requested.copy()
            factor = min(1., self.cfg.max_translation_mps/max(np.linalg.norm(requested[:2]), 1e-20),
                         self.cfg.max_yaw_rad_s/max(abs(requested[2]), 1e-20))
            target = factor*requested
            if measured['terminal'] or measured['base_contact'] or any(measured[k].any() for k in ('shaft_contact', 'coxa_contact', 'femur_contact')):
                raise ValueError('Measured terminal or nonfoot contact')
            margin = self._support(measured, self.current_leg)
            body_error = np.linalg.norm(measured['position_world_m'][:2]-self.position[:2])
            if body_error > self.cfg.maximum_body_tracking_error_m:
                raise ValueError('Actual body does not follow desired motion within declared bound')
            stance = [i for i in range(6) if i != self.current_leg and measured['contact'][i]]
            drift = np.linalg.norm(measured['reference_point_world_m']-self.measured_anchors, axis=-1)
            if drift[stance].max(initial=0) > self.cfg.maximum_stance_drift_m:
                raise ValueError('Actual planted toe drift exceeds declared bound')
            if self.current_leg is not None:
                i = self.current_leg
                actual_z = float(measured['reference_point_world_m'][i, 2])
                actual_speed = np.linalg.norm(measured['reference_point_velocity_world_mps'][i])
                if not measured['contact'][i]:
                    self.flight_count += 1; self.flight_seen = self.flight_count >= 2; self.contact_count = 0
                    self.flight_peak_z = max(self.flight_peak_z, actual_z)
                elif not self.flight_seen:
                    self.flight_count = 0
                    self.flight_baseline_z = actual_z; self.flight_peak_z = actual_z
                if self.flight_seen:
                    self.flight_peak_z = max(self.flight_peak_z, actual_z)
                    if (self.time >= self.swing.t0+.5*self.swing.duration
                            and actual_z < self.flight_peak_z-self.cfg.descent_drop_from_peak_m
                            and measured['reference_point_velocity_world_mps'][i, 2] < -.0001):
                        self.descent_seen = True
                if measured['contact'][i] and self.flight_seen and self.landing is None:
                    lift = self.flight_peak_z-self.flight_baseline_z
                    if (self.time < self.swing.t0+.5*self.swing.duration or not self.descent_seen
                            or lift < self.cfg.minimum_measured_lift_m):
                        raise ValueError('Returned contact lacks measured2mm flight, passed apex and actual descent')
                    if actual_speed > .04:
                        raise ValueError('Returned contact speed exceeds bounded landing admission')
                    # Start at the EXISTING virtual foot P/V/A, never at measured
                    # position. Preserve virtual XY to avoid a lateral target
                    # snap or restoring obsolete horizontal standing preload.
                    p0, v0, a0 = self.swing.sample(self.time)
                    endpoint = p0.copy()
                    endpoint[2] = actual_z+self.preload_world[i, 2]
                    original_error = float(np.linalg.norm(self.swing.end-measured['reference_point_world_m'][i]-self.preload_world[i]))
                    correction = float(np.linalg.norm(endpoint-self.swing.end))
                    if max(correction,original_error) > self.cfg.maximum_touchdown_error_m:
                        raise ValueError('Measured landing endpoint exceeds12mm correction bound')
                    duration = min(self.cfg.landing_blend_s,max(self.cfg.minimum_landing_blend_s,self.swing.end_time-self.time))
                    self.landing = Swing(self.time,duration,p0,endpoint,0.,v0,a0)
                    excursion = max(float(np.linalg.norm(self.landing.sample(t)[0]-p0))
                        for t in np.linspace(self.time,self.landing.end_time,51))
                    if excursion > self.cfg.maximum_touchdown_error_m:
                        raise ValueError('Landing reference excursion exceeds12mm bound')
                    self.landing_original_contact_error_m = original_error
                    self.landing_target_excursion_m = excursion
                    self.landing_trigger_s = self.time
                    self.landing_contact_origin = measured['reference_point_world_m'][i].copy()
                    self.landing_preload_world_m = endpoint-self.landing_contact_origin
                    self.landing_endpoint_correction_m = correction
                    self.landing_contact_gap_steps = 0; self.contact_count = 0
                    self.mode = 'landing_blend'
                trajectory = self.landing if self.landing is not None else self.swing
                end = trajectory.end_time
                if self.landing is not None:
                    if measured['contact'][i]:
                        self.landing_contact_gap_steps = 0
                        if np.linalg.norm(measured['reference_point_world_m'][i]-self.landing_contact_origin) > self.cfg.maximum_touchdown_error_m:
                            raise ValueError('Measured landing foot moved beyond bounded contact region')
                        if self.time >= end-1e-8:
                            # Stable measured support only; a timer/first touch
                            # alone never completes the step or updates anchors.
                            new_preload = self.landing.end-measured['reference_point_world_m'][i]
                            consistency_error = np.linalg.norm(new_preload-self.landing_preload_world_m)
                            if (consistency_error > self.cfg.maximum_touchdown_error_m or actual_speed > .04
                                    or np.linalg.norm(new_preload) > self.cfg.maximum_initial_preload_point_offset_m):
                                raise ValueError('Landing completion disagrees with measured position/speed')
                            self.contact_count += 1
                            if self.contact_count >= self.cfg.contact_confirm_steps:
                                self.anchors[i] = self.landing.end.copy()
                                self.measured_anchors[i] = measured['reference_point_world_m'][i].copy()
                                self.preload_world[i] = self.anchors[i]-self.measured_anchors[i]
                                self.current_leg = None; self.swing = None; self.landing = None
                                self.mode = 'contact_hold'; self.hold_until = self.time+self.cfg.contact_hold_s
                                self.touchdowns += 1
                    else:
                        self.landing_contact_gap_steps += 1; self.contact_count = 0
                        if self.landing_contact_gap_steps*dt > self.cfg.maximum_landing_contact_gap_s+1e-9:
                            raise ValueError('Landing contact loss exceeds bounded100ms reacquisition interval')
                    if self.current_leg is not None:
                        self.mode = 'landing_blend' if self.time < end else 'awaiting_landing_support'
                if self.current_leg is not None and self.time >= end:
                    if self.landing is None:self.mode = 'awaiting_contact'
                    if not measured['contact'][i]:
                        target = np.zeros(3); factor = 0.
                    if not self.flight_seen or self.time-end > self.cfg.max_touchdown_delay_s:
                        raise ValueError('Liftoff not observed or measured touchdown missing')
            speed = max(np.linalg.norm(target[:2]), abs(target[2])*.30)
            if self.current_leg is None and self.time >= self.hold_until and speed > 1e-6:
                i = LEGS.index(self.cfg.gait_order[self.order_index])
                self._support(measured, i)
                if not measured['contact'][i]:
                    raise ValueError('Cannot launch next wave leg without its measured initial contact')
                cycle = 6*(self.cfg.swing_s+self.cfg.contact_hold_s)
                horizon = self.cfg.swing_s+.5*(cycle-self.cfg.swing_s)
                predicted_p, predicted_R = self._predict(target, horizon)
                endpoint = predicted_R @ self.neutral[i]+predicted_p
                endpoint[2] = self.anchors[i, 2]  # Flat proof only; preserve initial virtual toe plane/preload.
                self.swing = Swing(self.time, self.cfg.swing_s, self.anchors[i], endpoint, self.cfg.lift_m)
                self.current_leg = i; self.mode = 'swing'; self.flight_seen = False; self.flight_count = 0; self.contact_count = 0
                self.landing = None; self.landing_trigger_s = None; self.landing_contact_origin = None
                self.landing_endpoint_correction_m = None; self.landing_contact_gap_steps = 0
                self.landing_original_contact_error_m = None; self.landing_target_excursion_m = None
                self.landing_preload_world_m = None
                self.flight_baseline_z = float(measured['reference_point_world_m'][i, 2])
                self.flight_peak_z = self.flight_baseline_z; self.descent_seen = False
                self.order_index = (self.order_index+1)%6; self.liftoffs += 1
            self.factor = factor; self.command_target = target.copy()
            finite_stop = (np.linalg.norm(requested) == 0 and self.current_leg is None
                           and np.max(np.abs(self.command)) <= self.cfg.stop_command_tolerance
                           and np.max(np.abs(self.command_rate)) <= self.cfg.stop_command_rate_tolerance)
            if finite_stop:
                # Explicit tiny-motion terminal knot. The following q/v/a check
                # still rejects a discontinuity outside the executable budget.
                self.command[:] = 0.; self.command_rate[:] = 0.
            self.position, self.yaw, self.command, self.command_rate = self._advance(target, dt)
            self.time += dt
            points = self.anchors.copy()
            if self.current_leg is not None:
                trajectory = self.landing if self.landing is not None else self.swing
                points[self.current_leg] = trajectory.sample(self.time)[0]
            body = self._body()
            local = (body.rotation_wb.T @ (points-self.position).T).T
            ik = self.g.ik(tensor(local[None]))
            if not bool(ik['valid'].all()):
                raise ValueError('Reference foot target unreachable; IK clipping is diagnostic only')
            q = ik['q_checked'][0].numpy()
            # A quiet hold is the existing executed knot, not a repeated IK
            # solution that can remove preload through rounded URDF transforms.
            if self.current_leg is None and np.max(np.abs(self.command)) < 1e-14 and np.max(np.abs(self.command_rate)) < 1e-12:
                q = self.q.copy()
            v = (q-self.q)/dt; a = (v-self.v)/dt
            self._bounds(q, v, a)
            self.last_diagnostics = dict(measured_projected_COM_support_margin_m=margin,
                body_tracking_error_m=body_error, measured_stance_toe_drift_m=drift,
                reference_point_world_m=points.copy(), max_reference_velocity_rad_s=float(np.max(np.abs(v))),
                max_reference_acceleration_rad_s2=float(np.max(np.abs(a))),
                minimum_joint_margin_rad=float(np.minimum(q-self.lower, self.upper-q).min()))
            self.q = q; self.v = v
            if self.current_leg is None and np.linalg.norm(requested) == 0:
                self.mode = 'reference_quiet_hold' if finite_stop else 'stopping_reference_motion'
                if finite_stop and self.reference_quiet_time is None:
                    self.reference_quiet_time = self.time
            return self._output(q, v, a, measured)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            self.failure = str(exc)
            return self._output(self.q, self.v, np.zeros_like(self.q), measured)
