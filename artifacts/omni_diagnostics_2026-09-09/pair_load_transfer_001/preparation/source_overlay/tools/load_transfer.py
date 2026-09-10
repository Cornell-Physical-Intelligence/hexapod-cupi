"""Proposed CPU-only LM+RM load-transfer diagnostic. No physics state writes.

Four-support rules below belong only to this proposed diagnostic. The copied
single-leg wave controller and its five-support gates are never modified.
"""
from dataclasses import dataclass, asdict
import numpy as np
from source004_wave_helpers import WaveContactReference, LEGS, tensor

PAIR = (1, 4)  # LM, RM in the frozen named leg order.
CORNERS = (0, 2, 3, 5)


@dataclass(frozen=True)
class TransferConfig:
    lift_m: float = .007
    baseline_s: float = 2.
    ramp_s: float = 3.
    unloaded_hold_s: float = 2.
    quiet_s: float = 12.  # 2 s settling plus >=10 s measured quiet review.
    minimum_support_margin_m: float = .050
    maximum_body_displacement_m: float = .015
    maximum_body_angle_rad: float = .10
    maximum_corner_drift_m: float = .010
    maximum_corner_slip_mps: float = .020
    requested_torque_limit_nm: float = 1.6
    minimum_pair_measured_lift_m: float = .002
    maximum_unloaded_normal_force_n: float = 1.
    minimum_unloaded_hold_s: float = 1.
    proposal_only: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == 'proposal_only': continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError('Every diagnostic scalar must be finite and positive: ' + name)
        if not self.proposal_only or self.lift_m <= 0 or self.lift_m > .007 or self.ramp_s < 3.:
            raise ValueError('This proposal admits only positive lift <=7 mm with >=3 s ramps')
        if self.baseline_s < 2. or self.unloaded_hold_s < 2. or self.quiet_s < 12.:
            raise ValueError('Keep bounded baseline, load observation and quiet windows')
        if self.requested_torque_limit_nm != 1.6:
            raise ValueError('The current 1.6 N m torque contract is unchanged')


class Quintic:
    """C2 joint-space return; terminal velocity/acceleration exactly zero."""
    def __init__(self, t0, duration, q, v, a, end):
        if (isinstance(duration, bool) or not np.isfinite(duration) or duration <= 0 or not np.isfinite(t0)):
            raise ValueError('Finite positive curve duration and start time required')
        q, v, a, end = [np.asarray(value, dtype=float) for value in (q, v, a, end)]
        if any(value.shape != (18,) or not np.isfinite(value).all() for value in (q, v, a, end)):
            raise ValueError('Finite named 18-joint curve P/V/A/end required')
        self.t0, self.duration = float(t0), float(duration)
        self.end = end.copy()
        self.c = np.zeros((6, 18))
        self.c[0], self.c[1], self.c[2] = q, duration*v, .5*duration**2*a
        rhs = np.stack((end-self.c[:3].sum(0), -self.c[1]-2*self.c[2], -2*self.c[2]))
        self.c[3:] = np.linalg.solve([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]], rhs)

    def sample(self, t):
        if t >= self.t0+self.duration-1e-10:
            return self.end.copy(), np.zeros(18), np.zeros(18)
        u = max(0., (t-self.t0)/self.duration)
        k = np.arange(6)
        return (u**k @ self.c,
                (k[1:]*u**(k[1:]-1)) @ self.c[1:]/self.duration,
                (k[2:]*(k[2:]-1)*u**(k[2:]-2)) @ self.c[2:]/self.duration**2)


class PairLoadTransfer:
    """reset(source004_snapshot), step(snapshot, stop=False, dt=.02).

    The output matches the source004 reference setter's named [1,18] P/V/A
    interface. No source004 entrypoint currently dispatches this diagnostic.
    """
    def __init__(self, joint_names_runtime, config=TransferConfig()):
        self.base = WaveContactReference(joint_names_runtime)
        self.cfg = config
        self.ready = False
        self.failure = None

    def reset(self, snapshot):
        m = self.base._read(snapshot)
        if not m['contact'].all():
            raise ValueError('Diagnostic starts with all six measured distal contacts')
        canonical = self.base._runtime(self.base.g.q0.numpy())
        if np.max(np.abs(m['joint_target_rad']-canonical)) > 2e-7:
            raise ValueError('Diagnostic requires the exact current canonical C target after admitted startup')
        self.base.reset(snapshot)  # Exact executed target, soft limits and preload checks.
        self.q0 = m['joint_target_rad'].copy()
        self.q = self.q0.copy(); self.v = np.zeros(18); self.a = np.zeros(18)
        self.p0 = m['position_world_m'].copy(); self.R0 = m['rotation_world_from_body'].copy()
        self.measured_toes0 = m['reference_point_world_m'].copy()
        self.t0 = self.t = float(m['time_s'])
        self.analytic_v = np.zeros(18); self.analytic_a = np.zeros(18)
        self.return_curve = None; self.stop_time = None; self.reference_quiet_time = None
        self.contact_count = np.zeros(2, dtype=int)
        self.flight_count = np.zeros(2, dtype=int)
        self.flight_seen = np.zeros(2, bool)
        self.flight_baseline_z = self.measured_toes0[list(PAIR), 2].copy()
        self.peak_lift = np.zeros(2)
        self.last_contact_z = self.flight_baseline_z.copy()
        self.current_unloaded_s = self.maximum_unloaded_s = 0.
        self.elapsed_return_s = None
        self.failure = None; self.mode = 'baseline'; self.ready = True
        local = self.base.neutral.copy()
        local[list(PAIR)] += self.R0.T @ np.array([0., 0., self.cfg.lift_m])
        ik = self.base.g.ik(tensor(local))
        if not ik['valid'].all():
            raise ValueError('Pair lift endpoint is unreachable; no IK clipping allowed')
        qend = ik['q_checked'].numpy()
        # Other twelve motor targets remain exactly as emitted at reset.
        for i in CORNERS: qend[i] = self.base._leg(self.q0)[i]
        self.qend = self.base._runtime(qend)
        self.base._bounds(qend, np.zeros((6,3)), np.zeros((6,3)))
        self.lift_curve = Quintic(self.t0+self.cfg.baseline_s, self.cfg.ramp_s,
                                   self.q0, np.zeros(18), np.zeros(18), self.qend)
        self.nominal_return_s = self.t0+self.cfg.baseline_s+self.cfg.ramp_s+self.cfg.unloaded_hold_s
        self._measure(snapshot, m)
        return self._output(self.q, self.v, self.a)

    def _measure(self, snapshot, m):
        if m['terminal'] or m['base_contact'] or any(m[k].any() for k in ('shaft_contact','coxa_contact','femur_contact')):
            raise ValueError('Terminal or non-foot contact: stop this diagnostic')
        if not m['contact'][list(CORNERS)].all():
            raise ValueError('A required corner support is missing')
        com = self.base._com(self.base._leg(m['joint_position_rad']),m['position_world_m'],m['rotation_world_from_body'])
        margin = self.base.helper.support_margin(m['contact_point_world_m'],CORNERS,com)
        if margin < self.cfg.minimum_support_margin_m:
            raise ValueError('Measured four-corner projected support margin below proposed bound')
        displacement = np.linalg.norm(m['position_world_m']-self.p0)
        angle = np.arccos(np.clip((np.trace(self.R0.T@m['rotation_world_from_body'])-1)/2,-1,1))
        drift = np.linalg.norm(m['reference_point_world_m'][list(CORNERS)]-self.measured_toes0[list(CORNERS)],axis=-1).max()
        if displacement > self.cfg.maximum_body_displacement_m or angle > self.cfg.maximum_body_angle_rad:
            raise ValueError('Measured body displacement/rotation exceeds proposed diagnostic bound')
        if drift > self.cfg.maximum_corner_drift_m:
            raise ValueError('Measured corner-foot drift exceeds proposed diagnostic bound')
        requested = self.base._one(snapshot,'computed_torque_nm',(18,))
        applied = self.base._one(snapshot,'applied_torque_nm',(18,))
        normal = self.base._one(snapshot,'normal_force_world_n',(6,3))
        reaction = self.base._one(snapshot,'reaction_force_world_n',(6,3))
        slip = self.base._one(snapshot,'distal_contact_slip_mps',(6,))
        if np.abs(requested).max() > self.cfg.requested_torque_limit_nm:
            raise ValueError('Measured requested torque exceeds unchanged 1.6 N m contract')
        if np.abs(applied).max() > 1.60001:
            raise ValueError('Measured applied torque exceeds unchanged contract')
        if slip[list(CORNERS)].max() > self.cfg.maximum_corner_slip_mps:
            raise ValueError('Measured corner slip exceeds proposed diagnostic bound')
        self.diagnostics = dict(measured_support_margin_m=float(margin),body_displacement_m=float(displacement),
            body_rotation_rad=float(angle),corner_drift_m=float(drift),normal_force_world_n=normal.copy(),
            reaction_force_world_n=reaction.copy(),max_requested_torque_nm=float(abs(requested).max()),
            max_applied_torque_nm=float(abs(applied).max()),corner_max_slip_mps=float(slip[list(CORNERS)].max()))
        return normal

    def _trajectory(self, next_time, stop):
        if stop and self.stop_time is None:
            self.stop_time = self.t
        should_return = self.stop_time is not None or self.t >= self.nominal_return_s-1e-10
        if should_return and self.return_curve is None:
            self.return_curve = Quintic(self.t,self.cfg.ramp_s,self.q,self.analytic_v,self.analytic_a,self.q0)
        if self.return_curve is not None:
            if next_time >= self.return_curve.t0+self.return_curve.duration-1e-10:
                self.mode = 'reference_quiet_hold'
                self.reference_quiet_time = self.return_curve.t0+self.return_curve.duration
            else: self.mode = 'return'
            return self.return_curve.sample(next_time)
        if next_time <= self.t0+self.cfg.baseline_s+1e-10:
            self.mode = 'baseline'; return self.q0.copy(),np.zeros(18),np.zeros(18)
        if next_time < self.t0+self.cfg.baseline_s+self.cfg.ramp_s-1e-10:
            self.mode = 'unloading'; return self.lift_curve.sample(next_time)
        self.mode = 'unloaded_hold'; return self.qend.copy(),np.zeros(18),np.zeros(18)

    def step(self, snapshot, *, stop=False, dt=.02):
        if not self.ready: raise RuntimeError('Reset from a settled six-support snapshot first')
        if self.failure: return self._output(None,None,None)
        try:
            if dt != .02: raise ValueError('This proposed schedule is bound to 50 Hz')
            m = self.base._read(snapshot)
            if abs(float(m['time_s'])-self.t) > 1e-7: raise ValueError('Measured and reference times disagree')
            if np.max(np.abs(m['joint_target_rad']-self.q)) > 2e-7:
                raise ValueError('Executed target differs from preceding emitted reference')
            normal = self._measure(snapshot,m)
            for j,leg in enumerate(PAIR):
                if not m['contact'][leg]:
                    if self.flight_count[j] == 0: self.flight_baseline_z[j] = self.last_contact_z[j]
                    self.flight_count[j] += 1
                    self.peak_lift[j] = max(self.peak_lift[j],m['reference_point_world_m'][leg,2]-self.flight_baseline_z[j])
                    if self.flight_count[j] >= 2: self.flight_seen[j] = True
                    self.contact_count[j] = 0
                else:
                    self.flight_count[j] = 0
                    self.contact_count[j] += 1
                    self.last_contact_z[j] = m['reference_point_world_m'][leg,2]
            both_unloaded = (self.flight_seen.all() and not m['contact'][list(PAIR)].any()
                and (np.linalg.norm(normal[list(PAIR)],axis=-1) <= self.cfg.maximum_unloaded_normal_force_n).all()
                and (self.peak_lift >= self.cfg.minimum_pair_measured_lift_m).all())
            self.current_unloaded_s = self.current_unloaded_s+dt if both_unloaded and self.mode == 'unloaded_hold' else 0.
            self.maximum_unloaded_s = max(self.maximum_unloaded_s,self.current_unloaded_s)
            q,analytic_v,analytic_a = self._trajectory(self.t+dt,bool(stop))
            v=(q-self.q)/dt; a=(v-self.v)/dt
            self.base._bounds(self.base._leg(q),self.base._leg(v),self.base._leg(a))
            self.q,self.v,self.a=q,v,a
            self.analytic_v,self.analytic_a=analytic_v,analytic_a
            self.t += dt
            if self.reference_quiet_time is not None and self.t-self.reference_quiet_time >= .6:
                if not m['contact'].all() or (self.contact_count < 3).any():
                    raise ValueError('Six-support measured return not confirmed within 0.6 s')
            return self._output(q,v,a)
        except (ValueError,KeyError) as exc:
            self.failure=str(exc);self.mode='failed'
            return self._output(None,None,None)

    def _output(self,q,v,a):
        state=dict(mode=self.mode,proposal_only=True,config=asdict(self.cfg),
            pair_leg_names=[LEGS[i] for i in PAIR],required_support_leg_names=[LEGS[i] for i in CORNERS],
            measured_initial_body_position_world_m=self.p0.copy(),measured_initial_rotation_world_from_body=self.R0.copy(),
            initial_executed_target_rad=self.q0.copy(),initial_measured_toes_world_m=self.measured_toes0.copy(),
            initial_joint_preload_leg_major_rad=self.base.preload_q.copy(),
            virtual_target_endpoint_rad=self.qend.copy(),flight_seen=self.flight_seen.copy(),flight_count=self.flight_count.copy(),
            measured_peak_lift_m=self.peak_lift.copy(),stable_return_contact_samples=self.contact_count.copy(),
            longest_measured_unloaded_hold_s=self.maximum_unloaded_s,stop_requested_time_s=self.stop_time,
            reference_quiet_time_s=self.reference_quiet_time,diagnostics=self.diagnostics,
            return_curve=None if self.return_curve is None else dict(start_s=self.return_curve.t0,duration_s=self.return_curve.duration,coefficients=self.return_curve.c.copy()))
        return dict(q_ref=None if q is None else q[None],v_ref=None if v is None else v[None],a_ref=None if a is None else a[None],
            valid=np.array([self.failure is None]),target_time_s=self.t,failure_reason=self.failure,state=state,
            analytic_velocity_rad_s=self.analytic_v[None],analytic_acceleration_rad_s2=self.analytic_a[None],
            requested_body_twist=np.zeros(3),admitted_body_twist=np.zeros(3),body_pose_prescribed=False,
            diagnostic_outcome='not_yet_physically_scored',stage2_complete=False)
