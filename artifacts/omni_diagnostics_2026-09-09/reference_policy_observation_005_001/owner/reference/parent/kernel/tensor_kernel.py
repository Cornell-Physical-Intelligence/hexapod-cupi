"""Device-resident C-study geometry kernels. No contact FSM, actor, or physics.

Static file identity and joint-name checks happen once at construction. Dynamic
shape/dtype/device checks raise; numerical failures return per-replica masks.
Diagnostic IK solutions are never executable until every upstream and local
gate has passed. No .cpu(), .numpy(), .item(), or scalar tensor truth is used
on the dynamic path. Float64 only: changing precision needs separate evidence.
"""
from pathlib import Path
import hashlib
import json
import math
import torch

HERE = Path(__file__).resolve().parent
DT = .02
PROFILES = {'formal_004': (2., 1.75), 'diagnostic_003': (1.5, 1.25)}


def finite_rows(value):
    return torch.isfinite(value).reshape(value.shape[0], -1).all(-1)


class TensorGeometry:
    def __init__(self, joint_names_runtime, *, binding, profile, device='cpu'):
        contract = json.loads((HERE / 'source_contract.json').read_text())
        for relative, expected in contract['source_inputs'].items():
            if hashlib.sha256((HERE / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source input mismatch: ' + relative)
        if binding not in contract['bindings'] or profile not in PROFILES:
            raise ValueError('Explicit 5/7mm source binding and .03/.04 profile required')
        self.binding = binding
        self.profile = profile
        self.configuration = contract['bindings'][binding]['configuration']
        self.identity = dict(binding=binding, profile=profile,
            controller_sha256=contract['bindings'][binding]['controller_sha256'],
            source_contract_sha256=hashlib.sha256((HERE/'source_contract.json').read_bytes()).hexdigest(),
            dtype='float64', contact_state_machine_implemented=False,
            physical_admission=False, policy_training_allowed=False)
        c = json.loads((HERE / 'geometry_constants.json').read_text())
        self.names = tuple(joint_names_runtime)
        self.leg_names = tuple(c.pop('names_leg_major'))
        if len(self.names) != 18 or len(set(self.names)) != 18 or set(self.names) != set(self.leg_names):
            raise ValueError('Exact18 unique named C-study joints required')
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.to_leg = torch.tensor([self.names.index(n) for n in self.leg_names], device=self.device)
        self.to_runtime = torch.argsort(self.to_leg)
        # Encoder nominal matches the admitted plan's float32 runtime values;
        # geometric q0 retains the original JSON angles at full precision.
        observation_contract = json.loads((HERE/'oracle/observation002/source_contract.json').read_text())
        self.observation_nominal = torch.tensor([observation_contract['nominal_joint_positions'][n] for n in self.names],
                                                device=self.device, dtype=self.dtype)
        for key, value in c.items():
            setattr(self, key, torch.tensor(value, dtype=self.dtype, device=self.device))
        self.eye3 = torch.eye(3, dtype=self.dtype, device=self.device)
        self.eye4 = torch.eye(4, dtype=self.dtype, device=self.device)
        a = self.axes
        self.skew = torch.zeros((6, 3, 3, 3), dtype=self.dtype, device=self.device)
        self.skew[..., 0, 1], self.skew[..., 0, 2], self.skew[..., 1, 0] = -a[..., 2], a[..., 1], a[..., 2]
        self.skew[..., 1, 2], self.skew[..., 2, 0], self.skew[..., 2, 1] = -a[..., 0], -a[..., 1], a[..., 0]
        self.skew2 = self.skew @ self.skew
        self.total_velocity, self.reference_velocity = PROFILES[profile]

    def check(self, value, shape, name, dtype=None):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.device != self.device or value.dtype != (dtype or self.dtype)):
            raise ValueError('Wrong tensor shape/dtype/device: ' + name)
        return value

    def leg(self, runtime):
        self.check(runtime, (runtime.shape[0], 18), 'runtime joints')
        return runtime.index_select(-1, self.to_leg).reshape(-1, 6, 3)

    def runtime(self, leg):
        self.check(leg, (leg.shape[0], 6, 3), 'leg-major joints')
        return leg.flatten(1).index_select(-1, self.to_runtime)

    def rotation_valid(self, rotation):
        finite = finite_rows(rotation)
        clean = torch.where(finite[:, None, None], rotation, self.eye3)
        orthogonal = (clean.transpose(-1, -2) @ clean - self.eye3).abs().amax((-1, -2)) <= 2e-4
        proper = (torch.linalg.det(clean)-1).abs() <= 2e-4
        return finite & orthogonal & proper

    def fk(self, q_leg):
        n = q_leg.shape[0]
        self.check(q_leg, (n, 6, 3), 'FK q')
        valid = finite_rows(q_leg)
        q = torch.where(valid[:, None, None], q_leg, 0.)
        T = self.eye4.expand(n, 6, 4, 4).clone()
        pivots, axes, transforms = [], [], []
        # Three chain coordinates, never a replica loop.
        for i in range(3):
            T = T @ self.origins[:, i]
            pivots.append(T[..., :3, 3])
            axes.append((T[..., :3, :3] @ self.axes[:, i, :, None]).squeeze(-1))
            angle = q[..., i, None, None]
            R = self.eye3 + angle.sin()*self.skew[:, i] + (1-angle.cos())*self.skew2[:, i]
            Q = self.eye4.expand(n, 6, 4, 4).clone()
            Q[..., :3, :3] = R
            T = T @ Q
            transforms.append(T)
        foot = (T[..., :3, :3] @ self.toes[..., None]).squeeze(-1) + T[..., :3, 3]
        J = torch.stack([torch.cross(a, foot-p, dim=-1) for a, p in zip(axes, pivots)], -1)
        return dict(feet_body_m=foot, jacobian_m=J, transforms=torch.stack(transforms, 2), valid=valid)

    def ik(self, targets_body_m, *, tolerance=2e-5):
        n = targets_body_m.shape[0]
        self.check(targets_body_m, (n, 6, 3), 'IK targets')
        if not math.isfinite(tolerance) or tolerance <= 0 or tolerance > 2e-5:
            raise ValueError('IK tolerance cannot exceed the scalar oracle')
        finite_target = torch.isfinite(targets_body_m).all(-1)
        # Quarantine nonfinite limbs before SVD, without altering their mask.
        targets = torch.where(finite_target[..., None], targets_body_m, 0.)
        d = targets-self.yaw_origin
        lateral = ((self.hip+self.tip)*self.tangent).sum(-1)
        rho2 = d[..., :2].square().sum(-1)
        radial2 = rho2-lateral.square()
        radius = radial2.clamp_min(0).sqrt()-(self.hip*self.radial).sum(-1)
        height = d[..., 2]-self.hip[..., 2]
        ty, tz = (self.tip*self.radial).sum(-1), self.tip[..., 2]
        K = torch.sqrt(ty.square()+tz.square())
        cosine = (radius.square()+height.square()-self.length.square()-K.square())/(2*self.length*K)
        delta = cosine.clamp(-1, 1).acos()
        yaw = torch.atan2(d[..., 1], d[..., 0]) - (lateral/rho2.clamp_min(1e-20).sqrt()).clamp(-1, 1).asin() - self.radial_angle
        yaw = torch.remainder(yaw+math.pi, 2*math.pi)-math.pi
        q = torch.stack((yaw, torch.atan2(height, radius)+torch.atan2(K*delta.sin(), self.length+K*delta.cos()),
                         delta+torch.atan2(tz, ty)), -1)
        finite = finite_target & torch.isfinite(q).all(-1)
        in_limits = ((q >= self.lower) & (q <= self.upper)).all(-1)
        checked = torch.nan_to_num(q).maximum(self.lower).minimum(self.upper)
        forward = self.fk(checked)
        error = (forward['feet_body_m']-targets).norm(dim=-1)
        error = torch.where(finite_target, error, torch.inf)
        sigma = torch.linalg.svdvals(forward['jacobian_m'])[..., -1]
        reachable = (radial2 > 0) & (cosine.abs() <= 1) & finite_target
        valid = finite & reachable & in_limits & (error <= tolerance) & (sigma > .002)
        return dict(q_checked=checked, valid=valid, reachable=reachable, in_limits=in_limits & finite_target,
            finite=finite, error_m=error, min_jacobian_singular_value_m=sigma,
            minimum_joint_margin_rad=torch.minimum(checked-self.lower, self.upper-checked).amin(-1))

    def com(self, q_runtime, position_world_m, rotation_world_from_body):
        n = q_runtime.shape[0]
        self.check(position_world_m, (n, 3), 'COM position')
        self.check(rotation_world_from_body, (n, 3, 3), 'COM rotation')
        f = self.fk(self.leg(q_runtime))
        T = f['transforms']
        points = (T[..., :3, :3] @ self.link_com[..., None]).squeeze(-1)+T[..., :3, 3]
        local = ((points*self.link_mass[..., None]).sum((1, 2))+self.body_mass*self.body_com)/self.mass
        world = (rotation_world_from_body @ local[..., None]).squeeze(-1)+position_world_m
        valid = f['valid'] & finite_rows(position_world_m) & self.rotation_valid(rotation_world_from_body) & finite_rows(world)
        return dict(body_m=local, world_m=world, valid=valid)

    def polynomial(self, coefficients, start_s, duration_s, lift_m, time_s, active, *, world_up):
        """Batched original Swing.sample, also usable for landing with lift=0.

        An inactive trajectory uses finite all-zero sentinels. Preserve the
        original swing and landing separately; this method chooses neither.
        """
        n = coefficients.shape[0]
        for value, shape, name in ((coefficients, (n, 6, 3), 'coefficients'), (start_s, (n,), 'start'),
                (duration_s, (n,), 'duration'), (lift_m, (n,), 'lift'), (time_s, (n,), 'time'), (world_up, (n, 3), 'world up')):
            self.check(value, shape, name)
        self.check(active, (n,), 'trajectory active', torch.bool)
        finite = finite_rows(coefficients) & torch.isfinite(start_s+duration_s+lift_m+time_s) & finite_rows(world_up)
        valid_active = finite & (duration_s > 0) & (lift_m >= 0) & ((world_up.norm(dim=-1)-1).abs() <= 1e-6)
        sentinel = (coefficients == 0).all((-1, -2)) & (start_s == 0) & (duration_s == 0) & (lift_m == 0)
        valid = torch.where(active, valid_active, finite & sentinel)
        safe = active & valid
        coeff = torch.where(safe[:, None, None], coefficients, 0.)
        duration = torch.where(safe, duration_s, 1.)
        u = ((torch.where(safe, time_s-start_s, 0.))/duration).clamp(0, 1)
        k = torch.arange(6, device=self.device, dtype=self.dtype)
        p = ((u[:, None]**k)[..., None]*coeff).sum(1)
        v = ((k[1:]*u[:, None]**(k[1:]-1))[..., None]*coeff[:, 1:]).sum(1)/duration[:, None]
        a = ((k[2:]*(k[2:]-1)*u[:, None]**(k[2:]-2))[..., None]*coeff[:, 2:]).sum(1)/duration[:, None].square()
        lift = torch.where(safe, lift_m, 0.)
        p = p + (lift*64*(u**3-3*u**4+3*u**5-u**6))[:, None]*world_up
        v = v + (lift*(192*u**2-768*u**3+960*u**4-384*u**5)/duration)[:, None]*world_up
        a = a + (lift*(384*u-2304*u**2+3840*u**3-1920*u**4)/duration.square())[:, None]*world_up
        return dict(position_world_m=p, velocity_world_mps=v, acceleration_world_mps2=a,
                    valid=valid & finite_rows(p) & finite_rows(v) & finite_rows(a), active=active)


class ReferenceKnots:
    """Only executable reference P/V history and local geometry/rate checks.

    upstream_valid MUST contain the independent contact, support, body tracking,
    drift, landing, stop and timer checks. Those gates are NOT implemented here.
    A false row latches failed until a fresh explicit reset. Invalid q_out is
    NaN, while q_checked remains clearly labelled diagnostic information.
    """
    def __init__(self, geometry, num_envs):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError('Positive replica count required')
        self.g, self.n = geometry, num_envs
        g = geometry
        with torch.inference_mode(False):
            self.q = torch.zeros((num_envs, 18), device=g.device, dtype=g.dtype)
            self.v = torch.zeros_like(self.q)
            self.lower = torch.zeros_like(self.q)
            self.upper = torch.zeros_like(self.q)
            self.time = torch.zeros(num_envs, device=g.device, dtype=g.dtype)
            self.episode = torch.full((num_envs,), -1, device=g.device, dtype=torch.int64)
            self.ready = torch.zeros(num_envs, device=g.device, dtype=torch.bool)
            self.failed = torch.zeros_like(self.ready)

    def reset(self, selected, episode, time_s, q_runtime, soft_limits_runtime, executable_velocity_runtime):
        g, n = self.g, self.n
        for value, shape, name, dtype in ((selected, (n,), 'reset selection', torch.bool),
                (episode, (n,), 'fresh episode counters', torch.int64), (time_s, (n,), 'reset time', g.dtype),
                (q_runtime, (n, 18), 'reset q', g.dtype), (soft_limits_runtime, (n, 18, 2), 'soft limits', g.dtype),
                (executable_velocity_runtime, (n, 18), 'reset executed velocity', g.dtype)):
            g.check(value, shape, name, dtype)
        lower = torch.maximum(soft_limits_runtime[..., 0], g.runtime(g.lower.expand(n, -1, -1)))
        upper = torch.minimum(soft_limits_runtime[..., 1], g.runtime(g.upper.expand(n, -1, -1)))
        ok = ((episode > self.episode) & (episode >= 0) & torch.isfinite(time_s) & finite_rows(q_runtime)
              & finite_rows(soft_limits_runtime) & (lower < upper).all(-1)
              & finite_rows(executable_velocity_runtime) & (executable_velocity_runtime.abs().amax(-1) <= 1e-6)
              & (q_runtime >= lower+.02).all(-1) & (q_runtime <= upper-.02).all(-1))
        commit = selected & ok
        self.q.copy_(torch.where(commit[:, None], q_runtime, self.q))
        self.v.copy_(torch.where(commit[:, None], 0., self.v))
        self.lower.copy_(torch.where(commit[:, None], lower, self.lower))
        self.upper.copy_(torch.where(commit[:, None], upper, self.upper))
        self.time.copy_(torch.where(commit, time_s, self.time))
        self.episode.copy_(torch.where(commit, episode, self.episode))
        self.ready |= commit
        self.failed &= ~commit
        self.ready &= ~(selected & ~ok)
        self.failed |= selected & ~ok
        return dict(reset_accepted=commit, reset_rejected=selected & ~ok)

    def advance(self, targets_world_m, desired_position_world_m, desired_rotation_world_from_body,
                sample_time_s, quiet_hold, upstream_valid):
        g, n = self.g, self.n
        for value, shape, name, dtype in ((targets_world_m, (n, 6, 3), 'reference feet', g.dtype),
                (desired_position_world_m, (n, 3), 'desired position', g.dtype),
                (desired_rotation_world_from_body, (n, 3, 3), 'desired rotation', g.dtype),
                (sample_time_s, (n,), 'input sample time', g.dtype), (quiet_hold, (n,), 'quiet hold', torch.bool),
                (upstream_valid, (n,), 'upstream gate mask', torch.bool)):
            g.check(value, shape, name, dtype)
        local = (targets_world_m-desired_position_world_m[:, None, :]) @ desired_rotation_world_from_body
        ik = g.ik(local)
        q = g.runtime(ik['q_checked'])
        q = torch.where(quiet_hold[:, None], self.q, q)
        v = (q-self.q)/DT
        a = (v-self.v)/DT
        timing = torch.isfinite(sample_time_s) & ((sample_time_s-self.time).abs() <= 1e-6)
        pose = finite_rows(desired_position_world_m) & g.rotation_valid(desired_rotation_world_from_body)
        bounds = (q >= self.lower+.02).all(-1) & (q <= self.upper-.02).all(-1)
        rates = (v.abs().amax(-1) <= g.reference_velocity+1e-5) & (a.abs().amax(-1) <= 6.+1e-5)
        ok = self.ready & ~self.failed & upstream_valid & ik['valid'].all(-1) & pose & timing & bounds & rates
        self.failed |= ~ok
        self.q.copy_(torch.where(ok[:, None], q, self.q))
        self.v.copy_(torch.where(ok[:, None], v, self.v))
        self.time.copy_(torch.where(ok, self.time+DT, self.time))
        return dict(valid=ok, q_out=torch.where(ok[:, None], q, torch.nan),
                    v_out=torch.where(ok[:, None], v, torch.nan), a_out=torch.where(ok[:, None], a, torch.nan),
                    q_checked=q, ik=ik, timing_valid=timing, pose_valid=pose,
                    joint_margin_valid=bounds, discrete_rates_valid=rates, upstream_valid=upstream_valid)
