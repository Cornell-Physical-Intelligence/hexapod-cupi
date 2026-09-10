"""Isolated CPU/tensor contingency. No simulator, actuator or learned policy.

Axes: command=(forward,left,yaw); body=(-Y,+X,+Z). Explicit serial-C-study
kinematics only. A valid kinematic reference is not a dynamically admitted gait.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

DTYPE = torch.float64
ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / 'artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300'
LEGS = ('lf', 'lm', 'lr', 'rf', 'rm', 'rr')

def tensor(x):
    if not isinstance(x, torch.Tensor):
        x = np.asarray(x)
    return torch.as_tensor(x, dtype=DTYPE, device='cpu')

def sinc_unscaled(x):
    """sin(x)/x with finite first/second derivatives at zero in Torch 2.8."""
    small = x.abs() < 1e-3
    safe = torch.where(small, torch.ones_like(x), x)
    polynomial = 1-x.square()/6+x**4/120-x**6/5040
    return torch.where(small, polynomial, x.sin()/safe)

def smootherstep(x):
    u = x.clamp(0., 1.)
    return u**3 * (10. + u * (-15. + 6. * u))

def nav_to_body(command):
    return torch.stack((command[..., 1], -command[..., 0]), -1)

def inverse_twist_flow(points, command, virtual_time):
    """Point coordinates after inverse constant SE(2) body motion.

    points=(6,3), command=(B,3), virtual_time=(B,6). Smooth at yaw=0.
    Negative virtual time is allowed; nothing moves the robot in physics.
    """
    velocity = nav_to_body(command)[:, None, :]
    angle = command[:, None, 2] * virtual_time
    a = virtual_time * sinc_unscaled(angle)
    b = virtual_time * .5 * angle * sinc_unscaled(angle/2).square()
    translation = torch.stack((a * velocity[..., 0] - b * velocity[..., 1],
                               b * velocity[..., 0] + a * velocity[..., 1]), -1)
    relative = points[None, :, :2] - translation
    c, s = angle.cos(), angle.sin()
    return torch.stack((c * relative[..., 0] + s * relative[..., 1],
                        -s * relative[..., 0] + c * relative[..., 1]), -1)

class SerialGeometry:
    """Read exact immutable benchmark URDF/foot points, derive named 3R chains."""
    def __init__(self):
        self.reference = json.loads((BENCH / 'candidate_c_reference.json').read_text())
        self.urdf_path = BENCH / 'f050_t060.urdf'
        self.urdf_sha = hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()
        if self.urdf_sha != self.reference['urdf_sha256']:
            raise ValueError('Frozen benchmark URDF identity mismatch')
        xml = ET.parse(self.urdf_path).getroot()
        joint_lookup = {j.get('name'): j for j in xml.findall('joint')}
        # Reference names explicitly define leg-major coxa/femur/tibia; check chains.
        self.names = [self.reference['joint_names'][3*i:3*i+3] for i in range(6)]
        origins, axes, limits = [], [], []
        for names in self.names:
            joints = [joint_lookup[name] for name in names]
            assert joints[0].find('parent').get('link') == 'body_mock'
            assert all(joints[i].find('child').get('link') == joints[i+1].find('parent').get('link') for i in range(2))
            leg_o, leg_a, leg_l = [], [], []
            for j in joints:
                o = j.find('origin'); T = np.eye(4)
                T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(o.get('rpy'), sep=' ')).as_matrix()
                T[:3, 3] = np.fromstring(o.get('xyz'), sep=' ')
                leg_o.append(T); leg_a.append(np.fromstring(j.find('axis').get('xyz'), sep=' '))
                leg_l.append([float(j.find('limit').get(k)) for k in ('lower','upper')])
            origins.append(leg_o); axes.append(leg_a); limits.append(leg_l)
        self.origins, self.axes = tensor(origins), tensor(axes)
        hard = tensor(limits); margin = .025 * (hard[...,1]-hard[...,0])
        self.lower, self.upper = hard[...,0]+margin, hard[...,1]-margin
        self.q0 = tensor(self.reference['stance']['q'])
        self.toes = tensor(self.reference['stance']['toes'])
        self.feet0, _, _ = self.fk(self.q0)
        _, _, transforms = self.fk(torch.zeros_like(self.q0))
        self.yaw_origin = transforms[0][..., :3, 3]
        self.radial = transforms[1][..., :3, 1]
        self.tangent = torch.cross(tensor([0,0,1]).expand_as(self.radial), self.radial, dim=-1)
        self.hip = transforms[1][..., :3, 3] - self.yaw_origin
        self.length = ((transforms[2][..., :3, 3] - transforms[1][..., :3, 3]) * self.radial).sum(-1)
        self.tip = (transforms[2][..., :3, :3] @ self.toes[..., None]).squeeze(-1)
        self.radial_angle = torch.atan2(self.radial[...,1], self.radial[...,0])

    def fk(self, q):
        """Exact URDF transforms and geometric Jacobian; q=(...,6,3)."""
        batch = q.shape[:-2]
        T = torch.eye(4, dtype=DTYPE).expand(*batch, 6, 4, 4).clone()
        pivots, axes, transforms = [], [], []
        for i in range(3):
            T = T @ self.origins[:, i]
            pivots.append(T[..., :3, 3]); axes.append((T[..., :3, :3] @ self.axes[:,i,:,None]).squeeze(-1))
            a = self.axes[:, i]
            K = torch.zeros((6,3,3), dtype=DTYPE)
            K[:,0,1], K[:,0,2], K[:,1,0] = -a[:,2], a[:,1], a[:,2]
            K[:,1,2], K[:,2,0], K[:,2,1] = -a[:,0], -a[:,1], a[:,0]
            angle = q[...,i,None,None]
            R = torch.eye(3,dtype=DTYPE) + angle.sin()*K + (1-angle.cos())*(K@K)
            Q = torch.eye(4,dtype=DTYPE).expand(*batch,6,4,4).clone()
            Q[..., :3, :3] = R
            T = T @ Q; transforms.append(T)
        foot = (T[..., :3, :3] @ self.toes[...,None]).squeeze(-1) + T[..., :3,3]
        J = torch.stack([torch.cross(a, foot-p, dim=-1) for a,p in zip(axes,pivots)], -1)
        return foot, J, transforms

    def ik(self, targets, tolerance=2e-5):
        """Serial knee branch, checked against exact URDF FK and soft limits.

        Rejects unreachable requests. Clipping below is only for finite diagnostic
        FK; q_checked must never be executed when valid is false.
        """
        d = targets - self.yaw_origin
        lateral = ((self.hip + self.tip) * self.tangent).sum(-1)
        rho2 = d[...,:2].square().sum(-1)
        radial2 = rho2 - lateral.square()
        radius = radial2.clamp_min(0).sqrt() - (self.hip*self.radial).sum(-1)
        height = d[...,2]-self.hip[...,2]
        ty, tz = (self.tip*self.radial).sum(-1), self.tip[...,2]
        K = torch.sqrt(ty.square()+tz.square())
        cosine = (radius.square()+height.square()-self.length.square()-K.square())/(2*self.length*K)
        delta = cosine.clamp(-1,1).acos()
        yaw = torch.atan2(d[...,1],d[...,0]) - (lateral/rho2.clamp_min(1e-20).sqrt()).clamp(-1,1).asin() - self.radial_angle
        yaw = torch.remainder(yaw+math.pi,2*math.pi)-math.pi
        q = torch.stack((yaw,
                         torch.atan2(height,radius)+torch.atan2(K*delta.sin(),self.length+K*delta.cos()),
                         delta+torch.atan2(tz,ty)), -1)
        finite = torch.isfinite(targets).all(-1) & torch.isfinite(q).all(-1)
        in_limits = ((q>=self.lower)&(q<=self.upper)).all(-1)
        checked = torch.nan_to_num(q).maximum(self.lower).minimum(self.upper)
        foot,J,_ = self.fk(checked)
        error = (foot-targets).norm(dim=-1)
        sigma = torch.linalg.svdvals(J)[...,-1]
        reachable = (radial2>0)&(cosine.abs()<=1)
        valid = finite & reachable & in_limits & (error<=tolerance) & (sigma>.002)
        return dict(q_checked=checked, valid=valid, reachable=reachable, in_limits=in_limits,
                    error_m=error, min_jacobian_singular_value_m=sigma,
                    minimum_joint_margin_rad=torch.minimum(checked-self.lower,self.upper-checked).amin(-1))

@dataclass(frozen=True)
class Config:
    duty: float = .65
    stance_travel_m: float = .10
    lift_m: float = .02
    min_frequency_hz: float = .8
    lift_full_speed_mps: float = .04
    command_filter_omega_rad_s: float = 4.

class TwistReference:
    def __init__(self, geometry=None, config=Config()):
        self.geometry = geometry or SerialGeometry()
        self.config = config
        self.offsets = tensor([0,.5,0,.5,0,.5])

    def speed(self, command):
        p = self.geometry.feet0[:,:2]
        Jp = torch.stack((-p[:,1], p[:,0]), -1)
        local_speed = nav_to_body(command)[:,None,:] + command[:,None,2,None]*Jp
        return local_speed.square().sum(-1).mean(-1).clamp_min(1e-30).sqrt()

    def frequency(self, command):
        return (self.config.min_frequency_hz**2 + (self.config.duty*self.speed(command)/self.config.stance_travel_m).square()).sqrt()

    def feet(self, phase, command, frequency_override=None):
        """C2 periodic foot path; exact inverse-twist stance for constant commands.

        Changing filtered commands/frequency modifies stance coordinates smoothly
        but does NOT preserve world no-slip exactly during acceleration/reversal.
        """
        b = self.config.duty
        p = torch.remainder(phase[:,None]+self.offsets, 1.)
        stance = p < b
        u = ((p-b)/(1-b)).clamp(0,1)
        # dg/dphase=1 and d2g/dphase2=0 on both sides of each junction.
        g = torch.where(stance, p-b/2, b/2+(1-b)*u-smootherstep(u))
        f = self.frequency(command) if frequency_override is None else torch.ones_like(phase)*frequency_override
        xy = inverse_twist_flow(self.geometry.feet0, command, g/f[:,None])
        amplitude = smootherstep(self.speed(command)/self.config.lift_full_speed_mps)
        lift = self.config.lift_m*amplitude[:,None]*64*u**3*(1-u)**3
        z = self.geometry.feet0[None,:,2] + torch.where(stance,torch.zeros_like(lift),lift)
        return torch.cat((xy,z[...,None]),-1)

    def state(self, phase, command, command_rate=None):
        feet = self.feet(phase,command)
        ik = self.geometry.ik(feet)
        output = dict(feet=feet, frequency=self.frequency(command), **ik)
        if command_rate is not None:
            _,vel = torch.autograd.functional.jvp(self.feet,(phase,command),(output['frequency'],command_rate))
            _,J,_ = self.geometry.fk(ik['q_checked'])
            qvel = torch.linalg.solve(J,vel[...,None]).squeeze(-1)
            output.update(foot_velocity=vel,q_velocity=qvel,
                          velocity_within_urdf_limit=(qvel.abs()<=50.26548246).all(-1))
        return output

class FilterState:
    """Exact critically damped filter for held targets; position/rate are continuous.

    No phase reset on command reversal. No clipping of direction/curvature, no
    immediate output switch on a new requested target. Bounds need a separate
    dynamically qualified command governor, not hidden target projection.
    """
    def __init__(self, reference, count=1):
        self.reference = reference
        self.command = torch.zeros(count,3,dtype=DTYPE)
        self.rate = torch.zeros_like(self.command)
        self.phase = torch.zeros(count,dtype=DTYPE)

    def step(self, target, dt):
        target = tensor(target)
        if target.shape != self.command.shape or not torch.isfinite(target).all() or not 0<dt<=.1:
            raise ValueError('Finite matched body twist and 0 < dt <= 0.1 required')
        old_frequency = self.reference.frequency(self.command)
        omega = self.reference.config.command_filter_omega_rad_s
        error = self.command-target; a = self.rate+omega*error; decay = math.exp(-omega*dt)
        self.command = target+(error+a*dt)*decay
        self.rate = (self.rate-omega*a*dt)*decay
        self.phase = torch.remainder(self.phase + .5*dt*(old_frequency+self.reference.frequency(self.command)),1.)
        return self.reference.state(self.phase,self.command,self.rate)

def compose_joint_feedback(q_reference, residual_action, geometry, *, reference_valid, scale_rad=.12):
    """Feedback remains active at zero command; not an actuator controller.

    Any changed baseline or added velocity feedforward changes action semantics.
    The original 1.6 Nm cap and validated slew/gains must be preserved separately.
    """
    if not reference_valid.all() or not torch.isfinite(q_reference).all() or not torch.isfinite(residual_action).all():
        raise ValueError('Invalid reference or feedback; no joint target emitted')
    if residual_action.shape != q_reference.shape:
        raise ValueError('Reference and residual named-joint shapes must match')
    raw = q_reference + scale_rad*residual_action.clamp(-1,1)
    limited = raw.maximum(geometry.lower).minimum(geometry.upper)
    return limited, (raw!=limited)
