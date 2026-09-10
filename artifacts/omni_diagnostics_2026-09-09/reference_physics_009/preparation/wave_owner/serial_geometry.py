"""Exact frozen C serial geometry; local byte-identical URDF/reference inputs."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
BENCH=Path(__file__).resolve().parent/'geometry'
DTYPE=torch.float64
def tensor(x):return torch.as_tensor(x,dtype=DTYPE,device='cpu')

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
