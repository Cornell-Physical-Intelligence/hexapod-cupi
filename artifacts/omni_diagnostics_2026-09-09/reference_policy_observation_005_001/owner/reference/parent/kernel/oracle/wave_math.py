"""Endpoint-matched swing copied from frozen support002; no physics writes."""
import numpy as np
import math
from dataclasses import dataclass
from scipy.spatial import ConvexHull
import xml.etree.ElementTree as ET


class MassGeometry:
    """Frozen serial masses/COMs for projected support geometry only."""
    def __init__(self, geometry):
        xml = ET.parse(geometry.urdf_path).getroot()
        joints = {joint.get('name'): joint for joint in xml.findall('joint')}
        links = {link.get('name'): link for link in xml.findall('link')}
        self.link_mass = np.zeros((6, 3)); self.link_com = np.zeros((6, 3, 3))
        for i, names in enumerate(geometry.names):
            for j, name in enumerate(names):
                inertial = links[joints[name].find('child').get('link')].find('inertial')
                self.link_mass[i, j] = float(inertial.find('mass').get('value'))
                self.link_com[i, j] = np.fromstring(inertial.find('origin').get('xyz'), sep=' ')
        inertial = links['body_mock'].find('inertial')
        self.body_mass = float(inertial.find('mass').get('value'))
        self.body_com = np.fromstring(inertial.find('origin').get('xyz'), sep=' ')
        self.mass = self.body_mass+self.link_mass.sum()

    def support_margin(self, feet, indices, point):
        if len(indices) < 3:
            return -math.inf
        try:
            hull = ConvexHull(feet[list(indices), :2])
            return float((-hull.equations[:, :2] @ point[:2]-hull.equations[:, 2]).min())
        except Exception:
            return -math.inf
def rotation(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

@dataclass
class BodyState:
    time_s:float
    position_w:np.ndarray
    rotation_wb:np.ndarray
    velocity_w:np.ndarray
    acceleration_w:np.ndarray
    omega_w:np.ndarray
    angular_acceleration_w:np.ndarray
    provenance:str='ideal_command-integrated_fixture_not_state_estimation'

class Swing:
    """World-frame quintic Hermite endpoint matching plus a C2 lift bump."""
    def __init__(self,t0,duration,p0,p1,lift,v0=None,a0=None):
        if duration<=0:raise ValueError('Positive swing duration required')
        self.t0=t0;self.duration=duration;self.end=np.array(p1,dtype=float);self.lift=lift
        p0=np.asarray(p0);v0=np.zeros(3) if v0 is None else np.asarray(v0);a0=np.zeros(3) if a0 is None else np.asarray(a0)
        coeff=np.zeros((6,3));coeff[0]=p0;coeff[1]=duration*v0;coeff[2]=.5*duration**2*a0
        # Terminal world velocity/acceleration are zero, matching planted stance.
        rhs=np.stack((self.end-coeff[:3].sum(0),-coeff[1]-2*coeff[2],-2*coeff[2]))
        coeff[3:]=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20.]]),rhs)
        self.coeff=coeff
    def sample(self,time_s):
        u=np.clip((time_s-self.t0)/self.duration,0,1);k=np.arange(6)
        p=(u**k)@self.coeff
        v=(k[1:]*u**(k[1:]-1))@self.coeff[1:]/self.duration
        a=(k[2:]*(k[2:]-1)*u**(k[2:]-2))@self.coeff[2:]/self.duration**2
        p[2]+=self.lift*64*(u**3-3*u**4+3*u**5-u**6)
        v[2]+=self.lift*(192*u**2-768*u**3+960*u**4-384*u**5)/self.duration
        a[2]+=self.lift*(384*u-2304*u**2+3840*u**3-1920*u**4)/self.duration**2
        return p,v,a
    @property
    def end_time(self):return self.t0+self.duration
