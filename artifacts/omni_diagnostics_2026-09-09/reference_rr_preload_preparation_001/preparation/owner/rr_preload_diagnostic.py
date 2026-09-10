"""One RR first-landing target correction; no force estimation or physics writes.

Fixed diagnostic, not a tunable controller. It preserves raw measured contacts
and cannot replace the parent's five-support/contact/torque/stop gates.
"""
import math
import numpy as np

PROPOSAL = {
    'name': 'rr_first_confirmed_landing_preload_001',
    'leg': 'rr', 'leg_index': 5, 'offset_z_m': -.0005, 'duration_s': .30,
    'trigger': 'first RR measured landing completion, before target-time advance',
    'scope': 'one left-strafe case only; not an RM or arc correction',
    'source_of_amplitude': 'directional002 measured target-minus-toe Z change about0.492mm',
    'not_physics_admitted': True, 'old_actor_or_observation_binding_permitted': False,
}

class RRFirstLandingPreload:
    def __init__(self):
        self.started = False; self.completed = False; self.status = 'unarmed'
        self.start_s = None; self.end_s = None; self.anchor_at_start = None
        self.measured_anchor_at_start = None; self.original_swing_endpoint = None
        self.offset_applied_m = np.zeros(3); self.velocity_mps = np.zeros(3); self.acceleration_mps2 = np.zeros(3)
        self.changed_this_step = False; self.failure = None; self.armed_touchdown_count = None
        self.last_sample_time_s = None; self.maximum_observed_contact_displacement_m = 0.

    @property
    def active(self):return self.started and not self.completed and self.failure is None

    @staticmethod
    def _vec(x):
        a=np.asarray(x,dtype=float)
        if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Finite3-vector preload geometry required')
        return a.copy()

    def arm(self, leg, confirmed_before, time_s, hold_duration_s, anchor, measured_anchor, original_swing_endpoint):
        if leg != 5 or self.started:return False
        if confirmed_before != 1:raise ValueError('RR correction requires the original LF then firstRR landing sequence')
        if not math.isfinite(time_s) or not math.isfinite(hold_duration_s) or abs(hold_duration_s-.30)>1e-12:
            raise ValueError('RR diagnostic retains the exact existing0.3s hold')
        a=self._vec(anchor);m=self._vec(measured_anchor);e=self._vec(original_swing_endpoint)
        endpoint=a+np.array([0.,0.,-.0005])
        if np.linalg.norm(endpoint-e)>.012 or np.linalg.norm(endpoint-m)>.025:
            raise ValueError('RR correction would exceed existing12mm endpoint or25mm preload bound')
        self.started=True;self.status='applying';self.start_s=float(time_s);self.end_s=self.start_s+.30
        self.anchor_at_start=a;self.measured_anchor_at_start=m;self.original_swing_endpoint=e
        self.armed_touchdown_count=int(confirmed_before+1);self.last_sample_time_s=self.start_s
        return True

    def require_contact(self, contact, measured_point):
        if not self.active:return
        if not bool(contact):raise ValueError('RR preload correction lost measured distal support')
        p=self._vec(measured_point)
        displacement=float(np.linalg.norm(p-self.measured_anchor_at_start))
        self.maximum_observed_contact_displacement_m=max(self.maximum_observed_contact_displacement_m,displacement)
        if displacement>.012:raise ValueError('RR correction measured contact moved beyond existing12mm region')
        # Includes the full intended correction, so the un-emitted future offset
        # is never certified by a synthetic contact update.
        endpoint=self.anchor_at_start+np.array([0.,0.,-.0005])
        if np.linalg.norm(endpoint-p)>.025:raise ValueError('RR correction exceeds existing measured25mm preload bound')

    def sample(self,time_s):
        if not self.started:raise ValueError('Cannot sample an unarmed correction')
        if not math.isfinite(time_s):raise ValueError('Finite correction sample time required')
        u=np.clip((float(time_s)-self.start_s)/.30,0.,1.)
        s=10*u**3-15*u**4+6*u**5
        ds=(30*u**2-60*u**3+30*u**4)/.30
        dds=(60*u-180*u**2+120*u**3)/(.30**2)
        offset=np.array([0.,0.,-.0005*s]);v=np.array([0.,0.,-.0005*ds]);a=np.array([0.,0.,-.0005*dds])
        if time_s>=self.end_s-1e-10:offset[2]=-.0005;v[:]=0.;a[:]=0.
        return offset,v,a

    def apply(self,time_s,anchors,measured_anchors,preload_world):
        self.changed_this_step=False
        if not self.active:return
        if not math.isfinite(time_s) or time_s<self.last_sample_time_s-1e-10 or time_s>self.end_s+.020000001:
            raise ValueError('RR correction time must advance within its existing finite hold')
        offset,v,a=self.sample(time_s)
        self.changed_this_step=bool(np.any(offset!=self.offset_applied_m))
        anchors[5]=self.anchor_at_start+offset
        # Never rewrite actual contact anchors or create a measured contact.
        preload_world[5]=anchors[5]-measured_anchors[5]
        self.offset_applied_m=offset;self.velocity_mps=v;self.acceleration_mps2=a;self.last_sample_time_s=float(time_s)
        if time_s>=self.end_s-1e-10:
            self.completed=True;self.status='completed'

    def abort(self,reason):
        if self.active:self.failure=str(reason);self.status='rejected'

    def state(self):
        return dict(proposal=PROPOSAL,started=self.started,completed=self.completed,status=self.status,
                    start_s=self.start_s,end_s=self.end_s,anchor_at_start_world_m=self.anchor_at_start,
                    measured_anchor_at_start_world_m=self.measured_anchor_at_start,
                    original_swing_endpoint_world_m=self.original_swing_endpoint,
                    offset_applied_world_m=self.offset_applied_m.copy(),analytic_velocity_world_mps=self.velocity_mps.copy(),
                    analytic_acceleration_world_mps2=self.acceleration_mps2.copy(),changed_this_step=self.changed_this_step,
                    last_sample_time_s=self.last_sample_time_s,armed_touchdown_count=self.armed_touchdown_count,
                    maximum_observed_contact_displacement_m=self.maximum_observed_contact_displacement_m,failure=self.failure)
