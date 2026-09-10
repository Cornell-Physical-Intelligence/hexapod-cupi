"""CPU proposal only: numerical output-side servo, not calibrated RS05 behavior."""
import numpy as np

RPM=np.array([0.,70.,275.,340.,450.,477.,480.])
NM=np.array([5.5,5.5,4.,3.,1.6,.5,0.])

def effort(q,dq,target,kp,kd):
 values=[np.asarray(x,dtype=float)for x in (q,dq,target,kp,kd)]
 if any(x.shape!=(18,)or not np.isfinite(x).all()for x in values):
  raise ValueError('Expected finite 18-joint named-order arrays')
 q,dq,target,kp,kd=values
 if (kp<0).any()or(kd<0).any():raise ValueError('Negative servo gains')
 raw=kp*(target-q)-kd*dq
 rpm=abs(dq)*60/(2*np.pi)
 ceiling=np.minimum(1.6,np.interp(rpm,RPM,NM,left=NM[0],right=0.))
 applied=np.clip(raw,-ceiling,ceiling)
 return {'requested_nm':raw,'software_applied_nm':applied,'ceiling_nm':ceiling,
  'clipped':raw!=applied,'above_vendor_stall_1p2':abs(applied)>1.2,
  'mechanical_power_w':applied*dq}
