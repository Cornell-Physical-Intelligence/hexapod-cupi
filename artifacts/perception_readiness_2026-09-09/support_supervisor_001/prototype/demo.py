from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_supervisor import fixture,geometry,envelope,check
m=fixture();xy=m.centers();pit=np.linalg.norm(xy-[.177,.177],axis=-1)<.045;m.height_m[pit]=-.12
geo=geometry(m);start=envelope(m,poses=[[0,0,0]],times=[0]);end=envelope(m,poses=[[0,0,np.pi/2]],times=[0]);swept=envelope(m,poses=[[0,0,0],[0,0,np.pi/2]])
flat=fixture();future=envelope(flat,times=[0,.5])
report=dict(scope='Synthetic CPU support/data-validity prototype; no executable braking model, actor, physics or sensorqualification',cases=dict(rotation_start=check(m,geo,start),rotation_end=check(m,geo,end),rotation_swept=check(m,geo,swept),flat_short_envelope=check(flat),flat_long_envelope_evidence_expires=check(flat,e=future)),coverage=dict(source='single synthetic snapshot',instantaneous_visibility_measured=False,accumulated_map_qualification=False),limits=['Given body-fixed pad path is a conservative synthetic support raster, not a physically admitted gait/stop prediction.','All point-contact support heights differ from actual footmesh/body/shaft collision clearance.','Permit means supplied support envelope passes this map check; force balance, torque and dynamics remain unproved.','No future reobservation is assumed. The0.25s evidence lease will reject longer stopping envelopes unless the evidence/uncertainty contract is explicitly revised and qualified.'])
p=Path(__file__).parent;(p/'report.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
for ax,required,title in [(axes[0],start.required|end.required,'Start/end only: both pass'),(axes[1],swept.required,'Swept rotation: pit detected')]:
    field=np.zeros(m.height_m.shape);field[required]=1;field[pit]=2;field[required&pit]=3
    ax.imshow(field.T,origin='lower',extent=[-.6,.6,-.6,.6],vmin=0,vmax=3,cmap='viridis',interpolation='nearest')
    ax.set(xlim=(-.05,.35),ylim=(-.05,.35),xlabel='Odom X (m)',ylabel='Odom Y (m)',title=title)
fig.suptitle('Observed ground can still be ineligible support — synthetic geometry only')
fig.savefig(p/'swept_support.png',dpi=160)
print(json.dumps({k:v['permit_candidate'] for k,v in report['cases'].items()}))
