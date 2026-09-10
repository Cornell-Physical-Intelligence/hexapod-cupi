"""Five explicit prototype parameter choices; kinematic feasibility only."""
import json
from pathlib import Path
from reference import *
torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent
g=SerialGeometry();out=[]
cases=json.loads((OUT/'report.json').read_text())['command_screen']['rows']
phase=torch.arange(256,dtype=DTYPE)/256
for travel,lift,fmin in ((.1,.02,.8),(.06,.02,.8),(.06,.01,.8),(.04,.01,.8),(.06,.01,1.2)):
    r=TwistReference(g,Config(stance_travel_m=travel,lift_m=lift,min_frequency_hz=fmin))
    fail=[];speedmax=0.;fmax=0.;margin=1.;error=0.
    for row in cases:
        c=tensor([row['command']]).expand(len(phase),3)
        feet=r.feet(phase,c);ik=g.ik(feet);freq=r.frequency(c)
        if not ik['valid'].all():fail.append(row['name'])
        h=1e-5;v=(r.feet(phase+h*freq,c)-r.feet(phase-h*freq,c))/(2*h)
        _,J,_=g.fk(ik['q_checked']);qv=torch.linalg.solve(J,v[...,None]).squeeze(-1)
        speedmax=max(speedmax,float(qv.abs().max()));fmax=max(fmax,float(freq.max()))
        margin=min(margin,float(ik['minimum_joint_margin_rad'].min()));error=max(error,float(ik['error_m'].max()))
    out.append(dict(configuration=r.config.__dict__,tested_cases=len(cases),passed_cases=len(cases)-len(fail),
                    failed_cases=fail,max_frequency_hz=fmax,max_joint_speed_rad_s=speedmax,
                    minimum_soft_limit_margin_rad=margin,max_ik_error_m=error))
(OUT/'parameter_screen.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([{k:v for k,v in row.items() if k!='failed_cases'} for row in out],indent=2))
