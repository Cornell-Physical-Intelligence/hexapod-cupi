from pathlib import Path
import sys,json,numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'oracle'))
from pair_motion import PairContactReference,PairMotionConfig
from ideal_fixture import IdealMeasuredFixture
from reference_residual import ReferenceResidualTarget,ResidualConfig

def evaluate(speed=.01,stop_at=26.,duration=44.,mutate=None,names=None,preload=0.):
 f=IdealMeasuredFixture(names=names,preload=preload);g=PairContactReference(f.names,PairMotionConfig(max_translation_mps=speed));r=g.reset(f.snapshot());limits=f.snapshot()['soft_joint_pos_limits_rad'][0]
 core=ReferenceResidualTarget(f.names,dict(zip(f.names,limits[:,0])),dict(zip(f.names,limits[:,1])),1,ResidualConfig('formal_004',.02,.25,2.,8.));core.reset(r['q_ref'],r['q_ref'])
 out=[];maxv=0.;maxa=0.;lag=0.;failure=None;minmargin=1.;stop_lifts=None
 for k in range(round(duration/.02)):
  snapshot=f.snapshot()
  if mutate is not None:mutate(k,g,f,snapshot)
  cmd=[speed,0,0] if 2<=k*.02<stop_at else [0,0,0]
  if k*.02>=stop_at and stop_lifts is None:stop_lifts=g.liftoffs
  r=g.step(snapshot,cmd)
  if not r['valid'][0]:failure={'control':k,'time_s':k*.02,'reason':r['failure_reason']};break
  e=core.step(r['q_ref'],np.zeros((1,18)),reference_valid=r['valid'])
  lag=max(lag,float(abs(e['target_position_rad'].numpy()-r['q_ref']).max()));maxv=max(maxv,float(abs(r['v_ref']).max()));maxa=max(maxa,float(abs(r['a_ref']).max()));minmargin=min(minmargin,r['diagnostics']['measured_support_margin_m'])
  out.append(r);f.advance(r)
 return {'speed':speed,'controls':len(out),'failure':failure,'completed_pairs':g.completed_pairs,'touchdowns':g.touchdowns,'liftoffs':g.liftoffs,'lifts_after_stop':None if stop_lifts is None else g.liftoffs-stop_lifts,'vmax':maxv,'amax':maxa,'zero_residual_lag':lag,'minimum_measured_fixture_margin':minmargin,'quiet_time':g.reference_quiet_time,'stop_latency':None if g.reference_quiet_time is None else g.reference_quiet_time-stop_at,'final_mode':g.mode,'physics':False},out,g,f
if __name__=='__main__':
 r,_,_,_=evaluate();print(json.dumps(r,indent=2))
