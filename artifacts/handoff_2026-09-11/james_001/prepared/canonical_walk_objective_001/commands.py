"""Proposed command-only curriculum; no physics, reset, actor or automatic promotion."""
from dataclasses import dataclass
import numpy as np

SCHEMA='canonical_omni_command_curriculum_v1'
DT=.02
LEVELS={0:((.02,.04),(.08,.16)),1:((.04,.08),(.16,.32)),2:((.08,.16),(.32,.50))}


@dataclass(frozen=True)
class Segment:
 controls:int
 command:tuple


def program(env_id,episode,level,seed=20260910):
 """25 equal-duration strata:1 quiet,24 motion; exactly20% total zero time."""
 if any(type(v)is not int or v<0 for v in [env_id,episode,seed])or level not in LEVELS:raise ValueError('Invalid explicit curriculum identity')
 slot=(env_id+7*episode)%25
 rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(env_id,episode,level)))
 linear,yaw=LEVELS[level];v=float(rng.uniform(*linear));w=float(rng.uniform(*yaw))
 if slot==0:return {'schema':SCHEMA,'kind':'quiet','slot':slot,'segments':[Segment(600,(0.,0.,0.))],'total_controls':600}
 if slot<=8:kind='axis';angle=((slot-1+episode)%4)*np.pi/2;command=np.array([v*np.cos(angle),v*np.sin(angle),0.])
 elif slot<=14:kind='diagonal';angle=np.pi/4+((slot-9+episode)%4)*np.pi/2;command=np.array([v*np.cos(angle),v*np.sin(angle),0.])
 elif slot<=18:kind='pure_yaw';command=np.array([0.,0.,w*(1 if (slot+episode)%2 else -1)])
 else:
  kind='arc';angle=((slot-19+episode)%8)*np.pi/4;command=np.array([v*np.cos(angle),v*np.sin(angle),w*(1 if rng.integers(2)else -1)])
 command[abs(command)<1e-14]=0.;a=tuple(command);b=tuple(-command);zero=(0.,0.,0.)
 if episode%2:segments=[Segment(250,a),Segment(250,b),Segment(100,zero)] # direct reversal, then stop
 else:segments=[Segment(250,a),Segment(50,zero),Segment(250,b),Segment(50,zero)]
 return {'schema':SCHEMA,'kind':kind,'slot':slot,'segments':segments,'total_controls':600}


def command_at(plan,control):
 if type(control)is not int or not 0<=control<plan['total_controls']:raise ValueError('Command clock outside declared program')
 for segment in plan['segments']:
  if control<segment.controls:return np.asarray(segment.command,dtype=np.float64).copy()
  control-=segment.controls
 raise ValueError('Incomplete command program')


class CommandBank:
 """Independent command-program epochs/clocks; never resets physical or policy history."""
 def __init__(self,n,level,seed=20260910):
  if type(n)is not int or n<=0 or level not in LEVELS:raise ValueError('Invalid explicit allocation')
  self.n=n;self.level=level;self.seed=seed;self.epochs=np.zeros(n,np.int64);self.clocks=np.zeros(n,np.int64)
  self.plans=[program(i,0,level,seed)for i in range(n)]
 def observe(self):return np.stack([command_at(p,int(t))for p,t in zip(self.plans,self.clocks)])
 def advance(self):
  if np.any(self.clocks>=600):raise ValueError('Program boundary requires explicit caller restart; no physical reset implied')
  self.clocks+=1
  return (self.clocks==600).copy()
 def reset_rows(self,rows):
  rows=np.asarray(rows)
  if rows.ndim!=1 or not np.issubdtype(rows.dtype,np.integer)or len(set(rows.tolist()))!=len(rows)or np.any(rows<0)or np.any(rows>=self.n):raise ValueError('Invalid selected reset rows')
  for i in rows:
   self.epochs[i]+=1;self.clocks[i]=0;self.plans[i]=program(int(i),int(self.epochs[i]),self.level,self.seed)
 def state(self):return {'schema':SCHEMA,'level':self.level,'seed':self.seed,'epochs':self.epochs.copy(),'clocks':self.clocks.copy()}
 @classmethod
 def from_state(cls,state):
  epochs=np.asarray(state['epochs']);clocks=np.asarray(state['clocks'])
  if state.get('schema')!=SCHEMA or epochs.ndim!=1 or clocks.shape!=epochs.shape or not np.issubdtype(epochs.dtype,np.integer)or not np.issubdtype(clocks.dtype,np.integer)or np.any(epochs<0)or np.any(clocks<0)or np.any(clocks>600):raise ValueError('Invalid command-program checkpoint')
  bank=cls(len(epochs),state['level'],state['seed']);bank.epochs=epochs.copy();bank.clocks=clocks.copy()
  bank.plans=[program(i,int(e),bank.level,bank.seed)for i,e in enumerate(epochs)]
  return bank


def held_out_cases(level):
 """Fixed new-model diagnostics, not substituted historical admission labels."""
 if level not in LEVELS:raise ValueError('Unknown level')
 lin,yaw=LEVELS[level];v=sum(lin)/2;w=sum(yaw)/2;rows=[]
 for i in range(8):
  angle=i*np.pi/4;rows.append((f'translation_{i}',(v*np.cos(angle),v*np.sin(angle),0.)))
 rows.extend([('yaw_positive',(0.,0.,w)),('yaw_negative',(0.,0.,-w))])
 for i in range(8):
  angle=np.pi/8+i*np.pi/4
  for sign in [-1,1]:rows.append((f'arc_{i}_yaw_{sign:+d}',(v*np.cos(angle),v*np.sin(angle),w*sign)))
 rows.extend((f'quiet_{i}',(0.,0.,0.))for i in range(6))
 result=[]
 for name,c in rows:
  c=np.asarray(c);c[abs(c)<1e-14]=0.
  result.append({'name':name,'segments':[Segment(100,(0.,0.,0.)),Segment(400,tuple(c)),Segment(600,(0.,0.,0.))],
          'total_controls':1100,'motion_controls':400 if c.any()else 0,'stop_controls':600,'quiet_last_controls':500,
          'training_distribution_admission':False,'requires_complete_reset_free_trace':True})
 return result
