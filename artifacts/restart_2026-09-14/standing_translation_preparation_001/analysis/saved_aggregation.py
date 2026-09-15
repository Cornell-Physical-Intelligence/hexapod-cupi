"""Exact selected saved-analyzer functions; independent copied replay oracle."""
import math,struct
LEGS=['lf','lm','lr','rf','rm','rr']
def require(ok,message):
 if not ok:raise ValueError(message)

def norm(x):return math.sqrt(sum(v*v for v in x))

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]

def products(p):
 f=p['normal_force_n'];v=p['normal_world']
 require(math.isfinite(f)and len(v)==3 and all(math.isfinite(x)for x in v),'Malformed force/normal')
 require(f32(f)==f and all(f32(x)==x for x in v),'Original contact inputs are not exact FP32')
 return [f32(f*x)for x in v]

def aggregate(patches,n):
 toe=[[[0.,0.,0.]for _ in LEGS]for _ in range(n)];body={};counts={};seen=set()
 for p in patches:
  e=p['env'];b=p['body'];k=p['buffer_index']
  require(type(e)is int and 0<=e<n and type(k)is int and k>=0 and k not in seen,'Patch mapping/duplicate buffer index')
  seen.add(k);v=products(p);key=(e,b);body.setdefault(key,[0.,0.,0.]);counts[key]=counts.get(key,0)+1
  for j in range(3):body[key][j]+=v[j]
  if p['category']=='toe':
   require(b in [l+'_tibia'for l in LEGS],'Toe on wrong body');l=LEGS.index(b[:2])
   for j in range(3):toe[e][l][j]+=v[j]
 return toe,body,counts
