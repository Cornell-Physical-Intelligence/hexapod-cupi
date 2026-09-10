"""Read-only CPU review of a frozen controller; no physical qualification."""
from pathlib import Path
import ast, hashlib, json, sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'reference_tensor_wave_004_001'
sys.path.insert(0,str(SOURCE))
from test_batch_wave import Fixture, pack
from batch_wave import BatchWave004
from wave_reference import WaveContactReference

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((SOURCE/'FREEZE_SHA256.json').read_text())
mismatches=[name for name,digest in manifest.items() if sha(SOURCE/name)!=digest]
assert not mismatches,mismatches
rng=np.random.default_rng(204001)
n=24
fixtures=[Fixture() for _ in range(n)]
refs=[]
for i,f in enumerate(fixtures):
 f.p=(np.array([.11,-.08,.13])+np.r_[rng.uniform(-.1,.1,2),0.]).astype(np.float32 if i%2 else np.float64)
 r=WaveContactReference(f.names);r.reset(f.snapshot());refs.append(r)
batch=BatchWave004(fixtures[0].names,n)
batch.reset(pack([f.snapshot() for f in fixtures]),torch.ones(n,dtype=torch.bool),torch.zeros(n,dtype=torch.int64))
for i,r in enumerate(refs):
 r.command=rng.uniform(-.002,.002,3);r.command_rate=rng.uniform(-.001,.001,3);r.yaw=float(rng.uniform(-1,1))
 for k,v in [('position',r.position),('command',r.command),('rate',r.command_rate),('yaw',r.yaw)]:batch.s[k][i]=torch.as_tensor(v,dtype=torch.float64)
target=rng.uniform(-.004,.004,(n,3))
p,R=batch._predict(torch.as_tensor(target,dtype=torch.float64))
errors=[]
for i,r in enumerate(refs):
 expected=r._predict(target[i],7.9)
 errors.append({'row':i,'position_float32':bool(batch.s['position_float32'][i]),'max_position_error_m':float(np.max(np.abs(p[i].numpy()-expected[0]))),'max_rotation_error':float(np.max(np.abs(R[i].numpy()-expected[1])))})
 assert errors[-1]['max_position_error_m']<2e-12,errors[-1]
 assert errors[-1]['max_rotation_error']<2e-12,errors[-1]
# Quantify why a single final cast is not equivalent to the inherited recurrence.
f32=batch.s['position_float32'].clone()
batch.s['position_float32'].fill_(False)
unrounded,_=batch._predict(torch.as_tensor(target,dtype=torch.float64))
collapsed=unrounded.float().double()
changed=(collapsed-p).abs().amax(-1)[f32]
assert (changed>0).any()
# Both reviewed production modules; constructor source hashing and offline tests are separate.
audit=[]
for relative in ['batch_wave.py','kernel/tensor_kernel.py']:
 tree=ast.parse((SOURCE/relative).read_text())
 forbidden=[];casts=[];loops=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
   if node.func.attr in ('cpu','numpy','item','tolist'):forbidden.append({'line':node.lineno,'call':node.func.attr})
   if node.func.attr in ('to','float','double'):casts.append({'line':node.lineno,'call':ast.unparse(node)})
  if isinstance(node,ast.For):loops.append({'line':node.lineno,'iterator':ast.unparse(node.iter)})
 assert not forbidden,(relative,forbidden)
 audit.append({'file':relative,'forbidden_host_conversion_calls':forbidden,'dtype_casts':casts,'loops':loops})
report={'scope':'Independent read-only CPU review; no CUDA/Isaac execution or physical admission','source_manifest_sha256':sha(SOURCE/'FREEZE_SHA256.json'),'manifest_file_count':len(manifest),'manifest_mismatches':mismatches,'mixed_precision_predictor_cases':errors,'max_position_error_m':max(r['max_position_error_m'] for r in errors),'max_rotation_error':max(r['max_rotation_error'] for r in errors),'float32_rows_changed_by_final_cast_only':int((changed>0).sum()),'max_final_cast_only_difference_m':float(changed.max()),'static_dynamic_module_audit':audit}
(HERE/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['mixed_precision_predictor_cases','static_dynamic_module_audit']},indent=2))
