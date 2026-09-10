"""Actual singleenv patch arithmetic: sourcefloat32 product thenfloat64 aggregation."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
from analyze_remote import aggregate_patches
root=Path(__file__).resolve().parents[2];single=root/'tmp/canonical_native_standing_terminal_004/run/standing';raw=root/'tmp/canonical_native_standing_terminal_002/run/standing/contacts.jsonl'
# Exact raw1 contacts equal current native4 according to its audited originalSHA.
require_sha='95ed8eb951dcf5aa56b79e4f43f53b3588d858b3a8af71e613aec79d9b09b216'
h=hashlib.sha256()
with raw.open('rb')as f:
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
assert h.hexdigest()==require_sha
session=json.loads((single/'session.json').read_text());legs=['lf','lm','lr','rf','rm','rr'];first=None;count=0;all_equal=True;chunk=-1
with raw.open()as f:
 for line in f:
  r=json.loads(line);seq=r['sequence']
  if seq//800!=chunk:
   chunk=seq//800
   with np.load(single/f'substeps_{chunk:03d}.npz')as z:force=z['distal_force_world_n']
  for leg in legs:
   ps=[p for p in r['patches']if p['body']==leg+'_tibia'];old,new,products=aggregate_patches(ps);numpy=np.zeros(3)
   for p in ps:
    if p['category']=='toe':
     product=float(p['normal_force_n'])*np.asarray(p['normal_world'],np.float32);assert product.dtype==np.float32;numpy+=product
   expected=force[seq%800,0,legs.index(leg)].tolist();assert new==numpy.tolist()==expected
   if old!=expected and first is None:first={'sequence':seq,'explicit_counter':r['explicit_counter'],'env':0,'body':leg+'_tibia','old_unrounded':old,'source_float32_product_then_float64_accumulator':new,'stored_NPZ':expected,'delta_n':[x-y for x,y in zip(old,expected)],'differing_products':products}
   count+=1
buffer=root/'tmp/canonical_native_standing_terminal_001/run/standing/failed_contact_buffer.npz'
with np.load(buffer)as z:dtypes={k:str(z[k].dtype)for k in z.files}
print(json.dumps({'passed':True,'numpy_version':np.__version__,'raw_dtype_evidence':{'file':str(buffer),'sha256':hashlib.sha256(buffer.read_bytes()).hexdigest(),'dtypes':dtypes},'contact_sha256':h.hexdigest(),'actual_sequences':8000,'actual_named_foot_aggregates':count,'all_exact_source_and_NPZ':True,'first_old_algorithm_mismatch':first},indent=2,allow_nan=False))
