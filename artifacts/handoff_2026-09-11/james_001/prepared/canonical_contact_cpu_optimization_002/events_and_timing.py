from parity import *

def main():
 session,geometry,poses,mapping=load();source=ROOT/'tmp/canonical_standing32_failure_analysis_root_002/analysis.json'
 assert sha(source)=='194e926651c3ac8f3c0aff6ebaef44cdb4ca39053b75a9898b51fe8259055870'
 raw=json.loads(source.read_text());cases=0;patches=0
 for row in raw['targeted_contact_samples']:
  n=32;capacity=32768;f=np.zeros((capacity,1),np.float32);p=np.zeros((capacity,3),np.float32);normal=p.copy();sep=f.copy();counts=np.zeros((19*n,1),np.uint32);starts=counts.copy();maps=[(e,body)for e in range(n)for _,body in mapping];pp=np.repeat(poses[0],n,axis=0)
  for obs in row['observations']:
   e=obs['env'];body=obs['body'];i=maps.index((e,body));actual=obs['raw_patches'];indices=[item['buffer_index']for item in actual]
   assert indices==list(range(min(indices),max(indices)+1))
   counts[i]=len(indices);starts[i]=min(indices);pp[e,geometry.names.index(body)]=np.asarray(obs['pose_xyzw'],np.float32)
   for item in actual:
    k=item['buffer_index'];f[k,0]=item['normal_force_n'];p[k]=item['point_world_m'];normal[k]=item['normal_world'];sep[k,0]=item['separation_m']
  data=[f,p,normal,sep,counts,starts]
  equal(old.classify_contacts(data,maps,pp,geometry,n),new.classify_contacts(data,maps,pp,geometry,n));cases+=1;patches+=sum(len(x['raw_patches'])for x in row['observations'])
 save('event_parity.json',{'exported_event_rows':cases,'selected_body_patch_records':patches,'old_candidate_arrays_bytes_and_patch_JSON_exact':True,'scope':'Actual exported adjacent-event body subsets at original env/buffer indices and float32 poses. Other sensors are explicitly empty fixtures; not full32-frame reproduction.','analyzer002_sha256':sha(source)})
 print('EVENT_PARITY',cases,patches,flush=True)
 groups=collections.Counter();examples={}
 with(ACTUAL/'contacts.jsonl').open()as stream:
  for line in stream:
   r=json.loads(line);key=(len(r['patches']),sum(v['inactive_zero_normal']for v in r['patches']),sum(v['normal_force_n']!=0 for v in r['patches']));groups[key]+=1;examples.setdefault(key,r)
 results=[]
 for n in [1,32]:
  weighted={'parent':0.,'candidate':0.};samples=[]
  for key,count in sorted(groups.items()):
   r=examples[key];inputs,maps=data_from_row(r,mapping,n);pp=np.repeat(poses[r['sequence']],n,axis=0)
   # Fresh unchanged translated pose/point pairs, float32 native precision.
   for e in range(n):
    delta=np.asarray([(e%8)*2.,(e//8)*2.,0.],np.float32);pp[e,:,:3]+=delta;inputs[1][e*1024:(e+1)*1024]+=delta
   trial={'parent':[],'candidate':[]}
   for label,fn in [('parent',old.classify_contacts),('candidate',new.classify_contacts)]:fn(inputs,maps,pp,geometry,n)
   for repeat in range(5):
    order=[('parent',old.classify_contacts),('candidate',new.classify_contacts)]
    if repeat%2:order.reverse()
    for label,fn in order:
     start=time.perf_counter();fn(inputs,maps,pp,geometry,n);trial[label].append(time.perf_counter()-start)
   for label in weighted:weighted[label]+=float(np.median(trial[label]))*count
   samples.append({'stratum':list(key),'actual_row_count':count,'seconds':trial})
  results.append({'replicas':n,'weighted_8000_substeps_seconds':weighted,'component_speedup':weighted['parent']/weighted['candidate'],'samples':samples})
  save('component_timing.json',{'scope':'LocalCPU classifier only, observed one-env stratum weights and synthetictranslatedreplication. Five interleaved timing pairs; no Spark/native/recording/GPU estimate.','machine':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'results':results});print('WEIGHTED',n,weighted,flush=True)

# Avoid colliding with the local six-array variable above.
data_from_row=data
if __name__=='__main__':main()
