from parity import *
session,geometry,poses,mapping=load();groups=collections.Counter();examples={}
for line in(ACTUAL/'contacts.jsonl').open():
 r=json.loads(line);key=(len(r['patches']),sum(v['inactive_zero_normal']for v in r['patches']),sum(v['normal_force_n']!=0 for v in r['patches']));groups[key]+=1;examples.setdefault(key,r)
result=[]
for n in (1,32):
 weighted=0.;samples=[]
 for key,count in sorted(groups.items()):
  row=examples[key];inputs,maps=data(row,mapping,n);pp=np.repeat(poses[row['sequence']],n,axis=0)
  new.classify_contacts(inputs,maps,pp,geometry,n);durations=[]
  for _ in range(3):
   t=time.perf_counter();new.classify_contacts(inputs,maps,pp,geometry,n);durations.append(time.perf_counter()-t)
  median=float(np.median(durations));weighted+=median*count;samples.append({'stratum':list(key),'actual_rows':count,'median_seconds':median,'trials_seconds':durations})
 result.append({'replicas':n,'weighted_8000_substep_classification_seconds':weighted,'samples':samples})
 save('candidate_timings.json',{'scope':'LocalCPU classification only, same31 actual strata and three medians; no Spark/native measurement.','result':result});print('CANDIDATE',n,weighted,flush=True)
