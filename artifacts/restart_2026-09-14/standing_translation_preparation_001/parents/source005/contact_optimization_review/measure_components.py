from measure import *
session,geometry,poses,mapping=load();groups=collections.Counter();examples={}
for line in(ACTUAL/'contacts.jsonl').open():
 r=json.loads(line);key=(len(r['patches']),sum(v['inactive_zero_normal']for v in r['patches']),sum(v['normal_force_n']!=0 for v in r['patches']));groups[key]+=1;examples.setdefault(key,r)
save('strata.json',{'strata':[{'patches':k[0],'inactive':k[1],'nonzero':k[2],'rows':v}for k,v in sorted(groups.items())]});print('STRATA',len(groups),flush=True)
result=[]
for n in (1,32):
 weighted={key:0. for key in ('classification','clearance','json')};samples=[]
 for key,count in sorted(groups.items()):
  row=examples[key];inputs,maps=data(row,mapping,n);pp=np.repeat(poses[row['sequence']],n,axis=0);classified=old.classify_contacts(inputs,maps,pp,geometry,n)
  durations={k:[]for k in weighted}
  for _ in range(3):
   t=time.perf_counter();old.classify_contacts(inputs,maps,pp,geometry,n);durations['classification'].append(time.perf_counter()-t)
   t=time.perf_counter();geometry.clearance(pp);durations['clearance'].append(time.perf_counter()-t)
   t=time.perf_counter();json.dumps({'sequence':row['sequence'],'explicit_counter':row['explicit_counter'],'patches':classified['patches']},allow_nan=False)+'\n';durations['json'].append(time.perf_counter()-t)
  med={k:float(np.median(v))for k,v in durations.items()}
  for k in weighted:weighted[k]+=med[k]*count
  samples.append({'stratum':list(key),'actual_rows':count,'median_seconds':med})
 result.append({'replicas':n,'weighted_8000_substep_seconds':weighted,'sum_measured_components_seconds':sum(weighted.values()),'samples':samples})
 save('components.json',{'scope':'LocalCPU exact unchanged runtime components on replicated actual rows, not Spark measurement or total phase wall time. Excludes native CPU/GPU extraction/synchronization, row stacking/compression, disk writes and startup. Three medians per stratum.','strata':len(groups),'result':result});print('RESULT',n,weighted,flush=True)
