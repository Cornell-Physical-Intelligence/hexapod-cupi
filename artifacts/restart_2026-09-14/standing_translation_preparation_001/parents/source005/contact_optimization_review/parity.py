from measure import *
spec=importlib.util.spec_from_file_location('candidate',HERE/'standing_math_candidate.py');new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)

def equal(a,b):
 assert set(a)==set(b)
 for k in a:
  if isinstance(a[k],np.ndarray):
   assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(),k
  elif k=='patches':assert json.dumps(a[k],allow_nan=False)==json.dumps(b[k],allow_nan=False),'patch list/order/value differs'
  else:assert a[k]==b[k],k

def main():
 session,geometry,poses,mapping=load();count=0;old_time=0.;new_time=0.;examples={};raw_patch_mismatches=0;max_shape_error=0.
 with(ACTUAL/'contacts.jsonl').open()as stream:
  for line in stream:
   row=json.loads(line);count+=1;key=(len(row['patches']),sum(x['inactive_zero_normal']for x in row['patches']),sum(x['normal_force_n']!=0 for x in row['patches']));examples.setdefault(key,row)
   inputs,maps=data(row,mapping);pp=poses[row['sequence']]
   t=time.perf_counter();a=old.classify_contacts(inputs,maps,pp,geometry,1);old_time+=time.perf_counter()-t
   t=time.perf_counter();b=new.classify_contacts(inputs,maps,pp,geometry,1);new_time+=time.perf_counter()-t;equal(a,b)
   if json.dumps(a['patches'])!=json.dumps(row['patches']):
    raw_patch_mismatches+=1
    assert len(a['patches'])==len(row['patches'])
    for x,y in zip(a['patches'],row['patches']):
     for k in x:
      if k!='shape_point_m':assert x[k]==y[k],k
     if x['shape_point_m']is not None:max_shape_error=max(max_shape_error,float(np.max(abs(np.asarray(x['shape_point_m'])-y['shape_point_m']))))
   if count%2000==0:print('ACTUAL_PARITY_ROWS',count,flush=True)
 assert count==8000
 result={'actual_rows_bitexact_candidate_vs_parent':count,'all_returned_arrays_dtype_shape_bytes_and_patch_json_order_exact':True,
  'source_bound_recorded_patch_replay_different_rows':raw_patch_mismatches,'source_bound_recorded_shape_max_abs_error_m':max_shape_error,
  'classification_seconds_full8000_one_env':{'parent':old_time,'candidate':new_time},'replicated_cases':[]}
 save('parity.json',result)
 for n in (2,8,32):
  times={'parent':0.,'candidate':0.}
  for key,row in sorted(examples.items()):
   inputs,maps=data(row,mapping,n);pp=np.repeat(poses[row['sequence']],n,axis=0)
   for e in range(n):
    delta=np.asarray([(e%8)*2.,(e//8)*2.,0.],np.float32);pp[e,:,:3]+=delta;inputs[1][e*1024:(e+1)*1024]+=delta
   t=time.perf_counter();a=old.classify_contacts(inputs,maps,pp,geometry,n);times['parent']+=time.perf_counter()-t
   t=time.perf_counter();b=new.classify_contacts(inputs,maps,pp,geometry,n);times['candidate']+=time.perf_counter()-t;equal(a,b)
  result['replicated_cases'].append({'replicas':n,'strata':len(examples),'grid_translation':True,'bitexact':True,'classification_seconds':times});save('parity.json',result)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
