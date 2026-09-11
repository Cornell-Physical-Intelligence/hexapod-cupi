#!/usr/bin/env python3
"""Bounded CPU-only contact-record serialization experiment, not a native writer.
All fields, ordered patches, float64 bits and inactive records survive roundtrip.
Inputs are local archived one-env prefix and selected actual32 support-event rows.
"""
import argparse,copy,gzip,hashlib,json,math,platform,statistics,struct,time,zlib
from pathlib import Path

KEYS=['buffer_index','env','body','category','normal_force_n','point_world_m','normal_world','separation_m','inactive_zero_normal','shape_point_m']
RECORD=struct.Struct('<IIHHBB11d')
MAGIC=b'CPREC001'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def current(rows):
 return ''.join(json.dumps(r,allow_nan=False)+'\n'for r in rows).encode()

def encode(rows):
 bodies=list(dict.fromkeys(p['body']for r in rows for p in r['patches']))
 cats=list(dict.fromkeys(p['category']for r in rows for p in r['patches']))
 bodyid={s:i for i,s in enumerate(bodies)};catid={s:i for i,s in enumerate(cats)}
 records=[];rowmeta=[]
 for r in rows:
  if list(r)!=['sequence','explicit_counter','patches']:raise ValueError('Unknown row schema/order')
  rowmeta.append([r['sequence'],r['explicit_counter'],len(r['patches'])])
  for p in r['patches']:
   if list(p)!=KEYS:raise ValueError('Unknown patch schema/order')
   if type(p['inactive_zero_normal'])is not bool:raise ValueError('Boolean type changed')
   v=[p['normal_force_n'],*p['point_world_m'],*p['normal_world'],p['separation_m'],*(p['shape_point_m']if p['shape_point_m']is not None else [0.,0.,0.])]
   if len(v)!=11 or not all(type(x)is float and math.isfinite(x)for x in v):raise ValueError('Invalid numeric record')
   records.append(RECORD.pack(p['buffer_index'],p['env'],bodyid[p['body']],catid[p['category']],p['inactive_zero_normal'],p['shape_point_m']is not None,*v))
 header=json.dumps({'schema':'contact_record_benchmark_v1','bodies':bodies,'categories':cats,'rows':rowmeta},allow_nan=False,separators=(',',':')).encode()
 return MAGIC+struct.pack('<I',len(header))+header+b''.join(records)

def decode(b):
 if b[:8]!=MAGIC:raise ValueError('Unknown codec')
 n=struct.unpack('<I',b[8:12])[0];h=json.loads(b[12:12+n]);offset=12+n
 if h['schema']!='contact_record_benchmark_v1':raise ValueError('Unknown schema')
 if len(b)-offset!=sum(r[2]for r in h['rows'])*RECORD.size:raise ValueError('Truncated or extra records')
 rows=[]
 for seq,counter,count in h['rows']:
  patches=[]
  for _ in range(count):
   a=RECORD.unpack_from(b,offset);offset+=RECORD.size
   if a[4]not in (0,1)or a[5]not in (0,1):raise ValueError('Invalid flags')
   vals=[a[0],a[1],h['bodies'][a[2]],h['categories'][a[3]],a[6],list(a[7:10]),list(a[10:13]),a[13],bool(a[4]),list(a[14:17])if a[5]else None]
   patches.append(dict(zip(KEYS,vals)))
  rows.append({'sequence':seq,'explicit_counter':counter,'patches':patches})
 return rows

def test_codec():
 p=dict(zip(KEYS,[3,0,'lf_tibia','toe',-0.,[0.,-0.,1.],[-0.,0.,0.],-0.,True,[.13,-0.,0.]]))
 rows=[{'sequence':4,'explicit_counter':5,'patches':[p]}]
 assert current(decode(encode(rows)))==current(rows)
 q=copy.deepcopy(rows);q[0]['patches'][0]['shape_point_m']=None;q[0]['patches'][0]['category']='body'
 assert current(decode(encode(q)))==current(q)
 for modify in [lambda a:a[0]['patches'][0].update(extra=1),lambda a:a[0]['patches'][0].update(normal_force_n=float('nan'))]:
  bad=copy.deepcopy(rows);modify(bad)
  try:encode(bad)
  except ValueError:pass
  else:raise AssertionError('Malformed input accepted')
 try:decode(encode(rows)[:-1])
 except ValueError:pass
 else:raise AssertionError('Truncation accepted')
 return {'signed_zero_and_shape_present':True,'shape_absent_and_body_label':True,'unknown_field_rejected':True,'nonfinite_rejected':True,'truncated_block_rejected':True}

def measure(f,repeats=7):
 f();times=[]
 for _ in range(repeats):
  t=time.perf_counter();v=f();times.append(time.perf_counter()-t)
 return v,{'median_s':statistics.median(times),'min_s':min(times),'max_s':max(times),'repetitions':repeats}

def bench(rows):
 plain=current(rows);packed=encode(rows);restored=current(decode(packed))
 assert plain==restored
 variants={}
 for name,func in [('current_json',lambda:current(rows)),('current_json_zlib1',lambda:zlib.compress(current(rows),1)),('binary',lambda:encode(rows)),('binary_zlib1',lambda:zlib.compress(encode(rows),1))]:
  b,t=measure(func)
  if name=='binary_zlib1':assert current(decode(zlib.decompress(b)))==plain
  if name=='current_json_zlib1':assert zlib.decompress(b)==plain
  variants[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'encoding':t}
 for name,func in [('current_json',lambda:[json.loads(line)for line in plain.splitlines()]),('binary',lambda:decode(packed))]:
  _,t=measure(func);variants[name]['decoding']=t
 count=sum(len(r['patches'])for r in rows)
 return {'rows':len(rows),'patches':count,'inactive_zero_normal_records':sum(p['inactive_zero_normal']for r in rows for p in r['patches']),
         'byte_exact_reconstructed_current_JSON':True,'ordered_source_JSON_sha256':hashlib.sha256(plain).hexdigest(),'variants':variants,
         'binary_zlib1_size_ratio':variants['binary_zlib1']['bytes']/len(plain),'binary_zlib1_encode_speed_ratio':variants['current_json']['encoding']['median_s']/variants['binary_zlib1']['encoding']['median_s']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=a.repo.resolve()
 one=repo/'artifacts/mkii_updated_2026-09-10/native_standing_004/terminal/run/standing/contacts.jsonl.gz'
 selected=repo/'artifacts/mkii_updated_2026-09-10/standing32_failure_analysis_001/actual002/analysis.json'
 rows=[]
 with gzip.open(one,'rt')as f:
  for _ in range(128):rows.append(json.loads(next(f)))
 d=json.loads(selected.read_text());parts=[]
 for r in d['targeted_contact_samples']:
  patches=[p for o in r['observations']for p in o['raw_patches']]
  parts.append({'sequence':r['sequence'],'explicit_counter':r['explicit_counter'],'patches':patches})
 report={'schema':'canonical_contact_record_CPU_microbenchmark_v1','native_executed':False,'runtime_adopted':False,
  'machine':platform.platform(),'python':platform.python_version(),'processor':platform.processor(),
  'scope':'In-memory serialization of already-built dictionaries. Includes encoding validations/compression, excludes native/PCIe transfers/classification/dict creation/geometry/disk writes. Local machine, not Spark.',
  'inputs':{str(one.relative_to(repo)):{'file_sha256':sha(one),'selection':'First128 complete one-env frames, no full-file decompression retained.'},str(selected.relative_to(repo)):{'file_sha256':sha(selected),'selection':'All218 targeted support-event sequence excerpts; only selected env/body patches, NOT complete32-env frames.'}},
  'tests':test_codec(),'one_env_first128':bench(rows),'actual32_selected218':bench(parts)}
 a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:report[k]for k in ['machine','python','tests','one_env_first128','actual32_selected218']},indent=2))
if __name__=='__main__':main()
