from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,os,platform,sys,time
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if(p/'robot/active_model.json').is_file())
SOURCE=Path(os.environ.get('HEXAPOD_STANDING_SOURCE',str(ROOT/'tmp/updated_native_standing_002'))).resolve()
ACTUAL=Path(os.environ.get('HEXAPOD_STANDING_RAW',str(ROOT/'tmp/canonical_native_standing_terminal_002/run/standing'))).resolve()
sys.path.insert(0,str(SOURCE));import standing_math as old
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):(HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def load():
    session=json.loads((ACTUAL/'session.json').read_text());names=session['body_names']
    geom=json.loads((SOURCE/'geometry/geometry.json').read_text())
    with np.load(SOURCE/'geometry/geometry_extrema.npz')as z:g=old.Geometry(geom,{k:z[k]for k in z.files},names)
    poses=np.concatenate([np.load(ACTUAL/f)['link_pose_xyzw']for f in session['substep_files']])
    mapping=[(0,s.rsplit('/',1)[1])for s in json.loads((ACTUAL/'contact_view.json').read_text())['sensor_paths']]
    return session,g,poses,mapping

def data(row,mapping,n=1):
    capacity=1024*n;f=np.zeros((capacity,1),np.float32);p=np.zeros((capacity,3),np.float32);normal=p.copy();sep=f.copy();counts=np.zeros((19*n,1),np.uint32);starts=counts.copy();maps=[]
    for e in range(n):
        for i,(_,body)in enumerate(mapping):
            maps.append((e,body));patch=[v for v in row['patches']if v['body']==body]
            if patch:
                indices=[v['buffer_index']for v in patch];assert indices==list(range(min(indices),max(indices)+1))
                counts[e*19+i]=len(patch);starts[e*19+i]=min(indices)+1024*e
            for q in patch:
                k=q['buffer_index']+1024*e;f[k,0]=q['normal_force_n'];p[k]=q['point_world_m'];normal[k]=q['normal_world'];sep[k,0]=q['separation_m']
    return [f,p,normal,sep,counts,starts],maps

def main():
    session,geometry,poses,mapping=load();hist=collections.Counter();examples={};allrows=0;inactive=0;categories=collections.Counter();nonzero=0
    start=time.perf_counter()
    with (ACTUAL/'contacts.jsonl').open()as stream:
        for line in stream:
            row=json.loads(line);sequence=row['sequence'];assert sequence==allrows;allrows+=1;k=len(row['patches']);hist[k]+=1
            examples.setdefault(k,row)
            inactive+=sum(v['inactive_zero_normal']for v in row['patches']);nonzero+=sum(v['normal_force_n']!=0 for v in row['patches']);categories.update(v['category']for v in row['patches'])
    assert allrows==8000
    distribution={'rows':allrows,'patch_histogram':dict(sorted(hist.items())),'mean_patches':sum(k*v for k,v in hist.items())/allrows,'inactive_zero_normal_records':inactive,'nonzero_force_records':nonzero,'categories':dict(categories),'actual_one_env_session_wall_s':session['wall_s'],'read_parse_wall_s':time.perf_counter()-start,'contacts_sha256':sha(ACTUAL/'contacts.jsonl'),'source_freeze_sha256':sha(SOURCE/'FREEZE_SHA256.json'),'machine':platform.platform(),'python':platform.python_version(),'numpy':np.__version__}
    save('distribution.json',distribution);print(json.dumps(distribution),flush=True)
    # Every distinct actual patch-count bucket is weighted by its observed frequency.
    timings={}
    for n in (1,32):
        weighted=0.;samples=[]
        for k,row in sorted(examples.items()):
            inputs,maps=data(row,mapping,n);pp=np.repeat(poses[row['sequence']],n,axis=0)
            for _ in range(1):old.classify_contacts(inputs,maps,pp,geometry,n)
            durations=[]
            for _ in range(3):
                t=time.perf_counter();old.classify_contacts(inputs,maps,pp,geometry,n);durations.append(time.perf_counter()-t)
            median=float(np.median(durations));weighted+=hist[k]*median;samples.append({'patches_per_env':k,'actual_rows':hist[k],'median_s':median,'trials_s':durations})
        timings[str(n)]={'weighted_8000_substeps_s':weighted,'weighted_mean_s':weighted/8000,'samples':samples}
        save('timing_old.json',{'distribution':distribution,'timings':timings,'scope':'Mac CPU exact unchanged classifier only, replicated actual contact tuples/poses. Does not measure Spark CPU, GPU, JSON emission, native extraction, mesh clearance or compression; no 32-robot physical outcome.'});print('N',n,'CONTACT_ONLY_WEIGHTED_SECONDS',weighted,flush=True)
if __name__=='__main__':main()
