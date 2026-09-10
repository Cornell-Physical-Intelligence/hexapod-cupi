"""CPU integrated-step timing on source-bound actual009 inputs, never GPU throughput."""
from pathlib import Path
import difflib,hashlib,json,platform,statistics,time
import numpy as np
import torch
from test_lazy_prediction import Original,Lazy,PARENT,pack,count_predict,HERE,OLD,NEW

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):
    for k in a.s:torch.testing.assert_close(a.s[k],b.s[k],atol=0,rtol=0,equal_nan=True,msg=k)

def main():
    torch.set_num_threads(1)
    with np.load(PARENT/'inputs/actual009_trace.npz') as z:arrays={k:z[k] for k in z.files}
    names=tuple(arrays['joint_names']);rows=[{k:v[i] for k,v in arrays.items() if v.shape[:1]==(2400,)} for i in range(2400)]
    def command(i):return [.005,0.,0.] if i<1399 else [0.,0.,0.]
    offsets=[int(x) for x in np.linspace(199,2119,32,dtype=int)];quiet_offset=2250;wanted=set(offsets+[quiet_offset]);states={}
    oracle=Original(names,1);oracle.reset(pack([rows[199]]),torch.tensor([True]),torch.tensor([0]))
    for i in range(199,quiet_offset+1):
        if i in wanted:states[i]={k:v.clone() for k,v in oracle.s.items()}
        oracle.step(pack([rows[i]]),torch.tensor([command(i)],dtype=torch.float64))
        assert oracle.active.all()
    results=[];controls=80
    for n in (1,32,128):
        for label,starts in [('actual_quiet',[quiet_offset]*n),('actual_asynchronous',[offsets[i%len(offsets)] for i in range(n)])]:
            initial={k:torch.cat([states[i][k] for i in starts]) for k in oracle.s}
            inputs=[(pack([rows[i+k] for i in starts]),torch.tensor([command(i+k) for i in starts],dtype=torch.float64)) for k in range(controls)]
            samples={'parent':[],'lazy':[]};counts={};modes=set()
            for repeat in range(4):
                objs={'parent':Original(names,n),'lazy':Lazy(names,n)}
                for obj in objs.values():
                    for k,v in initial.items():obj.s[k].copy_(v)
                    count_predict(obj)
                order=('parent','lazy') if repeat%2==0 else ('lazy','parent')
                for name in order:
                    obj=objs[name];elapsed=0.
                    for m,c in inputs:
                        start=time.perf_counter_ns();obj.step(m,c);elapsed+=(time.perf_counter_ns()-start)*1e-9
                        assert obj.active.all(),(n,label,name,obj.s['failure'].tolist())
                        if name=='lazy':modes.update(obj.s['mode'].tolist())
                    if repeat:samples[name].append(elapsed*1000/controls)
                    counts[name]=obj.predict_calls
                equal(objs['parent'],objs['lazy'])
            med={k:statistics.median(v) for k,v in samples.items()}
            row=dict(replicas=n,workload=label,controls_per_trial=controls,recorded_start_indices=starts,
                measured_trials=3,untimed_warmup_trials=1,full_state_endpoint_bit_exact=True,all_rows_valid=True,
                prediction_calls=counts,mode_ids=sorted(modes),milliseconds_per_control_samples=samples,
                median_milliseconds_per_control=med,local_CPU_speedup=med['parent']/med['lazy'])
            results.append(row);print(json.dumps({k:row[k] for k in ['replicas','workload','prediction_calls','median_milliseconds_per_control','local_CPU_speedup']}),flush=True)
    report=dict(scope='Local CPU complete BatchWave005.step timing with frozen real-recorded sensor states; not a physical rollout',
        environment=dict(platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),torch=torch.__version__,torch_threads=torch.get_num_threads(),device='cpu',dtype='float64'),
        source_sha256={k:digest(HERE/k) for k in ['lazy_prediction.py','test_lazy_prediction.py','benchmark.py']},
        parent_freeze_sha256=digest(PARENT/'FREEZE_SHA256.json'),actual009_trace_sha256=digest(PARENT/'inputs/actual009_trace.npz'),
        timing_method='Three trials after one warm-up, alternating parent/lazy order. Timed interval is the complete step call, including dynamic validation and output copying. Input construction, initialization, prediction-count wrapper setup, and equality assertions are outside the timer. torch.inference_mode is active.',
        cases=results,limitations=['No Spark, CUDA, simulator, observation packing, RSL collection, or optimization timing was measured here.',
            'Each launch-containing batch retains the full original prediction, including all 198 float32 position recurrence steps.',
            'The new launch.any boolean entails a device synchronization if used on CUDA; actual GPU savings are unmeasured.',
            'Asynchronous episodes can create launches on more control steps than a synchronized episode; speedups are workload-specific.',
            'New implementation identity is intentionally incompatible with the frozen observation/source binding; explicit reviewed consumer integration is still required.'],
        physical_admission=False,policy_training_allowed=False,gpu_launches=0)
    (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (HERE/'step.patch').write_text(''.join(difflib.unified_diff(OLD.splitlines(True),NEW.splitlines(True),fromfile='frozen_wave005.BatchWave005.step',tofile='lazy_successor.BatchWave005.step')))
if __name__=='__main__':
    with torch.inference_mode():main()
