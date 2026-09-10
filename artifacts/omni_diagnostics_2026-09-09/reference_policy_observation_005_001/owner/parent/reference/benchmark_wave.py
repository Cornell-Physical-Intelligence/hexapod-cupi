"""Like-for-like local CPU step timing; not Spark, GPU or PPO throughput."""
from copy import deepcopy
import cProfile
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time
import torch
from test_batch_wave import Fixture, WaveContactReference, pack, commanded, assert_row
from batch_wave import BatchWave004

HERE=Path(__file__).resolve().parent
REPEATS=7


def templates():
    f=Fixture();r=WaveContactReference(f.names);r.reset(f.snapshot())
    b=BatchWave004(f.names,1);b.reset(pack([f.snapshot()]),torch.tensor([True]),torch.tensor([0]))
    result={'stance':(deepcopy(r),deepcopy(f.snapshot()),{k:v.clone() for k,v in b.s.items()},[0.,0.,0.])}
    for k in range(135):
        if r.mode=='swing' and r.time>.8 and 'swing' not in result:
            result['swing']=(deepcopy(r),deepcopy(f.snapshot()),{k:v.clone() for k,v in b.s.items()},[.005,0.,0.])
        if r.mode=='landing_blend':
            result['landing']=(deepcopy(r),deepcopy(f.snapshot()),{k:v.clone() for k,v in b.s.items()},[.005,0.,0.]);break
        snap=f.snapshot();old=r.step(snap,[.005,0,0]);new=b.step(pack([snap]),commanded([[.005,0,0]]))
        if not old['valid'][0] or not new['valid'][0]:raise RuntimeError('Invalid synthetic fixture')
        torch.testing.assert_close(new['q_ref'],torch.as_tensor(old['q_ref']),atol=1e-11,rtol=0)
        f.advance(old)
    if len(result)!=3:raise RuntimeError('Missing representative phase')
    return result


def main():
    path=HERE/'TIMING_REPORT.json'
    if path.exists():raise FileExistsError('Use a successor timing report')
    torch.set_num_threads(1);base=templates();report=[]
    for n in (1,32,128):
        selected=[base[('stance','swing','landing')[i%3]] for i in range(n)]
        snaps=pack([r[1] for r in selected]);commands=commanded([r[3] for r in selected])
        pristine=BatchWave004(selected[0][0].names,n)
        for k in pristine.s:pristine.s[k].copy_(torch.cat([r[2][k] for r in selected]))
        def prepare_scalar():return [(deepcopy(r),deepcopy(s),list(c)) for r,s,_,c in selected]
        def scalar_run(rows):return [r.step(s,c) for r,s,c in rows]
        def prepare_batch():return deepcopy(pristine)
        def batch_run(b):return b.step(snaps,commands)
        scalar_result=scalar_run(prepare_scalar());batch_result=batch_run(prepare_batch())
        maxq=max(float((batch_result['q_ref'][i]-torch.tensor(r['q_ref'][0])).abs().max()) for i,r in enumerate(scalar_result))
        if maxq>1e-10 or not batch_result['valid'].all():raise ValueError('Timing candidates disagree')
        samples={}
        for name,setup,fn in [('frozen_scalar',prepare_scalar,scalar_run),('batched',prepare_batch,batch_run)]:
            fn(setup());times=[]
            for _ in range(REPEATS):
                workload=setup();start=time.perf_counter();fn(workload);times.append(time.perf_counter()-start)
            samples[name]=dict(seconds=times,median_s=statistics.median(times))
        profile=cProfile.Profile();workload=prepare_batch();profile.enable();batch_run(workload);profile.disable()
        output=io.StringIO();pstats.Stats(profile,stream=output).strip_dirs().sort_stats('cumulative').print_stats(20)
        (HERE/f'cprofile_batch_{n}.txt').write_text(output.getvalue())
        row=dict(replicas=n,phases=('stance','swing','landing'),**samples,q_max_error_rad=maxq,
                 speedup=samples['frozen_scalar']['median_s']/samples['batched']['median_s'])
        report.append(row);print(n,samples['frozen_scalar']['median_s']*1000,samples['batched']['median_s']*1000,flush=True)
    path.write_text(json.dumps(dict(rows=report,local_platform=platform.platform(),python=sys.version,
        torch_version=torch.__version__,torch_threads=1,dtype='float64_with_explicit_legacy_position_rounding',device='cpu',repeats=REPEATS,
        source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('batch_wave.py','benchmark_wave.py','test_batch_wave.py','source_contract.json')},
        scope='One complete reference step, including dynamic validation and typed state output; setup/cloning/CPU fixture packing excluded',
        physics_and_observation_and_PPO_included=False,Spark_or_GPU_measured=False,
        notes=['Same mixed phase states and commands for both implementations.',
               'Batched output clones its full typed state; scalar output builds its normal state dictionary.',
               'The legacy198-step position-rounding prediction scan is included for every row, includingquiet rows.',
               'No physical walking or policy admission is inferred.']),indent=2)+'\n')


if __name__=='__main__':main()
