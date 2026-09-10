"""Local CPU timing only: no Isaac, GPU, remote connection or physical admission."""
import argparse
import cProfile
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SIZES=(1,32,128)
REPEATS=7


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def reference_templates():
    source=ROOT/'tmp/omni_reference_wave_003'
    if sha(source/'wave_reference.py')!='2f9c6e5ef5119b299a5e2983bfceff666fd9b3ff312c1156ac909221c01fde8d':
        raise ValueError('Wrong current frozen reference')
    sys.path.insert(0,str(source))
    import torch
    torch.set_num_threads(1)
    from test_wave_reference import Fixture
    from wave_reference import WaveContactReference
    f=Fixture();r=WaveContactReference(f.names);r.reset(f.snapshot())
    templates={'stance':(deepcopy(r),deepcopy(f.snapshot()),[0.,0.,0.])}
    for k in range(180):
        if r.mode=='swing' and .8<r.time<1.2 and 'swing' not in templates:
            templates['swing']=(deepcopy(r),deepcopy(f.snapshot()),[.005,0.,0.])
        if r.mode=='landing_blend' and 'landing' not in templates:
            templates['landing']=(deepcopy(r),deepcopy(f.snapshot()),[.005,0.,0.]);break
        out=r.step(f.snapshot(),[.005,0.,0.])
        if not out['valid'][0]:raise RuntimeError(out['failure_reason'])
        f.advance(out)
    if set(templates)!=set(('stance','swing','landing')):raise RuntimeError('Missing representative reference phase')
    def setup(phase,n):
        r,snapshot,command=templates[phase]
        return [(deepcopy(r),deepcopy(snapshot),list(command)) for _ in range(n)]
    def run(rows):
        outputs=[]
        for r,snapshot,command in rows:
            out=r.step(snapshot,command)
            if not out['valid'][0]:raise RuntimeError(out['failure_reason'])
            outputs.append(out)
        return outputs
    identity=dict(source_sha256=sha(source/'wave_reference.py'),source_freeze_sha256=sha(source/'FREEZE_SHA256.json'),
        torch_version=torch.__version__,torch_threads=torch.get_num_threads(),configuration='frozen wave003 default7mm',
        phase_modes={k:v[0].mode for k,v in templates.items()},
        phase_times_s={k:v[0].time for k,v in templates.items()},
        synthetic_inputs=True,measurement='Single real controller step per independent replica; setup/deepcopy excluded')
    return setup,run,identity,templates


def observation_templates():
    source=ROOT/'tmp/reference_policy_observation_002';wave=ROOT/'tmp/omni_reference_wave_002'
    if sha(source/'FREEZE_SHA256.json')!='19c8a6d2f027c007dcd1455e843f3cd9110f2912e7678249698f3ee6a394725d':
        raise ValueError('Wrong frozen observation builder')
    sys.path.insert(0,str(source));sys.path.insert(0,str(wave))
    import torch
    torch.set_num_threads(1)
    from observation import ObservationBuilder,SOURCE_CONTRACT
    from build_fixtures import packet
    from test_landing import ActualPrefix
    fixtures=json.loads((source/'fixtures.json').read_text())
    p=ActualPrefix()
    for k in range(199,273):out=p.r.step(p.snapshot(k),[.005,0,0])
    templates={'stance':fixtures['standing'],'swing':packet(p.snapshot(273),out,'simulator_truth_instrumented'),
               'landing':fixtures['landing']}
    def setup(phase,n):
        packets=[deepcopy(templates[phase]) for _ in range(n)]
        b=ObservationBuilder(packets[0]['joint_names_runtime'],SOURCE_CONTRACT['nominal_joint_positions'],n)
        b.reset(list(range(n)),[p['episode_id'] for p in packets],[0.]*n)
        return b,packets
    def run(prepared):
        b,packets=prepared
        out=b.build(packets)
        if out['policy'].shape!=(len(packets),740):raise RuntimeError('Observation width changed')
        return out
    identity=dict(source_sha256=sha(source/'observation.py'),source_freeze_sha256=sha(source/'FREEZE_SHA256.json'),
        bound_reference='wave0025mm; deliberately separate from profiled wave0037mm',
        actor_width=740,critic_width=743,
        measurement='First validated build per independent replica; fixture/setup/copies excluded; actual validation, signatures and schema included')
    return setup,run,identity,None


def profile_top(setup,run,phase):
    workload=setup(phase,32)
    prof=cProfile.Profile();prof.enable();run(workload);prof.disable()
    output=io.StringIO();pstats.Stats(prof,stream=output).strip_dirs().sort_stats('cumulative').print_stats(20)
    return output.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--kind',choices=('reference','observation'),required=True)
    args=parser.parse_args();target=HERE/(args.kind+'_report.json')
    if target.exists():raise FileExistsError('Use a fresh version; timing reports are immutable')
    setup,run,identity,templates=reference_templates() if args.kind=='reference' else observation_templates()
    for phase in ('stance','swing','landing'):run(setup(phase,1))
    rows=[]
    for phase in ('stance','swing','landing'):
        for n in SIZES:
            elapsed=[];cpu=[]
            for _ in range(REPEATS):
                prepared=setup(phase,n)
                start_cpu=time.process_time();start=time.perf_counter();result=run(prepared)
                elapsed.append(time.perf_counter()-start);cpu.append(time.process_time()-start_cpu)
                del result,prepared
            row=dict(phase=phase,replicas=n,repeats=REPEATS,
                wall_seconds=elapsed,cpu_seconds=cpu,median_wall_s=float(np.median(elapsed)),
                p95_wall_s=float(np.percentile(elapsed,95)),median_per_env_ms=float(np.median(elapsed)*1000/n),
                maximum_serial_control_hz=float(1/np.median(elapsed)))
            rows.append(row);print(args.kind,phase,n,'median',round(row['median_wall_s']*1000,3),'ms',flush=True)
        (HERE/f'{args.kind}_{phase}_cprofile.txt').write_text(profile_top(setup,run,phase))
    report=dict(kind=args.kind,identity=identity,rows=rows,
        local_platform=platform.platform(),machine=platform.machine(),logical_cpus=os.cpu_count(),
        python=sys.version,numpy_version=np.__version__,timing_cpu_only=True,GPU_calls=0,physical_admission=False,
        notes=['Sequential independent instances; no multiprocessing or GPU was used.',
               'Clone/setup and input construction are excluded; these costs must be added in a real integration.',
               'These local CPU timings are not a Spark throughput prediction or training ETA.'])
    target.write_text(json.dumps(report,indent=2)+'\n')
    if templates is not None:
        # The existing geometry API already supports a batch dimension. Prove
        # numerical equivalence; this does not vectorize the contact state machine.
        import torch
        g=templates['swing'][0].g;q=templates['swing'][0].q
        geometry=[]
        for n in SIZES:
            batch=torch.as_tensor(np.repeat(q[None],n,axis=0),dtype=torch.float64)
            target_points=g.fk(batch)[0]
            serial_outputs=[];serial_times=[];batch_times=[]
            for _ in range(REPEATS):
                start=time.perf_counter();serial_outputs=[g.ik(target_points[i:i+1]) for i in range(n)];serial_times.append(time.perf_counter()-start)
                start=time.perf_counter();batched=g.ik(target_points);batch_times.append(time.perf_counter()-start)
            error=float((batched['q_checked']-torch.cat([v['q_checked'] for v in serial_outputs])).abs().max())
            if error>1e-12 or not torch.equal(batched['valid'],torch.cat([v['valid'] for v in serial_outputs])):
                raise ValueError('Batched existing geometry differs from individual evaluations')
            geometry.append(dict(replicas=n,serial_median_s=float(np.median(serial_times)),batch_median_s=float(np.median(batch_times)),
                speedup=float(np.median(serial_times)/np.median(batch_times)),q_max_error_rad=error,valid_masks_identical=True))
        (HERE/'existing_geometry_batch_probe.json').write_text(json.dumps(dict(rows=geometry,
            scope='Unmodified frozen geometry batch dimension only; no vectorized controller or state-machine claim'),indent=2)+'\n')


if __name__=='__main__':main()
