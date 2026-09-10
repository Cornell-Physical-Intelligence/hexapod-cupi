"""Local CPU timing, including dynamic masks but excluding static preparation.

No Spark, CUDA, Isaac, contact-FSM, full-observation or PPO throughput claim.
"""
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import numpy as np
import torch
from fixtures import rows, pack_state
from tensor_kernel import TensorGeometry
from state_geometry import extract_geometry_state

HERE=Path(__file__).resolve().parent
REPEATS=7


def timing(fn):
    fn()  # Warm numerical kernels; static construction already completed.
    samples=[]
    for _ in range(REPEATS):
        start=time.perf_counter();fn();samples.append(time.perf_counter()-start)
    return dict(seconds=samples,median_s=statistics.median(samples))


def main():
    output=HERE/'TIMING_REPORT.json'
    if output.exists():raise FileExistsError('Use a successor directory; timings are immutable')
    torch.set_num_threads(1)
    ref,examples=rows('wave003_7mm')
    g=TensorGeometry(ref.names,binding='wave003_7mm',profile='formal_004')
    report=[]
    for n in (1,32,128,1024):
        chosen=[examples[i%len(examples)] for i in range(n)]
        state=pack_state(chosen,g)
        q=state['target_position_rad']; qleg=g.leg(q)
        p=state['position_world_m']; R=state['rotation_world_from_body']
        scalar_q=qleg.numpy();scalar_p=p.numpy();scalar_R=R.numpy()
        def scalar_geometry():
            values=[]
            for i in range(n):
                foot,J,transforms=ref.g.fk(qleg[i])
                ik=ref.g.ik(foot)
                com=ref._com(scalar_q[i],scalar_p[i],scalar_R[i])
                values.append((foot,J,ik,com))
            return values
        def batch_geometry():
            forward=g.fk(qleg)
            return forward,g.ik(forward['feet_body_m']),g.com(q,p,R)
        scalar=scalar_geometry();batched=batch_geometry()
        error=0.
        for i,old in enumerate(scalar):
            error=max(error,float((old[2]['q_checked']-batched[1]['q_checked'][i]).abs().max()))
            np.testing.assert_allclose(old[3],batched[2]['world_m'][i].numpy(),atol=1e-12,rtol=0)
            torch.testing.assert_close(old[2]['valid'],batched[1]['valid'][i],atol=0,rtol=0)
        old=timing(scalar_geometry);new=timing(batch_geometry)
        # Same NEW block serial vs batched, not old full740 encoder vs partial233.
        individual=[{key:value[i:i+1] for key,value in state.items()} for i in range(n)]
        serial_state=timing(lambda:[extract_geometry_state(g,item) for item in individual])
        batch_state=timing(lambda:extract_geometry_state(g,state))
        report.append(dict(replicas=n,phase_mixture=sorted({r['mode'] for r in chosen}),
            frozen_scalar_FK_IK_COM=old,batch_FK_IK_COM=new,
            geometry_speedup=old['median_s']/new['median_s'],q_max_error_rad=error,valid_masks_identical=True,
            new233block_serial=serial_state,new233block_batched=batch_state,
            geometry_state_speedup=serial_state['median_s']/batch_state['median_s']))
        print(n,'geometry ms',old['median_s']*1000,new['median_s']*1000,
              'partial state ms',serial_state['median_s']*1000,batch_state['median_s']*1000,flush=True)
    output.write_text(json.dumps(dict(rows=report,repeats=REPEATS,local_platform=platform.platform(),
        python=sys.version,torch_version=torch.__version__,threads=torch.get_num_threads(),dtype='float64',device='cpu',
        binding=g.identity,sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in ('tensor_kernel.py','state_geometry.py','fixtures.py','benchmark.py','source_contract.json')},
        setup_and_transfer_cost_excluded=True,all_dynamic_numeric_masks_included=True,
        no_contact_state_machine=True,no_full_actor=True,no_GPU_test=True,no_PPO_throughput_claim=True,
        notes=['Original frozen scalar functions are the geometry timing baseline.',
               'The233-value state block serial baseline is this new kernel one replica at a time; it is not the full740 encoder.',
               'Local arm64CPU timing cannot predict SparkCPU or GPU timing.',
               'Vectorized reference/contact FSM, full encoder/history, device integration and physics remain unimplemented.']),indent=2)+'\n')


if __name__=='__main__':main()
