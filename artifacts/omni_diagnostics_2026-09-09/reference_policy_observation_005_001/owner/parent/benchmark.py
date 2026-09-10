"""Local CPU-only measured packing and encoding; no Spark throughput inference."""
from pathlib import Path
import json
import platform
import statistics
import time
import torch
from fixtures import Rig
from test_device_pack import sdk_fixture,capture
from device_pack import DeviceTelemetry

def timed(fn,repeats=25):
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();samples.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples),samples=len(samples))

def main():
    torch.set_num_threads(1);rows=[]
    for n in (1,32,128):
        env,layout,toes=sdk_fixture(n);adapter=DeviceTelemetry(env,layout,toes)
        rig=Rig(n);packet=rig.packet();rig.builder.build(packet)
        def encode():
            packet['step_indices']+=1
            packet['measurement']['time_s']=packet['step_indices'].to(torch.float64)*.02
            packet['reference']['state']['time']=packet['measurement']['time_s'].clone()
            return rig.builder.build(packet)
        for _ in range(3):capture(adapter,env);encode()
        row=dict(environments=n,device_pack=timed(lambda:capture(adapter,env)),observation_and_history=timed(encode))
        if not rig.builder.ready.all():raise RuntimeError('Benchmark encoded invalid rows')
        rows.append(row)
    report=dict(platform=platform.platform(),torch_version=torch.__version__,torch_threads=1,device='cpu',dtype='float64',
        scope='Complete validated synthetic SDK packing separately from typed reference/residual encoding and nonduplicate history update; measured local only',
        excluded='Physics, reference state machine, PPO, actual Isaac sensor timestamp integration, GPU, setup',rows=rows)
    Path(__file__).with_name('TIMING_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
