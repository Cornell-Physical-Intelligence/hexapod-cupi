#!/usr/bin/env python3
"""Bounded fixture-only Isaac validation using the same shared-GPU ownership rules."""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from experiments.c_length_study.tools.launch_length_training_spark import live_competitors, save, verified_source
from experiments.c_length_study.tools.launch_length_study_spark import preflight, resources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'))
    a=p.parse_args();a.source=a.source.resolve();a.output=a.output.resolve()
    if a.output.exists():p.error('Use a fresh output directory')
    locks=[];process=None;identity=None;report=None;stop=False
    def requested(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
    try:
        for path in ('/opt/wx/gpu.lock','/tmp/hexapod-isaac-gpu.lock'):
            fd=os.open(path,os.O_RDONLY);locks.append(fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot=preflight();verified_source(a.source);a.output.mkdir(parents=True)
        name='hexapod-terrain-smoke-'+uuid.uuid4().hex[:12]
        report={'status':'starting','source':str(a.source),'preflight':snapshot,'container_name':name,
                'started_unix':time.time(),'scope':'terrain geometry validation; no PPO'}
        save(a.output/'launcher.json',report)
        cli=['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml',
             '--profile','base','run','--rm','--no-deps','--name',name,'-w','/outputs',
             '-e','PYTHONDONTWRITEBYTECODE=1',
             '-e','PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/terrain-source/tools:/workspace/terrain-source/isaaclab',
             '-v',f'{a.source}:/workspace/terrain-source:rw','-v',f'{a.output}:/outputs:rw',
             '--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base',
             '/workspace/terrain-source/experiments/terrain/tools/validate_terrain_fixtures.py',
             '--catalog','/workspace/terrain-source/artifacts/terrain_readiness_2026-09-09/terrain_catalog.json',
             '--output','/outputs/fixtures','--headless','--device','cuda:0','--info',
             '--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry']
        deadline=time.monotonic()+300
        with (a.output/'run.log').open('w') as log:
            process=subprocess.Popen(cli,cwd=a.isaaclab,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if identity is None:
                    found=subprocess.run(['docker','inspect','--format','{{.Id}}',name],text=True,capture_output=True,timeout=20)
                    if found.returncode==0:
                        identity=found.stdout.strip();report.update(status='running',container_id=identity);save(a.output/'launcher.json',report)
                if stop or (a.output/'stop.request').exists():raise InterruptedError('Stop requested')
                if time.monotonic()>deadline:raise TimeoutError('Fixture smoke exceeded5minutes')
                processes,available=resources()
                if available<16*1024**3:raise MemoryError('Less than16GiB available')
                if identity:
                    top=subprocess.run(['docker','top',identity,'-eo','pid'],text=True,capture_output=True,timeout=20)
                    if top.returncode==0 and live_competitors(processes,{v.strip() for v in top.stdout.splitlines()[1:]},identity):
                        raise RuntimeError('Competing CUDA process; stopping only fixture smoke')
                time.sleep(5)
        path=a.output/'fixtures/validation.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if process.returncode or state.get('status')!='completed' or not state.get('all_fixture_smokes_passed'):
            raise RuntimeError('Fixture smoke did not pass; inspect state and run.log')
        report.update(status='completed',finished_unix=time.time());save(a.output/'launcher.json',report)
    except Exception as exc:
        if report is not None:
            report.update(status='failed',error=repr(exc),finished_unix=time.time());save(a.output/'launcher.json',report)
        raise
    finally:
        if identity:
            running=subprocess.run(['docker','inspect','--format','{{.State.Running}}',identity],text=True,capture_output=True,timeout=20)
            if running.returncode==0 and running.stdout.strip()=='true':
                subprocess.run(['docker','stop','--time','20',identity],timeout=30,capture_output=True)
        if process is not None and process.poll() is None:
            if not identity:process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for fd in reversed(locks):os.close(fd)


if __name__=='__main__':main()
