from pathlib import Path
p=Path('tmp/omni_velocity_launch_003/launch_guarded_remote.py').read_text()
p=p.replace('omni_velocity_probe_003\'','omni_velocity_recording_004\'').replace('forecast_pause_021','forecast_pause_023').replace('velocity-probe-003','velocity-recording-004').replace('restore-021','restore-023')
p=p.replace("    sys.path.insert(0, str(SOURCE / 'tools'))\n    from launch_velocity_probe_spark import check_source\n    from launch_length_study_spark import preflight\n    check_source(SOURCE)","""    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from launch_recording_spark import validate_inputs
    sys.path.insert(0, str(SOURCE / 'tools'))
    from launch_length_study_spark import preflight
    validate_inputs(SimpleNamespace(source=SOURCE,probe=BASE/'omni_velocity_probe_003',
        pilot=BASE/'omni_velocity_pilot_003',adapter=BASE/'omni_velocity_recording_adapter_004'))
    previous = call(['systemctl','--user','show','hexapod-omni-velocity-pilot-003-20260910.service','-p','ActiveState'])
    if 'ActiveState=active' in previous or 'ActiveState=activating' in previous:
        raise RuntimeError('Pilot owner has not exited; do not overlap recording')
    if not (BASE/'forecast_pause_022/restored.json').is_file():
        raise RuntimeError('Pilot forecasting restoration not yet recorded')""")
p=p.replace("reason='Fresh candidate standing and exploration calibration, then at most two stand-only PPO schema updates; no walking training'", "reason='Bounded actual final50-update PPO recording with labels; no training or qualification'")
p=p.replace('--on-active=30m','--on-active=20m').replace('RuntimeMaxSec=1320','RuntimeMaxSec=720')
p=p.replace("str(SOURCE / 'tools/launch_velocity_probe_spark.py'),\n            '--source', str(SOURCE), '--output', str(OUTPUT)","str(Path(__file__).resolve().parent / 'launch_recording_spark.py'),\n            '--source', str(SOURCE), '--probe', str(BASE/'omni_velocity_probe_003'),\n            '--pilot', str(BASE/'omni_velocity_pilot_003'), '--adapter', str(BASE/'omni_velocity_recording_adapter_004'),\n            '--output', str(OUTPUT)")
Path('tmp/omni_velocity_recording_launch_004/launch_guarded_remote.py').write_text(p)
