"""Ideal measured-motion/contact fixture: geometry and budgets, not physics."""
from pathlib import Path
import json, math, time, hashlib
import numpy as np
import torch
from test_wave_reference import Fixture
from wave_reference import WaveContactReference

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
cases = [('stand', [0., 0., 0.])]
for i in range(16):
    a = 2*math.pi*i/16
    cases.append((f'translation_{i*22.5:g}', [.005*math.cos(a), .005*math.sin(a), 0.]))
cases += [('yaw_left', [0., 0., .015]), ('yaw_right', [0., 0., -.015])]
for a in (0., math.pi/2):
    for w in (-.01, .01):
        cases.append((f'arc_{math.degrees(a):g}_{w:+g}', [.005*math.cos(a), .005*math.sin(a), w]))
rows = []
started = time.monotonic()
for name, command in cases:
    f = Fixture(); r = WaveContactReference(f.names); r.reset(f.snapshot())
    maximum_v = maximum_a = maximum_derating = 0.; min_margin = math.inf
    stop_liftoffs = None; failure = None; actual_commands = []
    for k in range(2000):
        requested = command if 200 <= k < 1400 else [0., 0., 0.]
        if k == 1400: stop_liftoffs = r.liftoffs
        out = r.step(f.snapshot(), requested)
        if not out['valid'][0]:
            failure = dict(step=k, reason=out['failure_reason']); break
        maximum_v = max(maximum_v, float(abs(out['v_ref']).max()))
        maximum_a = max(maximum_a, float(abs(out['a_ref']).max()))
        maximum_derating = max(maximum_derating, 1.-out['command_derating_factor'])
        min_margin = min(min_margin, out['diagnostics']['measured_projected_COM_support_margin_m'])
        actual_commands.append(out['actual_measured_command'])
        f.advance(out)
    latency = None
    if r.stop_requested_time is not None and r.reference_quiet_time is not None:
        latency = r.reference_quiet_time-r.stop_requested_time
    row = dict(name=name, requested_command=command, synthetic_fixture_pass=failure is None,
        failure=failure, control_steps=k+1, maximum_reference_velocity_rad_s=maximum_v,
        maximum_reference_acceleration_rad_s2=maximum_a, minimum_measured_support_geometry_margin_m=min_margin,
        maximum_target_derating_fraction=maximum_derating, planned_liftoffs=r.liftoffs,
        confirmed_synthetic_touchdowns=r.touchdowns,
        liftoffs_after_stop=None if stop_liftoffs is None else r.liftoffs-stop_liftoffs,
        finite_reference_stop_latency_s=latency, final_reference_mode=r.mode,
        final_admitted_command=r.command.tolist(), physics_qualified=False)
    rows.append(row)
    print(name, 'PASS' if failure is None else failure, 'v', round(maximum_v, 4), 'a', round(maximum_a, 4),
          'touchdowns', r.touchdowns, 'stop', latency, flush=True)
report = dict(kind='ideal_measured_motion_contact_fixture_only', physics_qualified=False,
    actual_GPU_runs=0, cases=rows, passed=sum(x['synthetic_fixture_pass'] for x in rows),
    cpu_seconds=time.monotonic()-started,
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))},
    limits=['Fixture moves its synthetic body exactly along virtual desired pose; actual physics must independently prove body tracking.',
            'Contacts are synthesized from reference toe height; real contact sensors must confirm flight and touchdown.',
            'No torque, collision, force distribution, manufacturing or terrain admission follows from this report.'])
(root/'cpu_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
